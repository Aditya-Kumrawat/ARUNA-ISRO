"""PhysicsEngine orchestrator — wires all L2 sub-modules into a stateful pipeline.

One instance per Kafka consumer/partition. Not thread-safe.
Produces one PhysicsFrame per paired (SoLEXS, HEL1OS) ObservationFrame.
"""
from __future__ import annotations
import logging
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from aruna.models.observation import ObservationFrame, QualityFlag
from .chianti_table import CHIANTILookupTable
from .dni    import DNIComputer
from .feka   import FeKaExtractor
from .hfd    import HFDComputer
from .hope   import HOPEEvaluator
from .isothermal import IsothermalFitter
from .models import (EngineState, PhysicsConfig, PhysicsFrame,
                     PhysicsQuality, HOPEResult)
from .qpp    import QPPDetector
from .shei   import SHEIComputer
from .tnpsv  import TNPSVComputer

logger = logging.getLogger(__name__)
_VERSION = "2.0.0"


def _total_flux(frame: ObservationFrame) -> float:
    return float(np.sum(frame.counts_per_s_kev))


def _band(frame: ObservationFrame, lo: float, hi: float) -> float:
    return frame.band_flux(lo, hi)


def _assign_phase(
    DNI: float, SHEI: float, dT_dt: float, hope_flag: bool, cfg: PhysicsConfig
) -> str:
    if DNI > cfg.hope_dni_threshold and SHEI > -0.8:
        return "IMPULSIVE"
    if hope_flag:
        return "HOPE"
    if SHEI < -1.5 and dT_dt < 0:
        return "DECAY"
    if dT_dt > 0 or DNI > 0.2:
        return "GRADUAL"
    return "QUIET"


class PhysicsEngine:
    """
    Stateful orchestrator for the L2 Physics Engine.

    Wires IsothermalFitter, DNIComputer, TNPSVComputer, SHEIComputer,
    HFDComputer, QPPDetector, FeKaExtractor, and HOPEEvaluator into a
    1-second cadence pipeline. Publishes PhysicsFrame to Kafka when a
    producer is supplied.

    Not thread-safe. One instance per Kafka consumer/partition.
    """

    def __init__(self, config: Optional[PhysicsConfig] = None, kafka_producer=None) -> None:
        self._cfg    = config or PhysicsConfig()
        self._pub    = kafka_producer
        self._fitter = IsothermalFitter()
        self._dni    = DNIComputer()
        self._tnpsv  = TNPSVComputer(self._cfg.tnpsv_history_len)
        self._shei   = SHEIComputer(self._cfg.shei_history_len)
        self._hfd    = HFDComputer(self._cfg.hfd_window)
        self._qpp    = QPPDetector(self._cfg.qpp_window, self._cfg.qpp_min_frames)
        self._feka   = FeKaExtractor()
        self._hope   = HOPEEvaluator()

        self._sol_buf: deque = deque(maxlen=self._cfg.hfd_window)
        self._hel_buf: deque = deque(maxlen=self._cfg.hfd_window)

        self._frames_processed: int  = 0
        self._hope_streak_sec: float = 0.0
        self._prev_T: float          = 0.0
        self._prev_EM: float         = 0.0
        self._prev_dT_dt: float      = 0.0
        self._prev_timestamp: Optional[datetime] = None
        self._hel_quiet_baseline: float = 500.0
        self._non_quiet_count: int = 0
        self._recent_quality: deque = deque(maxlen=self._cfg.degraded_window)
        self._state = EngineState.WARMUP
        self._prev_suit_uv_flux: float = 0.0   # for suit_uv_brightening_rate computation

    @property
    def state(self) -> EngineState:
        return self._state

    def reset(self) -> None:
        """Reset all sub-modules and counters to initial state."""
        self.__init__(self._cfg, self._pub)

    def process(
        self,
        sol: ObservationFrame,
        hel: ObservationFrame,
        suit: Optional[ObservationFrame] = None,
    ) -> PhysicsFrame:
        """Process one paired (SoLEXS, HEL1OS) frame and return a PhysicsFrame.

        Args:
            sol:  Dead-time-corrected SoLEXS ObservationFrame.
            hel:  Dead-time-corrected HEL1OS ObservationFrame.
            suit: Optional SUIT_FD frame resampled to 1s grid. When provided,
                  suit_uv_flux and suit_uv_brightening_rate are populated.

        Returns:
            PhysicsFrame with all L2 observables populated.
        """
        # ── timestamp proximity check ────────────────────────────────────────
        dt_sec = abs((sol.timestamp - hel.timestamp).total_seconds())
        if dt_sec > 1.0:
            logger.warning(
                "Timestamp mismatch sol=%s hel=%s delta=%.2fs",
                sol.timestamp.isoformat(), hel.timestamp.isoformat(), dt_sec,
            )

        self._sol_buf.append(sol)
        self._hel_buf.append(hel)
        self._frames_processed += 1

        # ── isothermal fit ───────────────────────────────────────────────────
        fit = self._fitter.fit(sol.counts_per_s_kev, sol.energy_bins_kev)
        phys_q = self._physics_quality(fit)

        # ── novel observables ────────────────────────────────────────────────
        hel_25_50 = _band(hel, 25.0, 50.0)
        sol_8_15  = _band(sol, 8.0, 15.0)
        DNI = self._dni.update(hel_25_50, sol_8_15)

        tnpsv_mag, tnpsv_angle = (
            self._tnpsv.update(fit.T_MK, fit.EM_cm3)
            if not np.isnan(fit.T_MK) else (0.0, 0.0)
        )
        shei, dshei = self._shei.update(hel.counts_per_s_kev, hel.energy_bins_kev)
        hfd         = self._hfd.update(_total_flux(sol))
        qpp_p, qpp_a = self._qpp.update(hel_25_50)
        feka_flux   = self._feka.extract(sol.counts_per_s_kev, sol.energy_bins_kev,
                                          fit.T_MK, fit.EM_cm3)

        # ── Gradients ────────────────────────────────────────────────────────
        dT_dt = 0.0
        d2T_dt2 = 0.0
        dEM_dt = 0.0

        if not np.isnan(fit.T_MK) and not np.isnan(fit.EM_cm3):
            if self._prev_T > 0.0:
                dT_dt = fit.T_MK - self._prev_T
                d2T_dt2 = dT_dt - self._prev_dT_dt
            self._prev_T = fit.T_MK
            self._prev_dT_dt = dT_dt

            if self._prev_EM > 0.0:
                dEM_dt = fit.EM_cm3 - self._prev_EM
            self._prev_EM = fit.EM_cm3

        # ── HOPE streak (timestamp-based) ────────────────────────────────────
        if self._prev_timestamp is not None:
            elapsed = (sol.timestamp - self._prev_timestamp).total_seconds()
            if dT_dt > 0 and 0 < elapsed <= 1.5:
                self._hope_streak_sec += elapsed
            else:
                self._hope_streak_sec = 0.0
        self._prev_timestamp = sol.timestamp

        # ── HOPE evaluation ──────────────────────────────────────────────────
        hope_res: HOPEResult = self._hope.evaluate(
            T=fit.T_MK if not np.isnan(fit.T_MK) else 0.0,
            dT_dt=dT_dt,
            EM=fit.EM_cm3 if not np.isnan(fit.EM_cm3) else 0.0,
            dEM_dt=dEM_dt,
            DNI=DNI,
            tnpsv_mag=tnpsv_mag,
            tnpsv_angle=tnpsv_angle,
            hel_flux=_total_flux(hel),
            hel_quiet_baseline=self._hel_quiet_baseline,
            streak_sec=self._hope_streak_sec,
            config=self._cfg,
        )

        # ── flare phase ──────────────────────────────────────────────────────
        phase = _assign_phase(DNI, shei, dT_dt, hope_res.hope_flag, self._cfg)

        # ── baseline + DNI reset ─────────────────────────────────────────────
        if phase == "QUIET":
            self._hel_quiet_baseline = (
                0.9 * self._hel_quiet_baseline + 0.1 * _total_flux(hel)
            )
            self._non_quiet_count = 0
        else:
            self._non_quiet_count += 1

        if phase == "QUIET" and self._non_quiet_count >= self._cfg.dni_reset_min_non_quiet:
            self._dni.reset()
            self._non_quiet_count = 0

        # ── engine state ─────────────────────────────────────────────────────
        self._recent_quality.append(phys_q == PhysicsQuality.FIT_OK)
        if self._frames_processed >= self._cfg.hfd_window:
            ok_rate = sum(self._recent_quality) / len(self._recent_quality)
            self._state = (
                EngineState.DEGRADED if ok_rate < (1 - self._cfg.degraded_threshold)
                else EngineState.ACTIVE
            )

        # ── calculate raw flux bands and provenance hash ────────────────────
        solexs_3_6kev = _band(sol, 3.0, 6.0)
        solexs_6_10kev = _band(sol, 6.0, 10.0)
        hel1os_10_25kev = _band(hel, 10.0, 25.0)
        hel1os_25_50kev = hel_25_50
        hel1os_50_70kev = _band(hel, 50.0, 70.0)

        # ── SUIT UV flux + brightening rate ─────────────────────────────────
        suit_uv_flux = 0.0
        suit_uv_brightening_rate = 0.0
        if suit is not None:
            suit_uv_flux = float(np.sum(suit.counts_per_s_kev))
            suit_uv_brightening_rate = suit_uv_flux - self._prev_suit_uv_flux
            self._prev_suit_uv_flux = suit_uv_flux

        # ── TNCI: Thermal-Nonthermal Coupling Index (PRD §13.4) ─────────────
        # TNCI = d/dt[T × EM] / F_HEL1OS
        # Using product rule: d/dt[T×EM] = dT_dt×EM + T×dEM_dt (1-second finite diff)
        hel_total_flux = _total_flux(hel)
        if not np.isnan(fit.T_MK) and not np.isnan(fit.EM_cm3) and hel_total_flux > 1e-30:
            d_thermal_proxy = dT_dt * fit.EM_cm3 + fit.T_MK * dEM_dt
            tnci = d_thermal_proxy / (hel_total_flux + 1e-30)
        else:
            tnci = 0.0

        import hashlib
        combined_checksums = f"{sol.checksum_sha256 or ''}{hel.checksum_sha256 or ''}"
        if suit is not None:
            combined_checksums += (suit.checksum_sha256 or '')
        provenance_hash = hashlib.sha256(combined_checksums.encode()).hexdigest()

        # ── assemble frame ───────────────────────────────────────────────────
        source_ids = [sol.frame_id, hel.frame_id]
        if suit is not None:
            source_ids.append(suit.frame_id)

        pf = PhysicsFrame(
            frame_id=str(uuid.uuid4()),
            timestamp=sol.timestamp,
            source_frame_ids=source_ids,
            T_MK=fit.T_MK,
            T_uncertainty_MK=fit.T_uncertainty_MK,
            EM_cm3=fit.EM_cm3,
            EM_uncertainty=fit.EM_uncertainty,
            fit_chi_squared=fit.chi2_reduced,
            DNI=DNI,
            tnpsv_magnitude=tnpsv_mag,
            tnpsv_angle_rad=tnpsv_angle,
            SHEI=shei,
            dSHEI_dt=dshei,
            HFD=hfd,
            QPP_period_sec=qpp_p,
            QPP_amplitude=qpp_a,
            FeKa_flux=feka_flux,
            flare_phase=phase,
            hope_flag=hope_res.hope_flag,
            hope_lead_time_sec=hope_res.lead_time_sec if hope_res.hope_flag else None,
            hope_reason=hope_res.reason if hope_res.hope_flag else None,
            engine_state=self._state.value,
            data_quality_flag=sol.quality_flag,
            physics_quality_flag=phys_q,
            fit_backend=fit.backend,
            dT_dt=dT_dt,
            dEM_dt=dEM_dt,
            d2T_dt2=d2T_dt2,
            solexs_3_6kev=solexs_3_6kev,
            solexs_6_10kev=solexs_6_10kev,
            hel1os_10_25kev=hel1os_10_25kev,
            hel1os_25_50kev=hel1os_25_50kev,
            hel1os_50_70kev=hel1os_50_70kev,
            suit_uv_flux=suit_uv_flux,
            suit_uv_brightening_rate=suit_uv_brightening_rate,
            tnci=tnci,
            provenance_hash=provenance_hash,
            quality_flag=sol.quality_flag,
            pipeline_version=_VERSION,
        )

        if self._pub:
            self._publish(pf)

        return pf

    def _physics_quality(self, fit) -> PhysicsQuality:
        if np.isnan(fit.T_MK):
            return PhysicsQuality.FIT_DIVERGED
        if fit.backend == "chianti_lookup":
            return PhysicsQuality.BACKEND_FALLBACK
        if fit.chi2_reduced > self._cfg.fit_chi2_low_confidence:
            return PhysicsQuality.LOW_CONFIDENCE
        return PhysicsQuality.FIT_OK

    def _publish(self, pf: PhysicsFrame) -> None:
        import json
        from datetime import datetime as _dt
        payload = {k: (v.isoformat() if isinstance(v, _dt) else
                       v.name if hasattr(v, "name") else v)
                   for k, v in pf.__dict__.items()
                   if not isinstance(v, np.ndarray)}
        payload["schema_version"] = pf.schema_version
        try:
            self._pub.send(
                self._cfg.kafka_topic_physics,
                key=b"physics",
                value=json.dumps(payload).encode(),
            )
        except Exception as exc:
            logger.error("Kafka publish failed: %s", exc)
