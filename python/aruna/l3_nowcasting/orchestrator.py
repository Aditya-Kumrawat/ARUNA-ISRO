"""
NowcastOrchestrator — wires RollingP85Baseline, ChangepointDetector,
ConsensusEngine, and CatalogService into a single frame-by-frame processor.
Not thread-safe; intended to be driven by a single Kafka consumer thread.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from aruna.l2_physics.models import PhysicsFrame
from aruna.models.nowcast import NowcastEvent
from .baseline import RollingP85Baseline
from .catalog import CatalogService
from .changepoint import ChangepointDetector
from .consensus import ConsensusEngine
from .models import NowcastConfig

logger = logging.getLogger(__name__)


def _isnan(v: float) -> bool:
    return v != v


class NowcastOrchestrator:
    """Processes PhysicsFrame objects one at a time. Not thread-safe."""

    def __init__(
        self,
        config: Optional[NowcastConfig] = None,
        kafka_producer=None,
    ) -> None:
        self._cfg = config or NowcastConfig()
        self._pub = kafka_producer
        self._frame_count: int = 0

        self._sol_baseline = RollingP85Baseline(
            window_sec=self._cfg.baseline_window_sec,
            percentile=self._cfg.baseline_percentile,
            recompute_interval=self._cfg.baseline_recompute_interval,
            sigma_multiplier=self._cfg.trigger_sigma_multiplier,
        )
        self._hel_baseline = RollingP85Baseline(
            window_sec=self._cfg.baseline_window_sec,
            percentile=self._cfg.baseline_percentile,
            recompute_interval=self._cfg.baseline_recompute_interval,
            sigma_multiplier=self._cfg.trigger_sigma_multiplier,
        )
        self._sol_cp = ChangepointDetector(
            flux_window=self._cfg.pelt_flux_window,
            penalty=self._cfg.pelt_penalty,
            min_size=self._cfg.pelt_min_size,
            model=self._cfg.pelt_model,
        )
        self._hel_cp = ChangepointDetector(
            flux_window=self._cfg.pelt_flux_window,
            penalty=self._cfg.pelt_penalty,
            min_size=self._cfg.pelt_min_size,
            model=self._cfg.pelt_model,
        )
        self._consensus = ConsensusEngine(self._cfg)
        self._catalog = CatalogService(self._cfg.db_url)

    def process(self, frame: PhysicsFrame) -> Optional[NowcastEvent]:
        self._frame_count += 1
        t_sec = self._frame_count

        flux_sol = frame.solexs_3_6kev + frame.solexs_6_10kev
        flux_hel = frame.hel1os_10_25kev + frame.hel1os_25_50kev + frame.hel1os_50_70kev

        _, sol_threshold = self._sol_baseline.update(flux_sol, t_sec)
        _, hel_threshold = self._hel_baseline.update(flux_hel, t_sec)

        sol_cp = self._sol_cp.update(flux_sol)
        self._hel_cp.update(flux_hel)  # advances internal buffer; SHEI hardening is the HEL trigger proxy

        sol_triggered = (
            sol_threshold is not None
            and flux_sol > sol_threshold
            and sol_cp is not None
        )
        hel_triggered = (
            hel_threshold is not None
            and flux_hel > hel_threshold
            and frame.dSHEI_dt > 0.0
            and frame.SHEI > -1.0
        )

        event = self._consensus.ingest(
            sol_triggered=sol_triggered,
            hel_triggered=hel_triggered,
            hope_flag=frame.hope_flag,
            hope_reason=frame.hope_reason,
            physics_frame=frame,
            flux_solexs=flux_sol,
            flux_hel1os=flux_hel,
        )

        if event is not None:
            # Freeze baselines only on confirmed DUAL_TRIGGER, not SOLEXS_ONLY advisory
            if event.alert_type == "FLARE_ONSET" and event.detection_method == "DUAL_TRIGGER":
                self._sol_baseline.freeze()
                self._hel_baseline.freeze()
            elif event.alert_type == "FLARE_END":
                # Clear the rolling window before unfreezing so the flare's
                # elevated flux does not inflate the threshold on the next
                # quiet period, which would suppress detection of future events.
                self._sol_baseline.clear()
                self._hel_baseline.clear()
                self._sol_baseline.unfreeze()
                self._hel_baseline.unfreeze()

            self._catalog.write(
                event,
                peak_T_MK=frame.T_MK if not _isnan(frame.T_MK) else 0.0,
                peak_EM_cm3=frame.EM_cm3 if not _isnan(frame.EM_cm3) else 0.0,
                peak_flux_solexs=flux_sol,
                peak_flux_hel1os=flux_hel,
                peak_DNI=frame.DNI,
            )
            if self._pub is not None:
                self._publish(event)

            logger.info(
                "NowcastEvent %s %s%s conf=%.2f",
                event.alert_type,
                event.goes_class,
                event.goes_subclass,
                event.confidence,
            )
        return event

    def _publish(self, event: NowcastEvent) -> None:
        payload = {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in event.__dict__.items()
        }
        try:
            self._pub.send(
                self._cfg.kafka_topic_output,
                key=b"nowcast",
                value=json.dumps(payload).encode(),
            )
        except Exception as exc:
            logger.error("Kafka publish failed: %s", exc)

    def close(self) -> None:
        self._catalog.close()
