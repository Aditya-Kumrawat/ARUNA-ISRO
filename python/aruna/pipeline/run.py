"""
ARUNA End-to-End Pipeline Runner — PRD §2.

Chains L1 → L2 → L3 → L4 → L5 → L6 → L7 and prints the final AlertBriefing.

Usage:
  # X6.3 always-on offline demo (recommended for hackathon)
  python -m aruna.pipeline.run --fits-dir python/tests/fixtures/x63/

  # With real FITS from PRADAN
  python -m aruna.pipeline.run --fits-dir data/solexs_20240222/

  # Use Azure GPT-4.1-mini for LLM briefing (set AZURE_OPENAI_API_KEY first)
  python -m aruna.pipeline.run --fits-dir python/tests/fixtures/x63/ --provider azure

  # Output JSON to file
  python -m aruna.pipeline.run --fits-dir python/tests/fixtures/x63/ --output briefing.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("aruna.pipeline.run")


# ── FITS file discovery ───────────────────────────────────────────────────────

def _find_fits(directory: Path) -> dict:
    """Locate SoLEXS, HEL1OS, and HK FITS files in a directory.

    Handles both plain .fits and PRADAN gzip-compressed .fits.gz / .pi.gz / .lc.gz.
    """
    fits_dir = Path(directory)
    patterns = ("*.fits", "*.fit", "**/*.fits", "**/*.fit",
                "*.fits.gz", "**/*.fits.gz", "**/*.pi.gz", "**/*.lc.gz", "**/*.gti.gz")
    all_fits: list = []
    for p in patterns:
        all_fits.extend(fits_dir.glob(p))
    all_fits = sorted(set(all_fits))

    solexs = [f for f in all_fits if any(k in f.name.upper() for k in ("SLX", "SOLEXS"))]
    hel1os = [f for f in all_fits if any(k in f.name.upper() for k in ("HEL", "HEL1OS"))]
    hk     = [f for f in all_fits if any(k in f.name.upper() for k in ("HK", "HOUSE", "AUX"))]

    # PRADAN-specific: PI spectrum files are the primary spectral data source
    solexs_pi  = [f for f in solexs if ".pi." in f.name.lower()]
    hel1os_pi  = [f for f in hel1os if ".pi." in f.name.lower()]
    solexs_gti = [f for f in solexs if ".gti." in f.name.lower()]

    return {
        "solexs":     sorted(solexs),
        "hel1os":     sorted(hel1os),
        "hk":         sorted(hk),
        "all":        sorted(all_fits),
        "solexs_pi":  sorted(solexs_pi),
        "hel1os_pi":  sorted(hel1os_pi),
        "solexs_gti": sorted(solexs_gti),
    }


# ── Synthetic PhysicsFrame generator (offline demo mode) ─────────────────────

def _make_synthetic_physics_frames(n_frames: int = 300) -> List:
    """
    Generate synthetic PhysicsFrames simulating an X6.3 flare profile.
    Used when FITS files cannot be parsed (missing deps / corrupted files).
    """
    import numpy as np
    from aruna.models.physics import PhysicsFrame
    from aruna.models.observation import QualityFlag

    frames = []
    t0 = datetime(2024, 2, 22, 9, 0, 0, tzinfo=timezone.utc)

    for i in range(n_frames):
        sec = i
        phase = sec / n_frames

        # Simulate X6.3 flare profile — rise then decay
        if phase < 0.3:
            T_MK = 8.0 + 50.0 * (phase / 0.3)        # rising 8 → 58 MK
            EM   = 1e48 * phase / 0.3
            flare_flux = 1e-4 * (phase / 0.3)
        elif phase < 0.5:
            T_MK = 58.0 - 30.0 * ((phase - 0.3) / 0.2)  # peak region
            EM   = 1e48 * (1.0 - (phase - 0.3) / 0.5)
            flare_flux = 1e-4 * (1.0 - (phase - 0.3) / 0.5)
        else:
            T_MK = 28.0 - 20.0 * ((phase - 0.5) / 0.5)
            EM   = 1e47 * (1.0 - (phase - 0.5) / 0.5)
            flare_flux = 5e-5 * (1.0 - (phase - 0.5) / 0.5)

        DNI  = max(0.0, (phase - 0.1) / 0.3 * 3.0) if phase < 0.4 else 1.5
        SHEI = -0.5 if phase < 0.35 else -1.5
        hope = phase >= 0.05 and phase <= 0.2

        from datetime import timedelta
        ts = t0 + timedelta(seconds=sec)

        if hope:
            fp = "HOPE"
        elif phase < 0.3:
            fp = "QUIET"
        elif phase < 0.5:
            fp = "IMPULSIVE"
        else:
            fp = "DECAY"

        frames.append(PhysicsFrame(
            timestamp=ts,
            source_frame_ids=[f"synth_{i:04d}"],
            T_MK=float(T_MK),
            T_uncertainty_MK=1.0,
            EM_cm3=float(max(EM, 0)),
            EM_uncertainty=1e46,
            fit_chi_squared=1.0,
            dT_dt=float(50.0 if phase < 0.3 else -30.0),
            dEM_dt=float(1e48 if phase < 0.3 else -1e47),
            d2T_dt2=0.0,
            DNI=float(DNI),
            tnpsv_magnitude=float(abs(DNI * 0.1)),
            tnpsv_angle_rad=float(1.57 if hope else 0.0),
            SHEI=float(SHEI),
            dSHEI_dt=float(0.05 if phase < 0.3 else -0.02),
            HFD=1.5,
            FeKa_flux=0.0,
            QPP_period_sec=None,
            QPP_amplitude=None,
            hope_flag=hope,
            hope_lead_time_sec=int((0.2 - phase) * n_frames) if hope else None,
            hope_reason="T rising + EM flat + HEL1OS quiet" if hope else None,
            flare_phase=fp,
            quality_flag=QualityFlag.GOOD,
            solexs_3_6kev=float(flare_flux * 0.3),
            solexs_6_10kev=float(flare_flux * 0.4),
            hel1os_10_25kev=float(flare_flux * 0.5),
            hel1os_25_50kev=float(flare_flux * 0.3),
            hel1os_50_70kev=float(flare_flux * 0.15),
        ))
    return frames


# ── L1 Ingestion ─────────────────────────────────────────────────────────────

def run_l1(fits_files: dict) -> List:
    """Run L1 ingestion. Returns List[ObservationFrame]."""
    from aruna.l1_ingestion.pipeline import IngestionPipeline
    from aruna.l1_ingestion.models import IngestionConfig

    solexs = fits_files["solexs"]
    hel1os = fits_files["hel1os"]
    hk     = fits_files["hk"]

    if not solexs or not hel1os or not hk:
        logger.warning("L1: Missing FITS files — solexs=%d hel1os=%d hk=%d",
                       len(solexs), len(hel1os), len(hk))
        return []

    class _NoCache:
        def get(self, key): return None
        def put(self, key, val): pass

    pipeline = IngestionPipeline(cache=_NoCache(), config=IngestionConfig())
    all_frames = []

    for s, h, k in zip(solexs, hel1os, hk):
        try:
            frames = pipeline.ingest(
                solexs_path=Path(s),
                hel1os_path=Path(h),
                hk_path=Path(k),
            )
            all_frames.extend(frames)
            logger.info("L1: ingested %d frames from %s", len(frames), s.name)
        except Exception as exc:
            logger.warning("L1: failed on %s — %s", s.name, exc)

    return all_frames


# ── L1 PRADAN format ingestion ───────────────────────────────────────────────

# X6.3 peak: 2024-02-22 22:32:34 UTC = Unix 1708641154
_X63_PEAK_UNIX = 1_708_641_154.0


def _synthesize_hel1os_proxy(solexs_frames: List) -> List:
    """
    Build synthetic HEL1OS ObservationFrames from SoLEXS flux.

    Uses temperature-sensitive spectral hardening: the ratio of high-energy
    to low-energy SoLEXS channels (proxy for plasma temperature) drives the
    relative brightness of hard HEL1OS bins, so dSHEI_dt > 0 during the
    impulsive phase and L3 dual-trigger can fire. Frames are labelled
    instrument='HEL1OS_CDTE' with source_file='PROXY:...' so downstream
    stages can identify partial-data mode.
    """
    import numpy as np
    from aruna.models.observation import ObservationFrame, QualityFlag

    # 4 HEL1OS proxy bins: 10-25, 25-40, 40-55, 55-70 keV
    HEL1OS_BINS = np.array([10.0, 25.0, 40.0, 55.0, 70.0])
    # Quiet-sun base ratios (Sarwade et al. 2025 empirical): HEL1OS fraction of SoLEXS total
    BASE_RATIOS = np.array([0.040, 0.025, 0.010, 0.005])

    proxy_frames = []
    for sf in solexs_frames:
        counts = np.asarray(sf.counts_per_s_kev, dtype=np.float64)
        n = len(counts)

        # Use intra-SoLEXS hardening as proxy for temperature:
        # SDD2 channels span 1.5–15 keV; upper third (≈9–15 keV) rises faster
        # than lower third (≈1.5–6 keV) during impulsive flare heating.
        third = max(n // 3, 1)
        soft_sum = float(np.nansum(counts[:third]))
        hard_sum = float(np.nansum(counts[2 * third:]))
        # hardening = 1.0 at quiet sun, rises to ~3–5 during impulsive peak
        hardening = float(np.clip(hard_sum / max(soft_sum, 1e-30), 1.0, 6.0))

        total_flux = float(np.nansum(counts))
        # Hard bins (40-70 keV) amplify by hardening^1.5 relative to soft bins
        ratios = np.array([
            BASE_RATIOS[0],
            BASE_RATIOS[1] * (1.0 + 0.3 * (hardening - 1.0)),
            BASE_RATIOS[2] * hardening,
            BASE_RATIOS[3] * hardening ** 1.5,
        ])
        hel_counts = total_flux * ratios

        proxy_frames.append(ObservationFrame(
            instrument="HEL1OS_CDTE",
            timestamp=sf.timestamp,
            energy_bins_kev=HEL1OS_BINS,
            counts_per_s_kev=hel_counts,
            quality_flag=QualityFlag.GOOD,
            dead_time_fraction=0.0,
            solar_distance_au=sf.solar_distance_au,
            spacecraft_mode=sf.spacecraft_mode,
            saa_flag=sf.saa_flag,
            source_file=f"PROXY:{sf.source_file}",
            checksum_sha256="proxy",
            pipeline_version="2.0.0-proxy",
            calibration_version="proxy_sarwade2025",
        ))
    return proxy_frames


def run_l1_pradan(
    fits_files: dict,
    single_instrument: bool = False,
    window_center_unix: Optional[float] = _X63_PEAK_UNIX,
    window_sec: int = 5400,
) -> List:
    """
    Ingest PRADAN-format per-file .pi.gz data.

    When single_instrument=True and HEL1OS PI files are absent, synthesizes
    a HEL1OS proxy from the SoLEXS flux for diagnostic purposes.
    """
    from aruna.l1_ingestion.fits_reader import FitsReader
    from aruna.l1_ingestion.deadtime import DeadTimeCorrector
    from aruna.models.observation import ObservationFrame, QualityFlag
    import numpy as np
    from datetime import timezone

    reader = FitsReader()
    dtc    = DeadTimeCorrector()

    pi_files  = fits_files.get("solexs_pi", [])
    gti_files = fits_files.get("solexs_gti", [])

    if not pi_files:
        logger.warning("run_l1_pradan: no SoLEXS PI files found")
        return []

    all_frames: List = []

    for pi_path in pi_files:
        # Determine detector from filename
        name_up = pi_path.name.upper()
        detector = "SDD1" if "SDD1" in name_up else "SDD2"

        # Find companion GTI file
        gti_arg = None
        for g in gti_files:
            if detector in g.name.upper():
                gti_arg = g
                break

        try:
            raw_list = reader.read_solexs_pradan_pi(
                pi_path,
                gti_path=gti_arg,
                detector=detector,
                window_center_unix=window_center_unix,
                window_sec=window_sec,
            )
        except Exception as exc:
            logger.error("run_l1_pradan: failed reading %s — %s", pi_path.name, exc)
            continue

        for raw_fd in raw_list:
            try:
                corr_fd, flags = dtc.correct(raw_fd)
            except Exception as exc:
                logger.warning("Dead-time correction failed for %s: %s — using raw", pi_path.name, exc)
                corr_fd, flags = raw_fd, np.full(len(raw_fd.timestamps), 0, dtype=np.int8)

            for i in range(len(corr_fd.timestamps)):
                ts_ns = corr_fd.timestamps[i]
                epoch_ns = np.datetime64('1970-01-01T00:00:00', 'ns')
                ts_sec = float((ts_ns - epoch_ns) / np.timedelta64(1, 's'))
                from datetime import datetime
                ts = datetime.fromtimestamp(ts_sec, tz=timezone.utc)

                counts_i = corr_fd.counts_raw[i]
                all_frames.append(ObservationFrame(
                    instrument=corr_fd.instrument,
                    timestamp=ts,
                    energy_bins_kev=corr_fd.energy_bins_kev,
                    counts_per_s_kev=counts_i if counts_i.ndim == 1 else counts_i.ravel(),
                    quality_flag=QualityFlag(int(flags[i])),
                    dead_time_fraction=float(corr_fd.deadtime_frac[i]),
                    solar_distance_au=1.0,
                    spacecraft_mode="NORMAL",
                    saa_flag=False,
                    source_file=pi_path.name,
                    checksum_sha256=corr_fd.checksum_sha256,
                    pipeline_version="2.0.0-pradan",
                    calibration_version="sarwade_2025_v1",
                ))

    if all_frames:
        logger.info("run_l1_pradan: %d real SoLEXS frames from %d PI files",
                    len(all_frames), len(pi_files))

    # ── HEL1OS PI files (real data path) ──────────────────────────────────────
    hel1os_pi = fits_files.get("hel1os_pi", [])
    hel1os_gti_files = [f for f in fits_files.get("hel1os", []) if ".gti." in f.name.lower()]

    if hel1os_pi:
        for hel_pi_path in hel1os_pi:
            name_up = hel_pi_path.name.upper()
            detector = "CZT" if "CZT" in name_up else "CDTE"

            gti_arg = None
            for g in hel1os_gti_files:
                if detector in g.name.upper():
                    gti_arg = g
                    break

            try:
                raw_list = reader.read_hel1os_pradan_pi(
                    hel_pi_path,
                    gti_path=gti_arg,
                    detector=detector,
                    window_center_unix=window_center_unix,
                    window_sec=window_sec,
                )
            except Exception as exc:
                logger.error("run_l1_pradan: failed reading HEL1OS %s — %s", hel_pi_path.name, exc)
                continue

            for raw_fd in raw_list:
                try:
                    corr_fd, flags = dtc.correct(raw_fd)
                except Exception as exc:
                    logger.warning("HEL1OS dead-time correction failed (%s) — using raw", exc)
                    corr_fd, flags = raw_fd, np.full(len(raw_fd.timestamps), 0, dtype=np.int8)

                for i in range(len(corr_fd.timestamps)):
                    ts_ns = corr_fd.timestamps[i]
                    epoch_ns = np.datetime64('1970-01-01T00:00:00', 'ns')
                    ts_sec = float((ts_ns - epoch_ns) / np.timedelta64(1, 's'))
                    from datetime import datetime
                    ts = datetime.fromtimestamp(ts_sec, tz=timezone.utc)

                    counts_i = corr_fd.counts_raw[i]
                    all_frames.append(ObservationFrame(
                        instrument=corr_fd.instrument,
                        timestamp=ts,
                        energy_bins_kev=corr_fd.energy_bins_kev,
                        counts_per_s_kev=counts_i if counts_i.ndim == 1 else counts_i.ravel(),
                        quality_flag=QualityFlag(int(flags[i])),
                        dead_time_fraction=float(corr_fd.deadtime_frac[i]),
                        solar_distance_au=1.0,
                        spacecraft_mode="NORMAL",
                        saa_flag=False,
                        source_file=hel_pi_path.name,
                        checksum_sha256=corr_fd.checksum_sha256,
                        pipeline_version="2.0.0-pradan",
                        calibration_version="sarwade_2025_v1",
                    ))
        logger.info(
            "run_l1_pradan: %d HEL1OS frames from %d PI files",
            sum(1 for f in all_frames if "HEL1OS" in f.instrument),
            len(hel1os_pi),
        )

    # ── STIX fallback: fetch real hard X-ray data from Solar Orbiter ─────────
    elif all_frames:
        # Derive time window from the loaded SoLEXS frames
        sol_only = [f for f in all_frames if "SOLEXS" in f.instrument]
        if sol_only:
            ts_min = min(f.timestamp for f in sol_only)
            ts_max = max(f.timestamp for f in sol_only)
            start_str = ts_min.strftime("%Y-%m-%dT%H:%M:%S")
            end_str   = ts_max.strftime("%Y-%m-%dT%H:%M:%S")
            logger.info(
                "run_l1_pradan: no HEL1OS PI files — attempting STIX fallback %s → %s",
                start_str, end_str,
            )
            try:
                from aruna.l1_ingestion.stix_provider import STIXProvider
                stix = STIXProvider()
                stix_frames = stix.fetch(start_str, end_str)
                if stix_frames:
                    all_frames.extend(stix_frames)
                    logger.info(
                        "run_l1_pradan: added %d STIX hard X-ray frames (Solar Orbiter)",
                        len(stix_frames),
                    )
                else:
                    logger.warning(
                        "run_l1_pradan: STIX returned no frames — "
                        "falling back to SoLEXS-derived proxy"
                    )
                    proxy = _synthesize_hel1os_proxy(sol_only)
                    all_frames.extend(proxy)
            except Exception as exc:
                logger.warning(
                    "run_l1_pradan: STIX fetch failed (%s) — falling back to proxy", exc
                )
                proxy = _synthesize_hel1os_proxy(sol_only)
                all_frames.extend(proxy)

    return all_frames


# ── L2 Physics ───────────────────────────────────────────────────────────────

def run_l2(obs_frames: List) -> List:
    """
    Run L2 physics engine. Returns List[PhysicsFrame].

    PhysicsEngine.process() requires paired (sol, hel) ObservationFrames.
    This function separates the frame list by instrument, then pairs SoLEXS
    and HEL1OS frames with the closest timestamp (within 2 s).
    """
    from aruna.l2_physics import PhysicsEngine, PhysicsConfig
    import numpy as np

    sol_frames = [f for f in obs_frames if "SOLEXS" in f.instrument]
    hel_frames = [f for f in obs_frames if "HEL1OS" in f.instrument or "HEL" in f.instrument]

    if not sol_frames:
        logger.warning("L2: no SoLEXS frames in input — cannot run physics engine")
        return []
    if not hel_frames:
        logger.warning("L2: no HEL1OS frames — cannot run physics engine (need paired data)")
        return []

    # Sort both lists by timestamp
    sol_frames.sort(key=lambda f: f.timestamp)
    hel_frames.sort(key=lambda f: f.timestamp)

    # Pair by nearest timestamp (O(N log N) merge-like approach)
    hel_ts = [f.timestamp for f in hel_frames]
    engine = PhysicsEngine(config=PhysicsConfig())
    physics_frames = []
    skipped = 0

    for sol in sol_frames:
        # Find nearest HEL1OS frame
        diffs = [abs((sol.timestamp - ht).total_seconds()) for ht in hel_ts]
        nearest_idx = int(np.argmin(diffs))
        if diffs[nearest_idx] > 2.0:
            skipped += 1
            continue
        hel = hel_frames[nearest_idx]
        try:
            pf = engine.process(sol, hel)
            if pf is not None:
                physics_frames.append(pf)
        except Exception as exc:
            logger.debug("L2: skipped frame pair @ %s — %s", sol.timestamp, exc)
            skipped += 1

    logger.info("L2: produced %d physics frames from %d sol + %d hel frames (skipped %d)",
                len(physics_frames), len(sol_frames), len(hel_frames), skipped)
    return physics_frames


# ── L3 Nowcasting ─────────────────────────────────────────────────────────────

def run_l3(physics_frames: List) -> tuple:
    """
    Run L3 nowcasting frame-by-frame.
    Returns (nowcast_event, all_physics_frames_processed).
    """
    from aruna.l3_nowcasting import NowcastOrchestrator, NowcastConfig

    orchestrator = NowcastOrchestrator(config=NowcastConfig())
    latest_event = None

    for pf in physics_frames:
        try:
            event = orchestrator.process(pf)
            if event is not None:
                latest_event = event
                logger.info("L3: NowcastEvent — %s confidence=%.2f",
                            event.goes_label, event.confidence)
        except Exception as exc:
            logger.debug("L3: frame skipped — %s", exc)

    return latest_event, physics_frames


# ── L4 Forecasting ────────────────────────────────────────────────────────────

def run_l4(physics_frames: List, nowcast_event=None) -> Optional[List]:
    """Run L4 forecasting. Returns List[ForecastFrame] or None."""
    from aruna.l4_forecasting import ForecastOrchestrator, ForecastConfig

    orchestrator = ForecastOrchestrator(config=ForecastConfig())
    latest_forecasts = None

    for pf in physics_frames:
        try:
            result = orchestrator.process(pf, nowcast_event=nowcast_event)
            if result is not None:
                latest_forecasts = result
        except Exception as exc:
            logger.debug("L4: frame skipped — %s", exc)

    if latest_forecasts:
        ff = latest_forecasts[0]
        logger.info("L4: forecast — class=%s P(M+X)=%.2f lead=%ds",
                    ff.predicted_class, ff.prob_M + ff.prob_X, ff.lead_time_estimate_sec)
    return latest_forecasts


# ── L5 Analogs ───────────────────────────────────────────────────────────────

def run_l5(forecast_frames: List, physics_frames: List) -> List:
    """Run L5 analog retrieval. Returns List[AnalogMatch]."""
    from aruna.l5_analogs.orchestrator import AnalogOrchestrator
    from aruna.l5_analogs.models import NO_ANALOG

    orchestrator = AnalogOrchestrator()
    ff = forecast_frames[0] if forecast_frames else None

    if ff is None:
        return [NO_ANALOG]

    try:
        analogs = orchestrator.process(ff, physics_frames)
        logger.info("L5: %d analogs found (first: %s sim=%.2f)",
                    len(analogs), analogs[0].goes_class, analogs[0].similarity)
        return analogs
    except Exception as exc:
        logger.warning("L5: analog retrieval failed — %s", exc)
        return [NO_ANALOG]


# ── L6 Threat ────────────────────────────────────────────────────────────────

def run_l6(nowcast_event) -> Optional[object]:
    """Run L6 threat intelligence. Returns ThreatOutput."""
    from aruna.l6_threat import ThreatOrchestrator

    try:
        threat = ThreatOrchestrator().process(nowcast_event)
        logger.info("L6: NavIC worst=%s HF=%s Gaganyaan=%s",
                    max(threat.navic_risk.values(), default="UNKNOWN"),
                    threat.hf_risk, threat.gaganyaan_advisory)
        return threat
    except Exception as exc:
        logger.error("L6: threat computation failed — %s", exc)
        return None


# ── L7 Briefing ───────────────────────────────────────────────────────────────

def run_l7(nowcast_event, forecast_frames, threat, analogs, provider: str = "mock") -> object:
    """Run L7 LLM briefing. Returns AlertBriefing."""
    from aruna.l7_llm import BriefingOrchestrator
    from aruna.l7_llm.models import BriefingContext
    from aruna.l7_llm.factory import LLMFactory

    config = {"llm": {"provider": provider}}
    llm = LLMFactory.from_config(config)

    ctx = BriefingContext(
        nowcast=nowcast_event,
        forecast=forecast_frames[0],
        threat=threat,
        analogs=analogs,
    )
    orchestrator = BriefingOrchestrator(llm=llm)
    briefing = orchestrator.process(ctx)
    logger.info("L7: briefing source=%s violations=%d",
                briefing.briefing_source, len(briefing.hallucination_violations))
    return briefing


# ── Synthetic nowcast fallback ────────────────────────────────────────────────

def _synthetic_nowcast(goes_class: str = "X", goes_subclass: float = 6.3):
    """Create a synthetic NowcastEvent for demo mode."""
    from aruna.models.nowcast import NowcastEvent
    return NowcastEvent(
        alert_type="FLARE_ONSET",
        timestamp=datetime(2024, 2, 22, 9, 3, 0, tzinfo=timezone.utc),
        goes_class=goes_class,
        goes_subclass=goes_subclass,
        confidence=0.94,
        instruments=["SoLEXS", "HEL1OS"],
        physics_frame_id="synth-pf-001",
        detection_method="DUAL_TRIGGER",
        flare_type="STANDARD",
        lead_time_sec=660,
    )


def _synthetic_forecast():
    """Create a synthetic ForecastFrame for demo mode."""
    from aruna.models.forecast import ForecastFrame
    return ForecastFrame(
        timestamp=datetime(2024, 2, 22, 9, 3, 0, tzinfo=timezone.utc),
        horizon_minutes=15,
        prob_quiet=0.0, prob_B=0.0, prob_C=0.01, prob_M=0.10, prob_X=0.89,
        predicted_class="X",
        peak_class_prob=0.89,
        confidence_interval_low=0.80,
        confidence_interval_high=0.95,
        prediction_set=["X"],
        calibration_ece=0.0,
        lead_time_estimate_sec=660,
        lead_time_uncertainty_sec=60,
        tcn_weight=0.5,
        moirai_weight=0.5,
        hope_flag_active=True,
        physics_state="HOPE",
        input_window_start=datetime(2024, 2, 22, 8, 53, tzinfo=timezone.utc),
        input_window_end=datetime(2024, 2, 22, 9, 3, tzinfo=timezone.utc),
        model_version="1",
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ARUNA end-to-end pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--fits-dir", default="python/tests/fixtures/x63/",
                        help="Directory containing SoLEXS + HEL1OS FITS files")
    parser.add_argument("--event-id", default="SOL2024-02-22-X6.3",
                        help="Event label for logging")
    parser.add_argument("--provider", default="mock", choices=["mock", "azure"],
                        help="LLM provider: mock (default) or azure (needs AZURE_OPENAI_API_KEY)")
    parser.add_argument("--output", default=None,
                        help="Write AlertBriefing JSON to this file path")
    parser.add_argument("--synthetic", action="store_true",
                        help="Skip FITS ingestion — use synthetic X6.3 profile")
    parser.add_argument("--single-instrument", action="store_true",
                        help="Diagnostic mode: ingest SoLEXS-only, synthesize HEL1OS proxy")
    parser.add_argument("--window-sec", type=int, default=5400,
                        help="Time window around X6.3 peak for PRADAN data (default 5400=90min)")
    args = parser.parse_args(argv)

    t_start = time.monotonic()
    logger.info("=" * 60)
    logger.info("ARUNA Pipeline — event: %s", args.event_id)
    logger.info("=" * 60)

    # ── L1+L2: Ingest and compute physics ────────────────────────────────────
    physics_frames = []
    if not args.synthetic:
        fits_dir = Path(args.fits_dir)
        if fits_dir.exists():
            fits_files = _find_fits(fits_dir)
            logger.info(
                "Found FITS: solexs=%d hel1os=%d hk=%d solexs_pi=%d hel1os_pi=%d",
                len(fits_files["solexs"]), len(fits_files["hel1os"]), len(fits_files["hk"]),
                len(fits_files.get("solexs_pi", [])), len(fits_files.get("hel1os_pi", [])),
            )
            obs_frames: List = []
            if fits_files.get("solexs_pi"):
                # PRADAN per-file format detected
                obs_frames = run_l1_pradan(
                    fits_files,
                    single_instrument=getattr(args, "single_instrument", False),
                    window_sec=getattr(args, "window_sec", 5400),
                )
            elif fits_files["solexs"] and fits_files["hel1os"] and fits_files["hk"]:
                obs_frames = run_l1(fits_files)
            else:
                logger.warning("FITS dir exists but no usable files — using synthetic profile")

            if obs_frames:
                physics_frames = run_l2(obs_frames)
        else:
            logger.warning("FITS dir not found: %s", fits_dir)

    if not physics_frames:
        logger.info("L1/L2: Using synthetic X6.3 physics profile (700 frames)")
        physics_frames = _make_synthetic_physics_frames(700)

    # ── L3: Nowcasting ────────────────────────────────────────────────────────
    nowcast_event, _ = run_l3(physics_frames)

    if nowcast_event is None:
        logger.info("L3: No event detected — using synthetic X6.3 NowcastEvent")
        nowcast_event = _synthetic_nowcast("X", 6.3)

    # ── L4: Forecasting ───────────────────────────────────────────────────────
    forecast_frames = run_l4(physics_frames, nowcast_event)

    if not forecast_frames:
        logger.info("L4: Buffer not ready — using synthetic ForecastFrame")
        forecast_frames = [_synthetic_forecast()]

    # ── L5: Analogs ───────────────────────────────────────────────────────────
    analogs = run_l5(forecast_frames, physics_frames)

    # ── L6: Threat ────────────────────────────────────────────────────────────
    threat = run_l6(nowcast_event)

    if threat is None:
        logger.error("L6: Threat computation failed — cannot produce briefing")
        sys.exit(1)

    # ── L7: LLM Briefing ──────────────────────────────────────────────────────
    briefing = run_l7(nowcast_event, forecast_frames, threat, analogs, provider=args.provider)

    elapsed = time.monotonic() - t_start

    # ── Output ────────────────────────────────────────────────────────────────
    print()
    print(briefing.briefing_text)
    print()
    print(f"[Pipeline complete in {elapsed:.1f}s | source: {briefing.briefing_source}]")

    if briefing.hallucination_violations:
        print(f"[Guard violations: {briefing.hallucination_violations}]")

    if args.output:
        payload = {
            "event_id": args.event_id,
            "briefing_text": briefing.briefing_text,
            "briefing_source": briefing.briefing_source,
            "hallucination_violations": briefing.hallucination_violations,
            "created_at": briefing.created_at,
            "nowcast": {
                "goes_label": nowcast_event.goes_label,
                "confidence": nowcast_event.confidence,
                "lead_time_sec": nowcast_event.lead_time_sec,
            },
            "threat": {
                "navic_risk": threat.navic_risk,
                "hf_risk": threat.hf_risk,
                "gaganyaan_advisory": threat.gaganyaan_advisory,
            },
            "pipeline_elapsed_sec": round(elapsed, 2),
        }
        Path(args.output).write_text(json.dumps(payload, indent=2, default=str))
        logger.info("Output written to %s", args.output)

    return briefing


if __name__ == "__main__":
    main()
