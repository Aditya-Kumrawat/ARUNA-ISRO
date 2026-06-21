#!/usr/bin/env python3
"""
batch_l1_l2.py — Process all PRADAN FITS files through ARUNA L1 + L2 and
cache PhysicsFrames as daily Parquet files.

Run this ONCE before training. Each Parquet file represents one day of
1-second-cadence physics observables that the TCN training loop reads directly.

Usage:
    python python/scripts/batch_l1_l2.py \
        --solexs-dir data/solexs/ \
        --hel1os-dir data/hel1os/ \
        --output-dir data/physics_cache/ \
        --workers 4

Output:
    data/physics_cache/YYYY-MM-DD.parquet   (one file per day)

Each Parquet row = one PhysicsFrame with columns matching FeatureExtractor:
    timestamp, T_MK, EM_cm3, DNI, tnpsv_magnitude, tnpsv_angle_rad,
    dT_dt, dEM_dt, d2T_dt2, SHEI, dSHEI_dt, FeKa_flux,
    hel1os_10_25kev, hel1os_25_50kev, hel1os_50_70kev,
    QPP_period_sec, QPP_amplitude, HFD, solexs_3_6kev, solexs_6_10kev,
    hope_flag, flare_phase, quality_flag
"""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import warnings
# Pandera import path changed in >=0.19; suppress until they remove the old API
warnings.filterwarnings("ignore", category=FutureWarning, module="pandera")

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("batch_l1_l2")

_DATE_RE = re.compile(r'(\d{8})')


# ── File discovery ─────────────────────────────────────────────────────────────

def _scan_dir(root: Path) -> dict[str, list[Path]]:
    """Group FITS files by date string (YYYYMMDD)."""
    patterns = ("*.fits", "*.fit", "*.fits.gz", "**/*.fits", "**/*.fits.gz",
                "**/*.pi.gz", "**/*.lc.gz", "**/*.gti.gz")
    all_files: list[Path] = []
    for p in patterns:
        all_files.extend(root.glob(p))
    for entry in root.iterdir():
        if entry.is_dir() and _DATE_RE.search(entry.name):
            for p in ("*.fits.gz", "**/*.pi.gz", "**/*.lc.gz", "**/*.fits"):
                all_files.extend(entry.glob(p))

    date_map: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(set(all_files)):
        # HEL1OS files may not include YYYYMMDD in the filename itself.
        # Fall back to scanning the full path (which contains dated parent dirs).
        m = _DATE_RE.search(f.name) or _DATE_RE.search(str(f))
        if m:
            date_map[m.group(1)].append(f)
    return dict(date_map)


def _pick_hk(files: list[Path]) -> Optional[Path]:
    """Find a housekeeping FITS file from a list; return None if absent."""
    for f in files:
        if any(k in f.name.upper() for k in ("_HK", "HOUSE", "AUX")):
            return f
    return None


def _pick_instrument(files: list[Path], keys: tuple[str, ...]) -> Optional[Path]:
    """Pick the primary spectral file (PI spectrum preferred over LC)."""
    pi_files = [f for f in files if ".pi." in f.name.lower() or f.suffix.lower() == ".pi"]
    candidates = pi_files or files
    for f in candidates:
        if any(k in f.name.upper() for k in keys):
            return f
    return None


def _pick_pradan_spectrum(files: list[Path], detector_keys: tuple[str, ...]) -> Optional[Path]:
    """Pick a PRADAN spectrum file for a detector (PI preferred, then SPECTRA FITS)."""
    for f in files:
        name_up = f.name.upper()
        if any(k in name_up for k in detector_keys) and ".PI." in name_up:
            return f
    for f in files:
        name_up = f.name.upper()
        if any(k in name_up for k in detector_keys) and "SPECTRA" in name_up:
            return f
    return None


def _pick_companion_gti(files: list[Path], detector_keys: tuple[str, ...]) -> Optional[Path]:
    """Pick GTI file for a detector when available."""
    for f in files:
        name_up = f.name.upper()
        if ".GTI." in name_up and any(k in name_up for k in detector_keys):
            return f
    return None


def _is_pradan_solexs(files: list[Path]) -> bool:
    names = [f.name.lower() for f in files]
    return any(".pi." in n for n in names) or any("solexs" in n and "sdd" in n for n in names)


def _is_pradan_hel1os(files: list[Path]) -> bool:
    names = [f.name.lower() for f in files]
    return any("spectra" in n for n in names) or any(".pi." in n for n in names)


# ── PhysicsFrame → dict (for Parquet serialisation) ───────────────────────────

def _frame_to_dict(pf) -> dict:
    return {
        "timestamp":          pf.timestamp.isoformat(),
        "T_MK":               float(pf.T_MK or 0.0),
        "EM_cm3":             float(pf.EM_cm3 or 0.0),
        "DNI":                float(pf.DNI or 0.0),
        "tnpsv_magnitude":    float(pf.tnpsv_magnitude or 0.0),
        "tnpsv_angle_rad":    float(pf.tnpsv_angle_rad or 0.0),
        "dT_dt":              float(pf.dT_dt or 0.0),
        "dEM_dt":             float(pf.dEM_dt or 0.0),
        "d2T_dt2":            float(pf.d2T_dt2 or 0.0),
        "SHEI":               float(pf.SHEI or 0.0),
        "dSHEI_dt":           float(pf.dSHEI_dt or 0.0),
        "FeKa_flux":          float(pf.FeKa_flux or 0.0),
        "hel1os_10_25kev":    float(pf.hel1os_10_25kev or 0.0),
        "hel1os_25_50kev":    float(pf.hel1os_25_50kev or 0.0),
        "hel1os_50_70kev":    float(pf.hel1os_50_70kev or 0.0),
        "QPP_period_sec":     float(pf.QPP_period_sec or 0.0),
        "QPP_amplitude":      float(pf.QPP_amplitude or 0.0),
        "HFD":                float(pf.HFD or 1.5),
        "solexs_3_6kev":      float(pf.solexs_3_6kev or 0.0),
        "solexs_6_10kev":     float(pf.solexs_6_10kev or 0.0),
        "hope_flag":          bool(pf.hope_flag),
        "flare_phase":        str(pf.flare_phase or "QUIET"),
        "quality_flag":       str(pf.quality_flag),
    }


# ── Single-day processor ───────────────────────────────────────────────────────

def _process_day(args: tuple) -> tuple[str, int, str]:
    """
    Worker function: process one day's FITS files → Parquet.
    Returns (date_str, n_frames_written, error_message_or_empty).
    """
    date_str, solexs_files, hel1os_files, out_dir = args

    out_path = Path(out_dir) / f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}.parquet"
    if out_path.exists():
        return (date_str, -1, "")   # already cached

    try:
        # Lazy imports inside worker to avoid multiprocessing issues
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

        from aruna.l1_ingestion.fits_reader import FitsReader
        from aruna.l1_ingestion.deadtime import DeadTimeCorrector
        from aruna.l2_physics.engine import PhysicsEngine
        from aruna.l1_ingestion.models import FitsData

        reader = FitsReader()
        dtc    = DeadTimeCorrector()
        engine = PhysicsEngine()

        # ── SoLEXS ────────────────────────────────────────────────────────────
        solexs_list: list[FitsData] = []
        sol_primary = _pick_instrument(solexs_files, ("SLX", "SOLEXS", "SDD"))
        if sol_primary is None:
            return (date_str, 0, "No SoLEXS file found")

        if _is_pradan_solexs(solexs_files):
            for det in ("SDD1", "SDD2"):
                det_keys = (det,)
                pi_path = _pick_pradan_spectrum(solexs_files, det_keys)
                if pi_path is None:
                    continue
                gti_path = _pick_companion_gti(solexs_files, det_keys)
                solexs_list.extend(
                    reader.read_solexs_pradan_pi(
                        pi_path,
                        gti_path=gti_path,
                        detector=det,
                        window_center_unix=None,
                        window_sec=86400,
                    )
                )
        else:
            solexs_list = reader.read_solexs(sol_primary)

        if not solexs_list:
            return (date_str, 0, "FitsReader returned empty SoLEXS list")

        sdd1 = next((f for f in solexs_list if "SDD1" in f.instrument.upper()), None)
        sdd2 = next((f for f in solexs_list if "SDD2" in f.instrument.upper()), None)

        if sdd1 is None or sdd2 is None:
            # Single-detector file — use whatever is present
            sol_data = solexs_list[0]
            sol_flags = np.zeros(len(sol_data.timestamps), dtype=np.int8)
        else:
            corr1, flags1 = dtc.correct(sdd1)
            corr2, flags2 = dtc.correct(sdd2)
            sol_data, sol_flags = dtc.select_solexs_detector(corr1, flags1, corr2, flags2)

        # ── HEL1OS ────────────────────────────────────────────────────────────
        hel_primary = _pick_instrument(hel1os_files, ("HEL", "HXS", "HXMT", "CDTE", "CZT"))
        if hel_primary is None:
            return (date_str, 0, "No HEL1OS file found")

        hel1os_list: list[FitsData] = []
        if _is_pradan_hel1os(hel1os_files):
            for det, keys in (("CDTE", ("CDTE",)), ("CZT", ("CZT",))):
                spec_path = _pick_pradan_spectrum(hel1os_files, keys)
                if spec_path is None:
                    continue
                gti_path = _pick_companion_gti(hel1os_files, keys)
                hel1os_list.extend(
                    reader.read_hel1os_pradan_pi(
                        spec_path,
                        gti_path=gti_path,
                        detector=det,
                        window_center_unix=None,
                        window_sec=86400,
                    )
                )
        else:
            hel1os_list = reader.read_hel1os(hel_primary)

        if not hel1os_list:
            return (date_str, 0, "FitsReader returned empty HEL1OS list")

        cdte = next((f for f in hel1os_list if "CDTE" in f.instrument.upper()), hel1os_list[0])
        hel_data, hel_flags = dtc.correct(cdte)

        # ── Align FitsData to overlapping time range ──────────────────────────
        from aruna.l1_ingestion.alignment import TimeAligner

        aligner = TimeAligner(drift_warn_sec=1.0, drift_max_sec=5.0)
        sol_data, hel_data, _ = aligner.align(sol_data, hel_data)

        n_sol = len(sol_data.timestamps)
        n_hel = len(hel_data.timestamps)
        if n_sol == 0 or n_hel == 0:
            return (date_str, 0, "Alignment produced empty frame lists")

        # After interpolation-based alignment, the resampled instrument may have
        # more frames than the original flags array (e.g. HEL1OS ~2157 → ~86396).
        # Pad with zeros (GOOD flag) for the synthetic interpolated frames.
        if n_sol > len(sol_flags):
            sol_flags = np.pad(sol_flags, (0, n_sol - len(sol_flags)))
        else:
            sol_flags = sol_flags[:n_sol]
        if n_hel > len(hel_flags):
            hel_flags = np.pad(hel_flags, (0, n_hel - len(hel_flags)))
        else:
            hel_flags = hel_flags[:n_hel]

        # ── Convert to ObservationFrames ──────────────────────────────────────
        from aruna.models.observation import ObservationFrame, QualityFlag

        def _fits_to_obs(fd: FitsData, flags: np.ndarray) -> List[ObservationFrame]:
            frames = []
            for i in range(len(fd.timestamps)):
                ts_ns = fd.timestamps[i]
                ts_sec = float(
                    (ts_ns - np.datetime64('1970-01-01T00:00:00', 'ns'))
                    / np.timedelta64(1, 's')
                )
                ts = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
                frames.append(ObservationFrame(
                    instrument=fd.instrument,
                    timestamp=ts,
                    energy_bins_kev=fd.energy_bins_kev,
                    counts_per_s_kev=fd.counts_raw[i],
                    quality_flag=QualityFlag(int(flags[i])),
                    dead_time_fraction=float(fd.deadtime_frac[i]),
                    calibration_version="sarwade_2025_v1",
                    solar_distance_au=1.0,
                    spacecraft_mode="SCIENCE",
                    saa_flag=False,
                    source_file=str(fd.source_file),
                    checksum_sha256="",
                    pipeline_version="2.0.0",
                ))
            return frames

        sol_obs = _fits_to_obs(sol_data, sol_flags)
        hel_obs = _fits_to_obs(hel_data, hel_flags)

        # ── L2 physics pass ───────────────────────────────────────────────────
        rows = []
        for sol_f, hel_f in zip(sol_obs, hel_obs):
            try:
                pf = engine.process(sol_f, hel_f)
                rows.append(_frame_to_dict(pf))
            except Exception:
                pass   # skip bad frames, don't abort the day

        if not rows:
            return (date_str, 0, "L2 produced no valid PhysicsFrames")

        df = pd.DataFrame(rows)

        # Guardrail: warn if any physics column is >50 % NaN/zero (bad day)
        numeric_cols = [c for c in df.columns if df[c].dtype in (float, "float64", "float32")]
        nan_rates = df[numeric_cols].isna().mean()
        bad_cols = nan_rates[nan_rates > 0.5].index.tolist()
        if bad_cols:
            logger.warning("%s: >50%% NaN in %s — data quality suspect", date_str, bad_cols)

        # Warn on sparse days but still write (build_dataset skips windows with too many NaN)
        if len(rows) < 3600:
            logger.warning("%s: only %d frames (< 1 hour) — sparse day", date_str, len(rows))

        # Atomic write: use a .tmp file and rename on success so a crash mid-write
        # does not leave a corrupt Parquet that looks cached on the next run.
        tmp_path = out_path.with_suffix(".tmp.parquet")
        try:
            df.to_parquet(tmp_path, index=False)
            tmp_path.rename(out_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        return (date_str, len(rows), "")

    except Exception as exc:
        return (date_str, 0, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-400:]}")


# ── Orchestrator ───────────────────────────────────────────────────────────────

def run_batch(
    solexs_dir: str,
    hel1os_dir: str,
    output_dir: str,
    workers: int,
    max_days: Optional[int] = None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sol_map = _scan_dir(Path(solexs_dir))
    hel_map = _scan_dir(Path(hel1os_dir))

    dual_dates = sorted(set(sol_map) & set(hel_map))
    if not dual_dates:
        print("No dates with both SoLEXS and HEL1OS files found. Run audit first.")
        sys.exit(1)

    already_cached = sum(
        1 for d in dual_dates
        if (out / f"{d[:4]}-{d[4:6]}-{d[6:]}.parquet").exists()
    )
    todo = [d for d in dual_dates
            if not (out / f"{d[:4]}-{d[4:6]}-{d[6:]}.parquet").exists()]

    if max_days is not None and max_days < len(todo):
        todo = todo[:max_days]

    print(f"Dual-instrument days : {len(dual_dates)}")
    print(f"Already cached       : {already_cached}")
    print(f"To process           : {len(todo)}" + (f"  (capped at --max-days {max_days})" if max_days else ""))
    if not todo:
        print("All days already cached. Nothing to do.")
        print("Next step: python python/scripts/build_dataset.py")
        return

    work_args = [
        (d, sol_map[d], hel_map[d], output_dir)
        for d in todo
    ]

    print(f"\nProcessing {len(todo)} days with {workers} worker(s) …\n")
    success = 0
    failed  = 0
    skipped = 0

    if workers > 1:
        with mp.Pool(workers) as pool:
            for date_str, n_frames, err in pool.imap_unordered(_process_day, work_args):
                _log_result(date_str, n_frames, err)
                if n_frames > 0:
                    success += 1
                elif n_frames == -1:
                    skipped += 1
                else:
                    failed += 1
    else:
        for arg in work_args:
            date_str, n_frames, err = _process_day(arg)
            _log_result(date_str, n_frames, err)
            if n_frames > 0:
                success += 1
            elif n_frames == -1:
                skipped += 1
            else:
                failed += 1

    print(f"\n{'='*50}")
    print(f"Done. Success={success}  Failed={failed}  Skipped={skipped}")
    if failed:
        print(f"WARNING: {failed} days failed — check logs above.")
    print(f"\nParquet cache written to: {output_dir}")
    print("Next step: python python/scripts/build_dataset.py")


def _log_result(date_str: str, n_frames: int, err: str) -> None:
    date_fmt = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    if n_frames == -1:
        logger.debug("%s — already cached, skipping", date_fmt)
    elif n_frames > 0:
        logger.info("%s — %d frames written", date_fmt, n_frames)
    else:
        logger.warning("%s — FAILED: %s", date_fmt, err[:200])


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch L1+L2 processing → Parquet cache")
    parser.add_argument("--solexs-dir",  default="data/solexs",
                        help="Root directory of SoLEXS FITS files")
    parser.add_argument("--hel1os-dir",  default="data/hel1os",
                        help="Root directory of HEL1OS FITS files")
    parser.add_argument("--output-dir",  default="data/physics_cache",
                        help="Output directory for daily Parquet files")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (use 4-8 on multi-core machines)")
    parser.add_argument("--max-days", type=int, default=None,
                        help="Process only the first N days (chronological). Useful for smoke-testing.")
    args = parser.parse_args()
    run_batch(args.solexs_dir, args.hel1os_dir, args.output_dir, args.workers, args.max_days)


if __name__ == "__main__":
    main()
