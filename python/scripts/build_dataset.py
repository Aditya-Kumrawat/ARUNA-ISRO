#!/usr/bin/env python3
"""
build_dataset.py — Slide 600-frame windows over the Parquet cache, join GOES
labels, apply the PRD temporal split, fit the feature scaler on train only,
and write train/val/test Parquet files ready for train_tcn.py.

Usage:
    python python/scripts/build_dataset.py \
        --cache-dir   data/physics_cache/ \
        --goes-file   data/goes_events.parquet \
        --output-dir  data/dataset/ \
        --scaler-out  models/feature_scaler.pkl \
        --horizon-min 30 \
        --stride-sec  60

Outputs:
    data/dataset/train.parquet   — columns: feature cols + label (int 0-4) + window_end (str)
    data/dataset/val.parquet
    data/dataset/test.parquet
    models/feature_scaler.pkl    — StandardScaler fitted on train windows only

Label encoding:
    0 = QUIET   1 = B   2 = C   3 = M   4 = X
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("build_dataset")

# ── Constants ──────────────────────────────────────────────────────────────────

SEQUENCE_LENGTH = 600   # 10 minutes at 1-second cadence

# PRD §8.9 temporal split (hard date boundaries)
TRAIN_START = pd.Timestamp("2023-09-02", tz="UTC")
TRAIN_END   = pd.Timestamp("2024-06-30 23:59:59", tz="UTC")
VAL_START   = pd.Timestamp("2024-07-01", tz="UTC")
VAL_END     = pd.Timestamp("2024-12-31 23:59:59", tz="UTC")
TEST_START  = pd.Timestamp("2025-01-01", tz="UTC")
TEST_END    = pd.Timestamp("2025-06-30 23:59:59", tz="UTC")

# Feature columns fed to the TCN (must match FeatureExtractor order)
THERMAL_COLS = [
    "T_MK", "EM_cm3", "DNI",
    "tnpsv_magnitude", "tnpsv_angle_rad",
    "dT_dt", "dEM_dt", "d2T_dt2",
]
NONTHERMAL_COLS = [
    "SHEI", "dSHEI_dt", "FeKa_flux",
    "hel1os_10_25kev", "hel1os_25_50kev", "hel1os_50_70kev",
    "QPP_period_sec", "QPP_amplitude",
]
SHARED_BASE_COLS = ["HFD", "solexs_3_6kev", "solexs_6_10kev"]
FEATURE_COLS = THERMAL_COLS + NONTHERMAL_COLS + SHARED_BASE_COLS
# (flare_phase one-hot added separately at training time from "flare_phase" col)

_CLASS_MAP = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}

# Phase string -> uint8 integer (stored as bytes; avoids UCS-4 numpy encoding issues)
_PHASE_TO_INT: dict[str, int] = {
    "QUIET": 0, "HOPE": 1, "IMPULSIVE": 2, "GRADUAL": 3, "DECAY": 4,
}


# ── GOES label join ────────────────────────────────────────────────────────────

def _max_goes_class(events: pd.DataFrame) -> int:
    """Return the highest GOES class integer (0-4) in a subset of events."""
    if events.empty:
        return 0
    letters = events["goes_class_letter"].dropna().str.upper()
    nums = [_CLASS_MAP.get(l, 0) for l in letters]
    return max(nums) if nums else 0


def _label_window(window_end: pd.Timestamp, goes: pd.DataFrame, horizon_min: int) -> int:
    """
    Assign the GOES class label for a window.

    Looks for any event whose PEAK falls in [window_end, window_end + horizon_min].
    Returns integer 0–4 (QUIET=0, B=1, C=2, M=3, X=4).
    """
    horizon_end = window_end + pd.Timedelta(minutes=horizon_min)
    in_window = goes[
        (goes["peak_time"] >= window_end) &
        (goes["peak_time"] <  horizon_end)
    ]
    return _max_goes_class(in_window)


# ── Window builder ─────────────────────────────────────────────────────────────

def _build_windows(
    df: pd.DataFrame,
    goes: pd.DataFrame,
    stride_sec: int,
    horizon_min: int,
    max_nan_frac: float = 0.10,
) -> list[dict]:
    """
    Slide windows of SEQUENCE_LENGTH rows over df (one day's PhysicsFrames).
    Returns list of dicts {features: np.ndarray (L, F), label: int, window_end: str,
                           flare_phase_seq: list[str]}.
    """
    n = len(df)
    results = []
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    feature_arr = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    phase_arr   = df["flare_phase"].to_numpy(dtype=str)

    for start in range(0, n - SEQUENCE_LENGTH + 1, stride_sec):
        end = start + SEQUENCE_LENGTH

        window_feats  = feature_arr[start:end]
        window_phases = phase_arr[start:end]

        # Skip windows with too many NaN / zero rows (warm-up or bad data)
        nan_frac = np.isnan(window_feats).mean()
        if nan_frac > max_nan_frac:
            continue

        # Replace NaN with 0 (FeatureExtractor does the same)
        window_feats = np.nan_to_num(window_feats, nan=0.0)

        window_end_ts = timestamps.iloc[end - 1]
        label = _label_window(window_end_ts, goes, horizon_min)

        results.append({
            "features":     window_feats,       # (600, 22) — phase added later
            "flare_phases": window_phases,       # (600,) str
            "label":        label,
            "window_end":   window_end_ts.isoformat(),
        })

    return results


def _windows_to_flat_df(windows: list[dict]) -> pd.DataFrame:
    """
    Flatten each window into a single Parquet row:
    - 22 feature cols × 600 timesteps → stored as serialised numpy bytes
    - flare_phase sequence → bytes
    - label (int), window_end (str)
    """
    rows = []
    for w in windows:
        # Encode phases as uint8 integers (0-4) — avoids numpy UCS-4 encoding issues
        # that would make the one-hot decoding in train_tcn.py produce all-zeros.
        phase_ints = np.array(
            [_PHASE_TO_INT.get(str(p).strip(), 0) for p in w["flare_phases"]],
            dtype=np.uint8,
        )
        rows.append({
            "features_bytes": w["features"].astype(np.float32).tobytes(),
            "phases_bytes":   phase_ints.tobytes(),   # (SEQUENCE_LENGTH,) uint8
            "label":          w["label"],
            "window_end":     w["window_end"],
        })
    return pd.DataFrame(rows)


# ── Split assignment ───────────────────────────────────────────────────────────

def _assign_split(window_end_str: str) -> str:
    ts = pd.Timestamp(window_end_str, tz="UTC")
    if TRAIN_START <= ts <= TRAIN_END:
        return "train"
    if VAL_START <= ts <= VAL_END:
        return "val"
    if TEST_START <= ts <= TEST_END:
        return "test"
    return "other"


# ── Scaler ─────────────────────────────────────────────────────────────────────

def _fit_scaler(train_df: pd.DataFrame, scaler_out: str) -> None:
    """Fit StandardScaler on all training windows and save to pickle."""
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        logger.warning("scikit-learn not installed — skipping scaler fitting")
        return

    if train_df.empty:
        logger.warning("No training windows to fit scaler on")
        return

    logger.info("Fitting StandardScaler on %d training windows …", len(train_df))
    n_features = len(FEATURE_COLS)

    # Incremental fit to avoid loading all windows into RAM at once
    scaler = StandardScaler()
    features_bytes = train_df["features_bytes"].to_numpy()
    for raw in features_bytes:
        feats = np.frombuffer(raw, dtype=np.float32).reshape(SEQUENCE_LENGTH, n_features)
        scaler.partial_fit(feats)

    Path(scaler_out).parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_out, "wb") as fh:
        pickle.dump(scaler, fh)
    logger.info("Scaler saved → %s", scaler_out)


# ── Main ───────────────────────────────────────────────────────────────────────

def build(
    cache_dir: str,
    goes_file: str,
    output_dir: str,
    scaler_out: str,
    horizon_min: int,
    stride_sec: int,
) -> None:
    cache   = Path(cache_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load GOES catalog
    if not Path(goes_file).exists():
        print(f"GOES catalog not found at {goes_file}. Run download_goes_labels.py first.")
        sys.exit(1)
    goes = pd.read_parquet(goes_file)
    goes["peak_time"] = pd.to_datetime(goes["peak_time"], utc=True)
    logger.info("Loaded %d GOES events", len(goes))

    parquet_files = sorted(cache.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files in {cache_dir}. Run batch_l1_l2.py first.")
        sys.exit(1)
    logger.info("Found %d cached days to window", len(parquet_files))

    all_windows: list[dict] = []

    for pq_file in parquet_files:
        try:
            df = pd.read_parquet(pq_file)
            if len(df) < SEQUENCE_LENGTH:
                logger.debug("%s: only %d rows — skipping", pq_file.name, len(df))
                continue
            wins = _build_windows(df, goes, stride_sec, horizon_min)
            all_windows.extend(wins)
            logger.info("%s: %d windows", pq_file.stem, len(wins))
        except Exception as exc:
            logger.warning("%s: failed — %s", pq_file.name, exc)

    if not all_windows:
        print("ERROR: No windows built. Check cache and GOES file.")
        sys.exit(1)

    logger.info("Total windows: %d", len(all_windows))

    # Flatten and split
    flat_df = _windows_to_flat_df(all_windows)
    flat_df["split"] = flat_df["window_end"].apply(_assign_split)

    splits = flat_df.groupby("split").size()
    logger.info("Split counts: %s", dict(splits))

    for split_name in ("train", "val", "test"):
        subset = flat_df[flat_df["split"] == split_name].drop(columns=["split"])
        out_path = out_dir / f"{split_name}.parquet"
        subset.to_parquet(out_path, index=False)
        logger.info("Saved %d %s windows → %s", len(subset), split_name, out_path)

    # Label distribution per split
    print("\n=== Label Distribution ===")
    _CLASS_NAMES = {0: "QUIET", 1: "B", 2: "C", 3: "M", 4: "X"}
    for split_name in ("train", "val", "test"):
        subset = flat_df[flat_df["split"] == split_name]
        if subset.empty:
            continue
        print(f"\n{split_name.upper()} ({len(subset)} windows):")
        for lbl, name in _CLASS_NAMES.items():
            count = (subset["label"] == lbl).sum()
            pct   = count / len(subset) * 100
            print(f"  {name} ({lbl}): {count:>6}  ({pct:5.1f}%)")

    # Fit scaler on training set only
    train_df = flat_df[flat_df["split"] == "train"]
    _fit_scaler(train_df, scaler_out)

    print(f"\nDataset written to {output_dir}")
    print("Next step: python python/scripts/train_tcn.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build windowed TCN dataset from Parquet cache")
    parser.add_argument("--cache-dir",   default="data/physics_cache")
    parser.add_argument("--goes-file",   default="data/goes_events.parquet")
    parser.add_argument("--output-dir",  default="data/dataset")
    parser.add_argument("--scaler-out",  default="models/feature_scaler.pkl")
    parser.add_argument("--horizon-min", type=int, default=30,
                        help="Minutes ahead to look for GOES event peak")
    parser.add_argument("--stride-sec",  type=int, default=60,
                        help="Window stride in seconds (lower = more windows)")
    args = parser.parse_args()
    build(
        args.cache_dir, args.goes_file, args.output_dir,
        args.scaler_out, args.horizon_min, args.stride_sec,
    )


if __name__ == "__main__":
    main()
