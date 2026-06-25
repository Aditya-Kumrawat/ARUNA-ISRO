#!/usr/bin/env python3
"""
Stage 1: L3 Nowcasting End-to-End Validation

Streams physics_cache parquet files through NowcastOrchestrator, matches
detected events against GOES catalog, and reports operational metrics.

Usage (full run):
    python python/scripts/validate_l3.py \
        --physics-cache data/physics_cache \
        --goes-events data/goes_events.parquet \
        --out experiments/l3_baseline

Robustness modes (test B2/B3 fixes):
    --zero-hel1os   zero all HEL1OS channels before streaming (simulates HEL dropout)
    --zero-solexs   zero all SoLEXS channels before streaming (simulates SoLEXS dropout)

Calibration mode (B1 fix):
    --calibrate     fit solexs_to_wm2_factor from matched events and report it

Outputs (in --out dir):
    metrics.json        machine-readable metrics dict
    report.txt          human-readable summary
    confusion.txt       GOES classification confusion matrix
    false_alarms.csv    FA events list
    missed_events.csv   missed GOES events list
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Make sure the aruna package is importable from the repo root
_repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_repo_root / "python"))

from aruna.l2_physics.models import PhysicsFrame, PhysicsQuality
from aruna.l3_nowcasting.goes_class import GOESClassifier
from aruna.l3_nowcasting.models import NowcastConfig
from aruna.l3_nowcasting.orchestrator import NowcastOrchestrator
from aruna.models.nowcast import NowcastEvent
from aruna.models.observation import QualityFlag

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("validate_l3")

# ── Matching tolerances ────────────────────────────────────────────────────────
MATCH_WINDOW_MIN = 10   # L3 onset within ±10 min of GOES peak counts as detection
LEAD_TIME_MAX_MIN = 60  # ignore lead times > 60 min (likely false match)

GOES_CLASS_ORDER = ["A", "B", "C", "M", "X"]


# ── PhysicsFrame reconstruction from parquet row ───────────────────────────────

def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return default if (f != f) else f  # NaN check
    except (TypeError, ValueError):
        return default


def row_to_physics_frame(row) -> PhysicsFrame:
    """Convert one physics_cache parquet row to a PhysicsFrame."""
    ts_raw = row.get("timestamp") or row.get("time") or row.get("datetime")
    if ts_raw is None:
        raise ValueError("No timestamp column found in physics_cache row")
    if isinstance(ts_raw, (int, float)):
        ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
    elif isinstance(ts_raw, str):
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
    elif hasattr(ts_raw, "to_pydatetime"):
        ts = ts_raw.to_pydatetime()
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts_raw
        if hasattr(ts, "tzinfo") and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

    flare_phase = str(row.get("flare_phase", "QUIET"))

    return PhysicsFrame(
        frame_id=str(uuid.uuid4()),
        timestamp=ts,
        source_frame_ids=[],
        T_MK=_safe_float(row.get("T_MK", 0.0)),
        T_uncertainty_MK=_safe_float(row.get("T_uncertainty_MK", 0.0)),
        EM_cm3=_safe_float(row.get("EM_cm3", 0.0)),
        EM_uncertainty=_safe_float(row.get("EM_uncertainty", 0.0)),
        fit_chi_squared=_safe_float(row.get("fit_chi_squared", 0.0)),
        DNI=_safe_float(row.get("DNI", 0.0)),
        tnpsv_magnitude=_safe_float(row.get("tnpsv_magnitude", 0.0)),
        tnpsv_angle_rad=_safe_float(row.get("tnpsv_angle_rad", 0.0)),
        SHEI=_safe_float(row.get("SHEI", -2.0)),
        dSHEI_dt=_safe_float(row.get("dSHEI_dt", 0.0)),
        HFD=_safe_float(row.get("HFD", 0.0)),
        QPP_period_sec=None,
        QPP_amplitude=None,
        FeKa_flux=_safe_float(row.get("FeKa_flux", 0.0)),
        flare_phase=flare_phase,
        hope_flag=bool(row.get("hope_flag", False)),
        hope_lead_time_sec=None,
        hope_reason=None,
        engine_state="ACTIVE",
        data_quality_flag=QualityFlag.GOOD,
        physics_quality_flag=PhysicsQuality.FIT_OK,
        fit_backend="cached",
        pipeline_version="2.0.0",
        dT_dt=_safe_float(row.get("dT_dt", 0.0)),
        dEM_dt=_safe_float(row.get("dEM_dt", 0.0)),
        d2T_dt2=_safe_float(row.get("d2T_dt2", 0.0)),
        solexs_3_6kev=_safe_float(row.get("solexs_3_6kev", 0.0)),
        solexs_6_10kev=_safe_float(row.get("solexs_6_10kev", 0.0)),
        hel1os_10_25kev=_safe_float(row.get("hel1os_10_25kev", 0.0)),
        hel1os_25_50kev=_safe_float(row.get("hel1os_25_50kev", 0.0)),
        hel1os_50_70kev=_safe_float(row.get("hel1os_50_70kev", 0.0)),
        suit_uv_flux=_safe_float(row.get("suit_uv_flux", 0.0)),
        suit_uv_brightening_rate=_safe_float(row.get("suit_uv_brightening_rate", 0.0)),
        tnci=_safe_float(row.get("tnci", 0.0)),
    )


def apply_zero_mode(frame: PhysicsFrame, zero_hel: bool, zero_sol: bool) -> PhysicsFrame:
    """Zero out instrument channels to test degraded-mode detection."""
    if not zero_hel and not zero_sol:
        return frame
    f = copy.copy(frame)
    if zero_hel:
        f.hel1os_10_25kev = 0.0
        f.hel1os_25_50kev = 0.0
        f.hel1os_50_70kev = 0.0
        f.SHEI = -2.0
        f.dSHEI_dt = 0.0
    if zero_sol:
        f.solexs_3_6kev = 0.0
        f.solexs_6_10kev = 0.0
    return f


# ── Physics cache loading ──────────────────────────────────────────────────────

def load_physics_cache(cache_path: Path) -> pd.DataFrame:
    """Load all parquet files from physics_cache directory, sorted by timestamp."""
    files = sorted(cache_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {cache_path}")
    logger.info("Loading %d parquet files from %s", len(files), cache_path)
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    ts_col = next((c for c in ["timestamp", "time", "datetime"] if c in df.columns), None)
    if ts_col:
        df = df.sort_values(ts_col).reset_index(drop=True)
    logger.info("Physics cache: %d rows spanning %s → %s", len(df),
                df[ts_col].iloc[0] if ts_col else "?",
                df[ts_col].iloc[-1] if ts_col else "?")
    return df


def load_goes_events(goes_path: Path, start: Optional[datetime] = None,
                     end: Optional[datetime] = None) -> pd.DataFrame:
    """Load GOES flare catalog and filter to the physics_cache date range."""
    df = pd.read_parquet(goes_path)
    # normalise column names
    df.columns = [c.lower().strip() for c in df.columns]
    peak_col = next((c for c in ["peak_time", "peak", "event_peak_time", "time_peak"] if c in df.columns), None)
    if peak_col is None:
        raise ValueError(f"Cannot find peak_time column; found: {list(df.columns)}")
    df = df.rename(columns={peak_col: "peak_time"})
    df["peak_time"] = pd.to_datetime(df["peak_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["peak_time"])

    if start:
        df = df[df["peak_time"] >= pd.Timestamp(start)]
    if end:
        df = df[df["peak_time"] <= pd.Timestamp(end)]

    class_col = next((c for c in ["goes_class", "class", "flare_class"] if c in df.columns), None)
    if class_col:
        df = df.rename(columns={class_col: "goes_class"})
    else:
        df["goes_class"] = "U"

    df = df.sort_values("peak_time").reset_index(drop=True)
    logger.info("GOES catalog: %d events", len(df))
    return df


# ── Event matching ─────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    goes_idx: int
    goes_class: str
    goes_peak_time: datetime
    l3_event: Optional[NowcastEvent]
    lead_time_min: Optional[float]
    l3_goes_class: Optional[str]  # what L3 classified it as


def match_l3_to_goes(
    l3_onsets: List[NowcastEvent],
    goes_df: pd.DataFrame,
    window_min: int = MATCH_WINDOW_MIN,
) -> Tuple[List[MatchResult], List[NowcastEvent]]:
    """
    Greedy nearest-neighbour match between L3 FLARE_ONSET events and GOES events.
    Returns (matched_results, false_alarm_events).
    """
    matched = []
    used_l3 = set()
    window_td = timedelta(minutes=window_min)

    for idx, row in goes_df.iterrows():
        goes_ts = row["peak_time"].to_pydatetime()
        if goes_ts.tzinfo is None:
            goes_ts = goes_ts.replace(tzinfo=timezone.utc)

        best: Optional[NowcastEvent] = None
        best_dt = timedelta(days=999)
        best_i = -1

        for i, ev in enumerate(l3_onsets):
            if i in used_l3:
                continue
            ev_ts = ev.timestamp
            if ev_ts.tzinfo is None:
                ev_ts = ev_ts.replace(tzinfo=timezone.utc)
            dt = abs(ev_ts - goes_ts)
            if dt <= window_td and dt < best_dt:
                best_dt = dt
                best = ev
                best_i = i

        if best is not None:
            used_l3.add(best_i)
            ev_ts = best.timestamp
            if ev_ts.tzinfo is None:
                ev_ts = ev_ts.replace(tzinfo=timezone.utc)
            lead_min = (goes_ts - ev_ts).total_seconds() / 60.0
            matched.append(MatchResult(
                goes_idx=int(idx),
                goes_class=_parse_goes_class(str(row.get("goes_class", "U"))),
                goes_peak_time=goes_ts,
                l3_event=best,
                lead_time_min=lead_min,
                l3_goes_class=best.goes_class,
            ))
        else:
            matched.append(MatchResult(
                goes_idx=int(idx),
                goes_class=_parse_goes_class(str(row.get("goes_class", "U"))),
                goes_peak_time=goes_ts,
                l3_event=None,
                lead_time_min=None,
                l3_goes_class=None,
            ))

    # unmatched L3 events = false alarms
    fa_events = [ev for i, ev in enumerate(l3_onsets) if i not in used_l3]
    return matched, fa_events


def _parse_goes_class(s: str) -> str:
    s = s.strip().upper()
    if not s:
        return "U"
    letter = s[0]
    return letter if letter in GOES_CLASS_ORDER else "U"


# ── Metrics computation ────────────────────────────────────────────────────────

def compute_metrics(
    matches: List[MatchResult],
    fa_events: List[NowcastEvent],
    total_days: float,
    label: str = "full",
) -> Dict:
    n_goes = len(matches)
    detected = [m for m in matches if m.l3_event is not None]
    missed = [m for m in matches if m.l3_event is None]
    n_det = len(detected)
    n_fa = len(fa_events)

    # recall by class group
    low_class = {"A", "B", "C"}
    high_class = {"M", "X"}

    low_goes = [m for m in matches if m.goes_class in low_class]
    high_goes = [m for m in matches if m.goes_class in high_class]
    low_det = [m for m in low_goes if m.l3_event is not None]
    high_det = [m for m in high_goes if m.l3_event is not None]

    recall_low = len(low_det) / len(low_goes) if low_goes else float("nan")
    recall_high = len(high_det) / len(high_goes) if high_goes else float("nan")
    recall_total = n_det / n_goes if n_goes else float("nan")
    precision = n_det / (n_det + n_fa) if (n_det + n_fa) > 0 else float("nan")
    fa_per_day = n_fa / total_days if total_days > 0 else float("nan")

    # GOES classification accuracy (only on matched events)
    class_correct = sum(
        1 for m in detected
        if m.l3_goes_class == m.goes_class
    )
    class_acc = class_correct / n_det if n_det > 0 else float("nan")

    # lead time stats
    lead_times = [m.lead_time_min for m in detected if m.lead_time_min is not None]
    lead_mean = float(np.mean(lead_times)) if lead_times else float("nan")
    lead_median = float(np.median(lead_times)) if lead_times else float("nan")

    # confusion matrix (detected events only)
    confusion: Dict[str, Dict[str, int]] = {}
    for cls in GOES_CLASS_ORDER:
        confusion[cls] = {c: 0 for c in GOES_CLASS_ORDER}
    for m in detected:
        if m.goes_class in confusion and m.l3_goes_class in confusion.get(m.goes_class, {}):
            confusion[m.goes_class][m.l3_goes_class] += 1

    return {
        "label": label,
        "n_goes_events": n_goes,
        "n_detected": n_det,
        "n_missed": n_goes - n_det,
        "n_false_alarms": n_fa,
        "total_days": round(total_days, 1),
        "recall_total": round(recall_total, 3) if recall_total == recall_total else None,
        "recall_low_class_BC": round(recall_low, 3) if recall_low == recall_low else None,
        "recall_high_class_MX": round(recall_high, 3) if recall_high == recall_high else None,
        "precision": round(precision, 3) if precision == precision else None,
        "false_alarms_per_day": round(fa_per_day, 3) if fa_per_day == fa_per_day else None,
        "goes_class_accuracy": round(class_acc, 3) if class_acc == class_acc else None,
        "lead_time_mean_min": round(lead_mean, 1) if lead_mean == lead_mean else None,
        "lead_time_median_min": round(lead_median, 1) if lead_median == lead_median else None,
        "confusion_matrix": confusion,
        "n_low_goes": len(low_goes),
        "n_high_goes": len(high_goes),
    }


def format_report(metrics: Dict, mode: str) -> str:
    lines = [
        f"=" * 60,
        f"ARUNA L3 Validation — {mode}",
        f"=" * 60,
        f"GOES events in window:    {metrics['n_goes_events']}",
        f"  B/C (low class):        {metrics['n_low_goes']}",
        f"  M/X (high class):       {metrics['n_high_goes']}",
        "",
        f"Detected:                 {metrics['n_detected']}",
        f"Missed:                   {metrics['n_missed']}",
        f"False alarms:             {metrics['n_false_alarms']}",
        f"Observation window:       {metrics['total_days']} days",
        "",
        f"Recall (total):           {metrics['recall_total']}",
        f"Recall (B/C class):       {metrics['recall_low_class_BC']}",
        f"Recall (M/X class):       {metrics['recall_high_class_MX']}",
        f"Precision:                {metrics['precision']}",
        f"False alarms/day:         {metrics['false_alarms_per_day']}",
        "",
        f"GOES class accuracy:      {metrics['goes_class_accuracy']}  ← B1 calibration quality",
        f"Lead time mean:           {metrics['lead_time_mean_min']} min",
        f"Lead time median:         {metrics['lead_time_median_min']} min",
        "",
        "GOES Classification Confusion Matrix (row=truth, col=L3 prediction):",
        "       " + "  ".join(f"{c:>5}" for c in GOES_CLASS_ORDER),
    ]
    for true_cls in GOES_CLASS_ORDER:
        row = metrics["confusion_matrix"].get(true_cls, {})
        lines.append(
            f"  {true_cls}:  " + "  ".join(f"{row.get(c, 0):>5}" for c in GOES_CLASS_ORDER)
        )
    lines.append("=" * 60)
    return "\n".join(lines)


def format_confusion_text(confusion: Dict) -> str:
    header = "       " + "  ".join(f"{c:>5}" for c in GOES_CLASS_ORDER)
    rows = [header]
    for true_cls in GOES_CLASS_ORDER:
        row = confusion.get(true_cls, {})
        rows.append(
            f"  {true_cls}:  " + "  ".join(f"{row.get(c, 0):>5}" for c in GOES_CLASS_ORDER)
        )
    return "\n".join(rows)


# ── Main run logic ─────────────────────────────────────────────────────────────

def run_validation(
    physics_df: pd.DataFrame,
    goes_df: pd.DataFrame,
    out_dir: Path,
    mode: str = "full",
    zero_hel: bool = False,
    zero_sol: bool = False,
    calibrate: bool = False,
) -> Dict:
    logger.info("Starting L3 validation — mode=%s", mode)
    cfg = NowcastConfig(db_url=f"sqlite:///{out_dir / 'catalog.db'}")
    orchestrator = NowcastOrchestrator(config=cfg, kafka_producer=None)

    ts_col = next((c for c in ["timestamp", "time", "datetime"] if c in physics_df.columns), None)
    all_events: List[NowcastEvent] = []
    n_frames = len(physics_df)

    for i, (_, row) in enumerate(physics_df.iterrows()):
        if i % 50000 == 0:
            logger.info("  Frame %d / %d (%.0f%%)", i, n_frames, 100 * i / n_frames)
        try:
            frame = row_to_physics_frame(row)
        except Exception as e:
            logger.debug("Skipping row %d: %s", i, e)
            continue
        frame = apply_zero_mode(frame, zero_hel, zero_sol)
        event = orchestrator.process(frame)
        if event is not None:
            all_events.append(event)

    orchestrator.close()
    logger.info("L3 emitted %d events total", len(all_events))

    onset_events = [e for e in all_events if e.alert_type == "FLARE_ONSET"]
    logger.info("FLARE_ONSET events: %d", len(onset_events))

    # Determine time range for FA/day calculation
    if ts_col and len(physics_df) > 0:
        t0 = pd.Timestamp(physics_df[ts_col].iloc[0])
        t1 = pd.Timestamp(physics_df[ts_col].iloc[-1])
        total_days = (t1 - t0).total_seconds() / 86400.0
    else:
        total_days = 1.0

    # Filter GOES events to same date range
    if ts_col and len(physics_df) > 0:
        t0_dt = pd.Timestamp(physics_df[ts_col].iloc[0]).to_pydatetime()
        t1_dt = pd.Timestamp(physics_df[ts_col].iloc[-1]).to_pydatetime()
        goes_window = goes_df[
            (goes_df["peak_time"] >= pd.Timestamp(t0_dt)) &
            (goes_df["peak_time"] <= pd.Timestamp(t1_dt))
        ].copy()
    else:
        goes_window = goes_df.copy()

    logger.info("GOES events in physics_cache window: %d", len(goes_window))

    matches, fa_events = match_l3_to_goes(onset_events, goes_window)
    metrics = compute_metrics(matches, fa_events, total_days, label=mode)

    # B1 calibration
    if calibrate and len([m for m in matches if m.l3_event is not None]) >= 3:
        detected = [m for m in matches if m.l3_event is not None]
        # Reconstruct peak counts from matched events
        peak_counts = []
        goes_wm2 = []
        goes_class_to_wm2 = {"B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}
        for m in detected:
            if m.l3_event and m.goes_class in goes_class_to_wm2:
                # Use midpoint of GOES class band as proxy W/m²
                base = goes_class_to_wm2[m.goes_class]
                # approximate: use current classification to back-calculate counts
                sol_flux = (m.l3_event.goes_subclass or 1.0) * base
                # get raw counts from the onset event's goes_subclass
                # (counts = sol_flux / current_factor)
                counts_approx = sol_flux / cfg.solexs_to_wm2_factor
                peak_counts.append(counts_approx)
                goes_wm2.append(base * (m.l3_event.goes_subclass or 1.0))

        if len(peak_counts) >= 3:
            calibrated_factor = GOESClassifier.calibrate_from_matched_events(
                peak_counts, goes_wm2
            )
            metrics["b1_calibrated_factor"] = calibrated_factor
            logger.info("B1 calibrated solexs_to_wm2_factor = %.3e (was %.3e)",
                        calibrated_factor, cfg.solexs_to_wm2_factor)

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    report = format_report(metrics, mode)
    print(report)
    with open(out_dir / "report.txt", "w") as f:
        f.write(report)

    with open(out_dir / "confusion.txt", "w") as f:
        f.write(format_confusion_text(metrics["confusion_matrix"]))

    # Save missed and false alarm event lists
    missed = [m for m in matches if m.l3_event is None]
    if missed:
        missed_rows = [
            {"goes_class": m.goes_class, "goes_peak_time": m.goes_peak_time.isoformat()}
            for m in missed
        ]
        pd.DataFrame(missed_rows).to_csv(out_dir / "missed_events.csv", index=False)

    if fa_events:
        fa_rows = [
            {
                "timestamp": ev.timestamp.isoformat(),
                "goes_class": ev.goes_class,
                "detection_method": ev.detection_method,
                "confidence": ev.confidence,
            }
            for ev in fa_events
        ]
        pd.DataFrame(fa_rows).to_csv(out_dir / "false_alarms.csv", index=False)

    logger.info("Results written to %s", out_dir)
    return metrics


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="ARUNA L3 Nowcasting Validation")
    p.add_argument("--physics-cache", type=Path, default=Path("data/physics_cache"),
                   help="Directory containing daily physics_cache parquet files")
    p.add_argument("--goes-events", type=Path, default=Path("data/goes_events.parquet"),
                   help="GOES flare catalog parquet file")
    p.add_argument("--out", type=Path, default=Path("experiments/l3_baseline"),
                   help="Output directory for results")
    p.add_argument("--zero-hel1os", action="store_true",
                   help="Zero HEL1OS channels (test B2/B3 fix: HEL dropout)")
    p.add_argument("--zero-solexs", action="store_true",
                   help="Zero SoLEXS channels (test B2 fix: SoLEXS dropout)")
    p.add_argument("--calibrate", action="store_true",
                   help="Fit B1 calibration factor from matched events")
    p.add_argument("--all-modes", action="store_true",
                   help="Run full + zero-hel + zero-sol in sequence and compare")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.physics_cache.exists():
        logger.error("Physics cache not found: %s", args.physics_cache)
        logger.error("Transfer physics_cache.zip from Drive and unzip to %s", args.physics_cache)
        sys.exit(1)

    if not args.goes_events.exists():
        logger.error("GOES catalog not found: %s", args.goes_events)
        logger.error("Run: python python/scripts/download_goes_labels.py "
                     "--start 2023-09-01 --end 2026-06-20 --out data/goes_events.parquet")
        sys.exit(1)

    physics_df = load_physics_cache(args.physics_cache)
    goes_df = load_goes_events(args.goes_events)

    if args.all_modes:
        results = {}
        for mode, zh, zs in [
            ("full",          False, False),
            ("zero_hel1os",   True,  False),
            ("zero_solexs",   False, True),
        ]:
            out = args.out / mode
            m = run_validation(physics_df, goes_df, out, mode=mode,
                               zero_hel=zh, zero_sol=zs, calibrate=args.calibrate)
            results[mode] = m

        print("\n" + "=" * 60)
        print("CROSS-MODE COMPARISON")
        print("=" * 60)
        headers = ["Mode", "Recall", "MX_Recall", "Precision", "FA/day", "ClassAcc"]
        print(f"  {'Mode':<16} {'Recall':>7} {'MX_Rec':>7} {'Prec':>7} {'FA/day':>7} {'ClsAcc':>7}")
        for mode, m in results.items():
            print(f"  {mode:<16} {str(m['recall_total']):>7} {str(m['recall_high_class_MX']):>7} "
                  f"{str(m['precision']):>7} {str(m['false_alarms_per_day']):>7} "
                  f"{str(m['goes_class_accuracy']):>7}")
        # save comparison
        args.out.mkdir(parents=True, exist_ok=True)
        with open(args.out / "comparison.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    else:
        mode = "zero_hel1os" if args.zero_hel1os else \
               "zero_solexs" if args.zero_solexs else "full"
        run_validation(physics_df, goes_df, args.out, mode=mode,
                       zero_hel=args.zero_hel1os, zero_sol=args.zero_solexs,
                       calibrate=args.calibrate)


if __name__ == "__main__":
    main()
