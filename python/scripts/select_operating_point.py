#!/usr/bin/env python3
"""
select_operating_point.py — Choose the operational alert policy on the
VALIDATION set, then report the resulting per-EVENT metrics on TEST.

Reading the best (threshold, persistence) off the test table is leakage: a
reviewer will rightly ask how threshold 0.01 was chosen. This selects the
policy on val — maximise val event recall subject to val false-alarms/day <=
--max-fa-per-day — then applies that *fixed* policy to test, with a bootstrap
95% CI on test event recall (honest given ~12 test events).

Optional --physics-gate adds a physics-informed AND condition to every alert:
the model probability must clear threshold AND the thermal signature must be
rising (dT/dt > 0 or dEM/dt > 0 over the last minute). This suppresses quiet
fluctuations (which have flat/falling T) while real flare onsets — which heat
the plasma — still pass. It is also the explainable story: "ARUNA does not
alert unless the physics agrees."

Usage (from project root):
    python python/scripts/select_operating_point.py \
        --dataset-dir data/dataset --max-fa-per-day 1.0 \
        --persistence 1,3,5,10 --physics-gate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))          # eval_models
sys.path.insert(0, str(Path(__file__).parent.parent))   # aruna package

from eval_models import run_tcn, _apply_persistence

SEQUENCE_LENGTH = 600
N_FEATURES_BASE = 19
_DT_IDX  = 5    # dT_dt   in FEATURE_COLS
_DEM_IDX = 6    # dEM_dt  in FEATURE_COLS (symmetric-log stored, sign preserved)


def _physics_ok(df: pd.DataFrame, lookback: int = 60) -> np.ndarray:
    """Per-window boolean: thermal signature rising (dT/dt>0 or dEM/dt>0) over
    the last `lookback` timesteps. Read from raw (unscaled) feature bytes."""
    ok = np.zeros(len(df), dtype=bool)
    for i, raw in enumerate(df["features_bytes"].to_numpy()):
        f = np.frombuffer(raw, dtype=np.float32).reshape(SEQUENCE_LENGTH, N_FEATURES_BASE)
        dT  = f[-lookback:, _DT_IDX].mean()
        dEM = f[-lookback:, _DEM_IDX].mean()
        ok[i] = (dT > 0.0) or (dEM > 0.0)
    return ok


def _events_and_scores(probs, labels, window_ends, physics_ok, gap_min: int = 30):
    ts = pd.to_datetime(pd.Series(window_ends), utc=True).to_numpy()
    order = np.argsort(ts)
    ts_s   = ts[order]
    lab_s  = labels[order]
    mx     = probs[order, 3] + probs[order, 4]
    phys_s = physics_ok[order]

    mx_idx = np.where(lab_s >= 3)[0]
    groups: list[list[int]] = []
    if len(mx_idx):
        cur = [int(mx_idx[0])]
        for j in mx_idx[1:]:
            if (ts_s[j] - ts_s[cur[-1]]) / np.timedelta64(1, "m") > gap_min:
                groups.append(cur); cur = [int(j)]
            else:
                cur.append(int(j))
        groups.append(cur)
    span_days = max((ts_s[-1] - ts_s[0]) / np.timedelta64(1, "D"), 1e-9)
    return ts_s, lab_s, mx, phys_s, groups, span_days


def _event_metrics(ts_s, lab_s, mx, phys_s, groups, span_days, thr, persist, use_gate):
    fire = mx >= thr
    if use_gate:
        fire = fire & phys_s
    fire = _apply_persistence(fire, persist)

    detected = np.zeros(len(groups), dtype=bool)
    leads = []
    for gi, g in enumerate(groups):
        lo, hi = g[0], g[-1]
        span_fire = fire[lo:hi + 1]
        if span_fire.any():
            detected[gi] = True
            first = lo + int(np.argmax(span_fire))
            leads.append(max((ts_s[hi] - ts_s[first]) / np.timedelta64(1, "s"), 0.0))

    quiet = lab_s == 0
    fa, inrun = 0, False
    for i in range(len(lab_s)):
        if quiet[i] and fire[i]:
            if not inrun:
                fa += 1; inrun = True
        else:
            inrun = False
    recall = detected.mean() if len(groups) else 0.0
    mean_lead = (np.mean(leads) / 60.0) if leads else 0.0
    return recall, fa / span_days, detected, mean_lead


def main() -> None:
    parser = argparse.ArgumentParser(description="Select alert policy on val, report on test")
    parser.add_argument("--dataset-dir",    default="data/dataset")
    parser.add_argument("--max-fa-per-day", default="1.0",
                        help="One budget, or comma-separated budgets to map the operating curve "
                             "(e.g. '0.5,1,2,5'). Threshold is selected on val at each budget.")
    parser.add_argument("--persistence",    default="1,3,5,10")
    parser.add_argument("--physics-gate",   action="store_true",
                        help="Require rising thermal signature (dT/dt>0 or dEM/dt>0) for every alert")
    parser.add_argument("--n-boot",         type=int, default=2000)
    args = parser.parse_args()

    use_gate = args.physics_gate
    persist_levels = [int(x) for x in args.persistence.split(",") if x.strip()]
    fa_budgets = [float(x) for x in str(args.max_fa_per_day).split(",") if x.strip()]
    thr_grid = np.round(np.arange(0.01, 1.00, 0.01), 2)
    print(f"Physics gate: {'ON (dT/dt>0 or dEM/dt>0)' if use_gate else 'OFF'}")

    # ── Run inference ONCE on each split, then sweep FA budgets ──────────────
    val  = pd.read_parquet(Path(args.dataset_dir) / "val.parquet")
    print(f"VAL: {len(val)} windows — running TCN ...")
    val_phys = _physics_ok(val) if use_gate else np.zeros(len(val), bool)
    v = _events_and_scores(run_tcn(val), val["label"].to_numpy(),
                           val["window_end"].to_numpy(), val_phys)

    test = pd.read_parquet(Path(args.dataset_dir) / "test.parquet")
    print(f"TEST: {len(test)} windows — running TCN ...")
    test_phys = _physics_ok(test) if use_gate else np.zeros(len(test), bool)
    t = _events_and_scores(run_tcn(test), test["label"].to_numpy(),
                           test["window_end"].to_numpy(), test_phys)

    n_test = len(t[4])  # groups
    rng = np.random.default_rng(42)

    print("\n" + "=" * 78)
    print("OPERATING CURVE — threshold selected on VAL at each FA budget, reported on TEST")
    print("=" * 78)
    print(f"  {'FA budget':>9} | {'thr':>4} {'pst':>3} | {'val rec':>7} {'val FA':>6} "
          f"| {'TEST recall (95% CI)':>22} {'TEST FA':>7} {'lead':>6}")
    print(f"  {'-'*9}-+-{'-'*4}-{'-'*3}-+-{'-'*7}-{'-'*6}-+-{'-'*22}-{'-'*7}-{'-'*6}")

    for budget in fa_budgets:
        best = None
        for p in persist_levels:
            for thr in thr_grid:
                rec, fa, _, lead = _event_metrics(*v, thr, p, use_gate)
                if fa <= budget:
                    key = (rec, -fa, p)
                    if best is None or key > (best[2], -best[3], best[1]):
                        best = (thr, p, rec, fa, lead)
        if best is None:
            print(f"  {budget:>9.2f} | (no val operating point under this budget)")
            continue
        thr, p, vrec, vfa, _ = best
        trec, tfa, detected, tlead = _event_metrics(*t, thr, p, use_gate)
        lo = hi = 0.0
        if n_test > 0:
            boots = [detected[rng.integers(0, n_test, n_test)].mean() for _ in range(args.n_boot)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
        ci = f"{trec:.0%} [{lo:.0%},{hi:.0%}] {int(detected.sum())}/{n_test}"
        print(f"  {budget:>9.2f} | {thr:>4.2f} {p:>3} | {vrec:>6.0%} {vfa:>6.2f} "
              f"| {ci:>22} {tfa:>7.2f} {tlead:>4.1f}m")

    print("\n  Every threshold was selected on val (never test). Pick the operational")
    print("  FA tolerance ISRO accepts; the TEST column is the honest expected performance.")


if __name__ == "__main__":
    main()
