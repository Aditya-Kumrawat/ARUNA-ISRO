#!/usr/bin/env python3
"""
select_operating_point.py — Choose the operational (threshold, persistence) on
the VALIDATION set, then report the resulting per-EVENT metrics on TEST.

Reading the best operating point off the test table is leakage: a reviewer will
rightly ask how threshold 0.01 was chosen. This selects the policy on val —
maximise val event recall subject to val false-alarms/day <= --max-fa-per-day —
then applies that *fixed* policy to test. It also reports a bootstrap 95% CI on
test event recall, because with ~12 test events "11/12" needs an honest band.

Usage (from project root):
    python python/scripts/select_operating_point.py \
        --dataset-dir data/dataset --max-fa-per-day 1.0 --persistence 1,3,5,10
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


def _events_and_scores(probs, labels, window_ends, gap_min: int = 30):
    ts = pd.to_datetime(pd.Series(window_ends), utc=True).to_numpy()
    order = np.argsort(ts)
    ts_s   = ts[order]
    lab_s  = labels[order]
    mx     = probs[order, 3] + probs[order, 4]

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
    return ts_s, lab_s, mx, groups, span_days


def _event_metrics(ts_s, lab_s, mx, groups, span_days, thr, persist):
    fire = _apply_persistence(mx >= thr, persist)
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
    parser = argparse.ArgumentParser(description="Select operating point on val, report on test")
    parser.add_argument("--dataset-dir",    default="data/dataset")
    parser.add_argument("--max-fa-per-day", type=float, default=1.0,
                        help="Max tolerated false alarms/day on val when selecting")
    parser.add_argument("--persistence",    default="1,3,5,10")
    parser.add_argument("--n-boot",         type=int, default=2000)
    args = parser.parse_args()

    persist_levels = [int(x) for x in args.persistence.split(",") if x.strip()]
    thr_grid = np.round(np.arange(0.01, 1.00, 0.01), 2)

    # ── 1. Select on VAL ────────────────────────────────────────────────────
    val = pd.read_parquet(Path(args.dataset_dir) / "val.parquet")
    print(f"VAL: {len(val)} windows — running TCN ...")
    v = _events_and_scores(run_tcn(val), val["label"].to_numpy(), val["window_end"].to_numpy())

    best = None  # (thr, persist, recall, fa, lead)
    for p in persist_levels:
        for thr in thr_grid:
            rec, fa, _, lead = _event_metrics(*v, thr, p)
            if fa <= args.max_fa_per_day:
                # maximise recall; tie-break on lower FA, then higher persistence (more robust)
                key = (rec, -fa, p)
                if best is None or key > (best[2], -best[3], best[1]):
                    best = (thr, p, rec, fa, lead)

    if best is None:
        print(f"\nNo operating point on val meets FA/day <= {args.max_fa_per_day}. "
              f"Relax --max-fa-per-day.")
        return

    thr, p, vrec, vfa, vlead = best
    print(f"\nSelected on VAL  ->  threshold={thr:.2f}  persistence={p}")
    print(f"   val: recall={vrec:.1%}  FA/day={vfa:.2f}  mean_lead={vlead:.1f} min")

    # ── 2. Apply fixed policy to TEST ───────────────────────────────────────
    test = pd.read_parquet(Path(args.dataset_dir) / "test.parquet")
    print(f"\nTEST: {len(test)} windows — running TCN ...")
    t = _events_and_scores(run_tcn(test), test["label"].to_numpy(), test["window_end"].to_numpy())
    rec, fa, detected, lead = _event_metrics(*t, thr, p)

    n = len(detected)
    lo = hi = 0.0
    if n > 0:
        rng = np.random.default_rng(42)
        boots = [detected[rng.integers(0, n, n)].mean() for _ in range(args.n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])

    print("\n" + "=" * 60)
    print("OPERATIONAL RESULT (policy fixed on val, measured on test)")
    print("=" * 60)
    print(f"  Operating point : threshold={thr:.2f}, persistence={p} consecutive windows")
    print(f"  Events detected : {int(detected.sum())}/{n}")
    print(f"  Event recall    : {rec:.1%}   95% CI [{lo:.1%}, {hi:.1%}]  (bootstrap, n={n} events)")
    print(f"  False alarms    : {fa:.2f}/day")
    print(f"  Mean lead time  : {lead:.1f} min")
    print("\nThis operating point was NOT tuned on test — it is the honest number to report.")


if __name__ == "__main__":
    main()
