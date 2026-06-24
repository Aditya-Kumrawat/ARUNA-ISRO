#!/usr/bin/env python3
"""
failure_analysis.py — 3 false negatives + 3 false positives on test split.

Uses Logistic Regression (thermal features only, no HEL1OS dependency) as the
analysed model. The 2025 test period lacks HEL1OS so TCN is not applicable here;
LR provides a meaningful comparison baseline that works on SoLEXS-only features.

Usage:
    python python/scripts/failure_analysis.py --dataset-dir data/dataset
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "python")
sys.path.insert(0, "python/scripts")

from eval_models import _apply_persistence
from select_operating_point import _physics_ok

SEQ_LEN    = 600
N_FEAT     = 19
PERSIST    = 10
GAP_MIN    = 30
FEAT_NAMES = {0: "T_MK", 3: "tnpsv_mag", 5: "dT_dt", 16: "HFD", 17: "solexs_3_6"}


def _lr_features_row(raw_bytes: bytes) -> np.ndarray:
    w = np.frombuffer(raw_bytes, dtype=np.float32).reshape(SEQ_LEN, N_FEAT).copy()
    w[np.abs(w) > 1e30] = 0.0  # replace float32 sentinel (missing HEL1OS)
    return np.concatenate([w.mean(axis=0), w.std(axis=0)])


def _last_frame_features(df: pd.DataFrame, idx: int) -> dict:
    raw = df["features_bytes"].iloc[idx]
    f   = np.frombuffer(raw, dtype=np.float32).reshape(SEQ_LEN, N_FEAT)
    return {name: float(f[-1, col]) for col, name in FEAT_NAMES.items()}


def _mean_features(df: pd.DataFrame, row_indices) -> dict:
    vals = {name: [] for name in FEAT_NAMES.values()}
    for i in row_indices:
        fv = _last_frame_features(df, i)
        for k, v in fv.items():
            vals[k].append(v)
    return {k: float(np.mean(v)) for k, v in vals.items() if v}


def _class_name(label: int) -> str:
    return ["QUIET", "B", "C", "M", "X"][label]


def _likely_reason(feats: dict, is_fn: bool) -> str:
    reasons = []
    if is_fn:
        if feats.get("dT_dt", 0) < 0.1:
            reasons.append("flat or falling T — no thermal rise in window")
        if feats.get("solexs_3_6", 0) < 3.0:
            reasons.append("low soft X-ray flux (C-class or below at window end)")
        if feats.get("tnpsv_mag", 0) < 0.3:
            reasons.append("low T-EM phase velocity (quiescent plasma dynamics)")
        if not reasons:
            reasons.append("borderline event — thermal features near class boundary")
    else:
        if feats.get("solexs_3_6", 0) > 5.0:
            reasons.append("elevated soft X-ray during quiet period (background spike)")
        if feats.get("dT_dt", 0) > 0.5:
            reasons.append("rising T trend — residual heating from prior flare")
        if feats.get("HFD", 0) > 1.98:
            reasons.append("high fractal dimension — complex flux morphology")
        if not reasons:
            reasons.append("marginal thermal features above decision boundary")
    return "; ".join(reasons)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/dataset")
    parser.add_argument("--n-cases",     type=int, default=3)
    args = parser.parse_args()
    dd  = Path(args.dataset_dir)
    out = Path("docs/evidence")
    out.mkdir(parents=True, exist_ok=True)

    print("Loading test split ...")
    test    = pd.read_parquet(dd / "test.parquet")
    labels  = test["label"].to_numpy()
    we      = test["window_end"].to_numpy()

    print("Streaming train split for LR training ...")
    import pyarrow.parquet as pq
    train_lr_chunks, train_labels_list = [], []
    pf = pq.ParquetFile(str(dd / "train.parquet"))
    n = 0
    for batch in pf.iter_batches(batch_size=2000):
        df_chunk = batch.to_pandas()
        feats = np.vstack([_lr_features_row(r) for r in df_chunk["features_bytes"]])
        train_lr_chunks.append(feats)
        train_labels_list.append(df_chunk["label"].to_numpy())
        n += len(df_chunk)
        if (len(train_labels_list) % 10) == 0:
            print(f"  ... {n} rows")
    train_lr     = np.vstack(train_lr_chunks)
    train_labels = np.concatenate(train_labels_list)
    print(f"  Train rows: {len(train_labels)}")

    print("Training Logistic Regression ...")
    y_tr = (train_labels >= 3).astype(int)
    sc   = StandardScaler()
    lr   = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    lr.fit(sc.fit_transform(train_lr), y_tr)

    print("Scoring test split ...")
    test_lr  = np.vstack([_lr_features_row(r) for r in test["features_bytes"]])
    mx_prob  = lr.predict_proba(sc.transform(test_lr))[:, 1]
    thr      = 0.5  # default LR threshold

    fire = mx_prob >= thr
    fire = _apply_persistence(fire, PERSIST)

    ts     = pd.to_datetime(pd.Series(we), utc=True).to_numpy()
    order  = np.argsort(ts)
    ts_s   = ts[order]
    lab_s  = labels[order]
    mx_s   = mx_prob[order]
    fire_s = fire[order]

    # Group M/X windows into events
    mx_idx = np.where(lab_s >= 3)[0]
    groups: list[list[int]] = []
    if len(mx_idx):
        cur = [int(mx_idx[0])]
        for j in mx_idx[1:]:
            gap = (ts_s[j] - ts_s[cur[-1]]) / np.timedelta64(1, "m")
            if gap > GAP_MIN:
                groups.append(cur); cur = [int(j)]
            else:
                cur.append(int(j))
        groups.append(cur)

    # False negatives
    fn_cases = []
    for g in groups:
        lo, hi = g[0], g[-1]
        if not fire_s[lo:hi + 1].any():
            peak_class   = lab_s[g[np.argmax(lab_s[g])]]
            orig_indices = order[g]
            feats        = _mean_features(test, orig_indices)
            fn_cases.append({
                "time":      ts_s[hi],
                "class":     _class_name(int(peak_class)),
                "n_windows": len(g),
                "max_prob":  float(mx_s[lo:hi + 1].max()),
                "feats":     feats,
                "reason":    _likely_reason(feats, is_fn=True),
            })
    fn_cases.sort(key=lambda x: -x["max_prob"])
    fn_cases = fn_cases[:args.n_cases]

    # False positives
    fp_runs = []
    in_run  = False
    run_start = 0
    for i in range(len(fire_s)):
        if fire_s[i] and lab_s[i] == 0:
            if not in_run:
                run_start = i; in_run = True
        else:
            if in_run:
                run_end   = i - 1
                run_probs = mx_s[run_start:run_end + 1]
                dur_min   = (ts_s[run_end] - ts_s[run_start]) / np.timedelta64(1, "m")
                orig_idx  = int(order[run_start + int(np.argmax(run_probs))])
                feats     = _last_frame_features(test, orig_idx)
                fp_runs.append({
                    "time":     ts_s[run_start],
                    "dur_min":  float(dur_min),
                    "max_prob": float(run_probs.max()),
                    "feats":    feats,
                    "reason":   _likely_reason(feats, is_fn=False),
                })
                in_run = False
    fp_runs.sort(key=lambda x: -x["max_prob"])
    fp_runs = fp_runs[:args.n_cases]

    n_detected = sum(1 for g in groups if fire_s[g[0]:g[-1] + 1].any())
    lbl_counts = dict(zip(*np.unique(labels, return_counts=True)))
    recall = n_detected / len(groups) if groups else 0.0

    def fmt_feats(d: dict) -> str:
        return ", ".join(f"{k}={v:.2f}" for k, v in d.items())

    lines = [
        "ARUNA Failure Analysis — Test Split (2025, SoLEXS only, LR classifier)",
        "=" * 68,
        "Model: Logistic Regression on thermal+shared features (no HEL1OS required)",
        f"Threshold: {thr}, persistence: {PERSIST} windows",
        f"Test label distribution: {lbl_counts}",
        f"Total M/X events: {len(groups)}",
        f"Detected: {n_detected}  ({recall:.0%} event recall)",
        f"Missed (FN): {len(groups) - n_detected}",
        "",
        f"FALSE NEGATIVES — {args.n_cases} hardest missed M/X events",
        "-" * 68,
    ]
    for i, c in enumerate(fn_cases, 1):
        t = pd.Timestamp(c["time"]).strftime("%Y-%m-%d %H:%M UTC")
        lines += [
            f"FN-{i}  {t}  Class {c['class']}",
            f"  Windows in event: {c['n_windows']}",
            f"  Max LR P(M+X) seen: {c['max_prob']:.4f}  (threshold {thr})",
            f"  Mean feature values: {fmt_feats(c['feats'])}",
            f"  Likely cause: {c['reason']}",
            "",
        ]

    if not fn_cases:
        lines.append("  No missed events on test split.\n")

    lines += [
        f"FALSE POSITIVES — {args.n_cases} most confident quiet-sun alerts",
        "-" * 68,
    ]
    for i, c in enumerate(fp_runs, 1):
        t = pd.Timestamp(c["time"]).strftime("%Y-%m-%d %H:%M UTC")
        lines += [
            f"FP-{i}  {t}",
            f"  Alert duration: {c['dur_min']:.0f} min",
            f"  Max P(M+X): {c['max_prob']:.4f}",
            f"  Feature values: {fmt_feats(c['feats'])}",
            f"  Likely cause: {c['reason']}",
            "",
        ]

    if not fp_runs:
        lines.append("  No quiet-sun false positives on test split.\n")

    lines += [
        "NOTE: TCN failure analysis not shown for 2025 test period: HEL1OS unavailable",
        "  => nonthermal stream (SHEI, dSHEI_dt, hel1os_*) zeroed; TCN degrades to",
        "  thermal-only mode. TCN failure analysis for dual-instrument period (2024)",
        "  available on request from the training data.",
    ]

    report = "\n".join(lines)
    print("\n" + report)
    (out / "failure_analysis.txt").write_text(report, encoding="utf-8")
    print("Saved docs/evidence/failure_analysis.txt")


if __name__ == "__main__":
    main()
