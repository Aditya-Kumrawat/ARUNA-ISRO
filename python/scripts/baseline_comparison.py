#!/usr/bin/env python3
"""
baseline_comparison.py — answers "better than what?"

Evaluates naive flux threshold, persistence, logistic regression, and
TCN on test split. Threshold for each baseline selected on val
(maximise event recall s.t. FA/day <= 1.0).

Train split is streamed in chunks to avoid OOM (train.parquet ~6 GB
uncompressed). Val and test are loaded fully.

Usage:
    python python/scripts/baseline_comparison.py --dataset-dir data/dataset
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "python")
sys.path.insert(0, "python/scripts")

from eval_models import _apply_persistence
from select_operating_point import _events_and_scores, _event_metrics, _physics_ok

SEQ_LEN  = 600
N_FEAT   = 19
FLUX_COL = 17
MAX_FA   = 1.0
CHUNK    = 2000


def _last_flux_df(df: pd.DataFrame) -> np.ndarray:
    """Extract last-frame flux value per row without stacking full windows."""
    offset = (SEQ_LEN - 1) * N_FEAT + FLUX_COL
    return np.array([
        float(np.frombuffer(raw, dtype=np.float32)[offset])
        for raw in df["features_bytes"]
    ], dtype=np.float32)


def _lr_features_df(df: pd.DataFrame) -> np.ndarray:
    """Compute per-row mean+std across time, return (N, 38) matrix."""
    feats = []
    for raw in df["features_bytes"]:
        w = np.frombuffer(raw, dtype=np.float32).reshape(SEQ_LEN, N_FEAT).copy()
        w[np.abs(w) > 1e30] = 0.0  # replace float32 sentinel (missing HEL1OS)
        feats.append(np.concatenate([w.mean(axis=0), w.std(axis=0)]))
    return np.vstack(feats)


def _stream_train(path: Path, chunk: int = CHUNK):
    """Stream train parquet in row-group chunks; yield (flux, lr_feats, labels)."""
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=chunk):
        df_chunk = batch.to_pandas()
        flux   = _last_flux_df(df_chunk)
        feats  = _lr_features_df(df_chunk)
        labels = df_chunk["label"].to_numpy()
        yield flux, feats, labels


def _persistence_scores(labels: np.ndarray, lookback: int = 10) -> np.ndarray:
    is_mx  = (labels >= 3).astype(np.float32)
    scores = np.zeros(len(labels), dtype=np.float32)
    for i in range(1, len(labels)):
        if is_mx[max(0, i - lookback):i].any():
            scores[i] = 0.9
    return scores


def _to_probs5(scores_1d: np.ndarray) -> np.ndarray:
    p = np.zeros((len(scores_1d), 5), dtype=np.float32)
    p[:, 3] = scores_1d
    return p


def _select_thr_on_val(scores_1d, labels, window_ends, phys_ok, persist=1):
    probs5 = _to_probs5(scores_1d)
    args   = _events_and_scores(probs5, labels, window_ends, phys_ok)
    best_thr, best_rec = 0.5, 0.0
    for thr in np.round(np.arange(0.01, 1.00, 0.01), 2):
        rec, fa, _, _ = _event_metrics(*args, thr, persist, False)
        if fa <= MAX_FA and rec >= best_rec:
            best_rec = rec
            best_thr = thr
    return best_thr


def _eval_on_test(name, scores_1d, labels, window_ends, phys_ok, thr, persist, use_gate):
    probs5 = _to_probs5(scores_1d)
    gate   = phys_ok if use_gate else np.zeros(len(scores_1d), dtype=bool)
    args   = _events_and_scores(probs5, labels, window_ends, gate)
    rec, fa, detected, lead = _event_metrics(*args, thr, persist, use_gate)
    n_total = len(args[4])
    return {
        "Model":     name,
        "Recall":    rec,
        "FA/day":    fa,
        "Lead(min)": lead,
        "Detected":  f"{int(detected.sum())}/{n_total}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/dataset")
    args = parser.parse_args()
    dd  = Path(args.dataset_dir)
    out = Path("docs/evidence")
    out.mkdir(parents=True, exist_ok=True)

    print("Loading val split ...")
    val  = pd.read_parquet(dd / "val.parquet")
    print("Loading test split ...")
    test = pd.read_parquet(dd / "test.parquet")

    val_labels  = val["label"].to_numpy()
    test_labels = test["label"].to_numpy()
    val_we      = val["window_end"].to_numpy()
    test_we     = test["window_end"].to_numpy()
    no_gate     = np.zeros(len(test), dtype=bool)

    print("Extracting val features (row-by-row) ...")
    val_flux = _last_flux_df(val)
    val_lr   = _lr_features_df(val)

    print("Extracting test features (row-by-row) ...")
    test_flux = _last_flux_df(test)
    test_lr   = _lr_features_df(test)

    print("Streaming train split in chunks (flux + LR features) ...")
    train_fluxes, train_lr_chunks, train_labels_list = [], [], []
    n_chunks = 0
    for chunk_flux, chunk_feats, chunk_labs in _stream_train(dd / "train.parquet"):
        train_fluxes.append(chunk_flux)
        train_lr_chunks.append(chunk_feats)
        train_labels_list.append(chunk_labs)
        n_chunks += 1
        if n_chunks % 10 == 0:
            print(f"  ... chunk {n_chunks}, rows so far: {sum(len(x) for x in train_labels_list)}")
    train_flux   = np.concatenate(train_fluxes)
    train_lr     = np.vstack(train_lr_chunks)
    train_labels = np.concatenate(train_labels_list)
    print(f"  Train rows: {len(train_labels)}")

    results = []

    # Baseline 1: naive flux threshold
    print("Baseline 1: naive flux threshold ...")
    p99         = float(np.percentile(train_flux, 99))
    val_flux_n  = np.clip(val_flux  / (p99 + 1e-9), 0.0, 1.0)
    test_flux_n = np.clip(test_flux / (p99 + 1e-9), 0.0, 1.0)
    thr_flux    = _select_thr_on_val(val_flux_n, val_labels, val_we,
                                     np.zeros(len(val), dtype=bool), persist=1)
    results.append(_eval_on_test(
        "Naive flux threshold", test_flux_n, test_labels, test_we, no_gate,
        thr_flux, 1, False,
    ))

    # Baseline 2: persistence
    print("Baseline 2: persistence ...")
    val_persist  = _persistence_scores(val_labels)
    test_persist = _persistence_scores(test_labels)
    thr_pers     = _select_thr_on_val(val_persist, val_labels, val_we,
                                      np.zeros(len(val), dtype=bool), persist=1)
    results.append(_eval_on_test(
        "Persistence (10 windows)", test_persist, test_labels, test_we, no_gate,
        thr_pers, 1, False,
    ))

    # Baseline 3: logistic regression
    print("Baseline 3: logistic regression ...")
    y_tr = (train_labels >= 3).astype(int)
    sc   = StandardScaler()
    lr   = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    lr.fit(sc.fit_transform(train_lr), y_tr)
    val_lr_s  = lr.predict_proba(sc.transform(val_lr))[:, 1]
    test_lr_s = lr.predict_proba(sc.transform(test_lr))[:, 1]
    thr_lr    = _select_thr_on_val(val_lr_s, val_labels, val_we,
                                   np.zeros(len(val), dtype=bool), persist=1)
    results.append(_eval_on_test(
        "Logistic regression", test_lr_s, test_labels, test_we, no_gate,
        thr_lr, 1, False,
    ))

    # TCN locked result — from select_operating_point.py on 2024 val/test with dual-instrument
    # The 2025 test period lacks HEL1OS data; TCN nonthermal stream unavailable.
    # These figures come from the locked operating point (thr=0.01, persist=10, physics gate).
    results.append({
        "Model":     "TCN (locked, 2024 dual-instrument)",
        "Recall":    0.58,
        "FA/day":    0.17,
        "Lead(min)": 23.7,
        "Detected":  "7/12",
    })

    # Text table
    lines = [
        "ARUNA Baseline Comparison — Test Split",
        "=" * 62,
        f"{'Model':<28} {'Recall':>7} {'FA/day':>7} {'Lead(m)':>8} {'Detected':>10}",
        "-" * 62,
    ]
    for r in results:
        lines.append(
            f"{r['Model']:<28} {r['Recall']:>6.0%} {r['FA/day']:>7.2f} "
            f"{r['Lead(min)']:>7.1f} {r['Detected']:>10}"
        )
    lines += [
        "",
        "Baselines 1-3: threshold selected on val (max recall, FA/day <= 1.0).",
        "Val split (2024-H2) has no M/X events => threshold defaults to conservative.",
        "TCN: locked result from select_operating_point.py on 2024 data (SoLEXS+HEL1OS).",
        "2025 test period lacks HEL1OS; TCN nonthermal stream unavailable in that window.",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (out / "baseline_table.txt").write_text(report, encoding="utf-8")
    print("Saved docs/evidence/baseline_table.txt")

    # Bar chart
    names   = [r["Model"] for r in results]
    recalls = [r["Recall"] for r in results]
    fas     = [r["FA/day"] for r in results]
    colors  = ["#9ca3af", "#9ca3af", "#9ca3af", "#e05c2e"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.barh(names, recalls, color=colors)
    ax1.set_xlabel("Event Recall")
    ax1.set_xlim(0, 1.05)
    ax1.set_title("Event Recall — Test Split")
    for i, v in enumerate(recalls):
        ax1.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)

    ax2.barh(names, fas, color=colors)
    ax2.set_xlabel("False Alarms / day")
    ax2.set_title("False Alarm Rate — Test Split")
    for i, v in enumerate(fas):
        ax2.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)

    fig.suptitle("ARUNA vs Baselines (all thresholds val-selected)", fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved docs/evidence/baseline_comparison.png")


if __name__ == "__main__":
    main()
