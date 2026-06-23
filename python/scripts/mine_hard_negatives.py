#!/usr/bin/env python3
"""
mine_hard_negatives.py — Hard-negative mining for the DualStreamTCN.

Scores every TRAIN-split QUIET window with the current model, finds the ones it
most confidently mis-scores as M+X (the false alarms), and appends those
specific windows — with their true QUIET label — to the balanced training
parquet. Retraining on the augmented set sharpens the decision boundary exactly
where the model is currently wrong, cutting the false-positive rate (and ECE)
with little cost to recall.

Leakage-safe: mines ONLY from the train split. val/test are never read here.

Usage (from project root):
    python python/scripts/mine_hard_negatives.py \
        --train-parquet    data/dataset/train.parquet \
        --balanced-parquet data/dataset/train_balanced_augmented.parquet \
        --scaler-path      models/feature_scaler_19.pkl \
        --tcn-out          models/tcn_v1.pt \
        --top-k            2000 \
        --dup              1 \
        --out              data/dataset/train_balanced_augmented.parquet

Then refit the scaler and retrain with --no-sampler.
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("mine_hard_negatives")

SEQUENCE_LENGTH = 600
N_FEATURES_BASE = 19


def _tcn_mx_scores(df: pd.DataFrame, weights_path: Path, scaler) -> np.ndarray:
    """Return P(M)+P(X) for each row in df."""
    import torch
    from aruna.l4_forecasting.tcn_model import DualStreamTCN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamTCN()
    ckpt  = torch.load(str(weights_path), map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("TCN loaded from %s on %s", weights_path, device)

    # Apply the same temperature as eval, if present (keeps scores consistent).
    temperature = 1.0
    temp_path = weights_path.parent / "tcn_temperature.npy"
    if temp_path.exists():
        temperature = float(np.load(str(temp_path))[0])
        logger.info("Using temperature T=%.4f", temperature)

    use_scaler = scaler is not None and getattr(scaler, "n_features_in_", N_FEATURES_BASE) == N_FEATURES_BASE

    def prep(row):
        feats = np.frombuffer(row.features_bytes, dtype=np.float32).copy()
        feats = feats.reshape(SEQUENCE_LENGTH, N_FEATURES_BASE)
        phases_int = np.frombuffer(row.phases_bytes, dtype=np.uint8)
        if use_scaler:
            feats = scaler.transform(feats).astype(np.float32)
        phase_onehot = np.zeros((SEQUENCE_LENGTH, 5), dtype=np.float32)
        for t in range(min(SEQUENCE_LENGTH, len(phases_int))):
            ip = int(phases_int[t])
            if 0 <= ip <= 4:
                phase_onehot[t, ip] = 1.0
        x_th = np.concatenate([feats[:, :8], np.zeros((SEQUENCE_LENGTH, 1), dtype=np.float32)], axis=1)
        x_nth = feats[:, 8:16]
        x_sh = np.concatenate([feats[:, 16:19], np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32), phase_onehot], axis=1)
        return x_th, x_nth, x_sh

    BATCH = 64
    N = len(df)
    scores = np.empty(N, dtype=np.float32)
    for s in range(0, N, BATCH):
        prepared = [prep(r) for r in df.iloc[s:s + BATCH].itertuples(index=False)]
        bt = torch.tensor(np.stack([p[0] for p in prepared]), dtype=torch.float32).to(device)
        bn = torch.tensor(np.stack([p[1] for p in prepared]), dtype=torch.float32).to(device)
        bs = torch.tensor(np.stack([p[2] for p in prepared]), dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(bt, bn, bs) / temperature, dim=1).cpu().numpy()
        scores[s:s + len(prepared)] = probs[:, 3] + probs[:, 4]
        if (s + BATCH) % 10000 < BATCH:
            logger.info("  scored: %d / %d", min(s + BATCH, N), N)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard-negative mining for ARUNA TCN")
    parser.add_argument("--train-parquet",    default="data/dataset/train.parquet",
                        help="Full train split to mine QUIET windows from")
    parser.add_argument("--balanced-parquet", default="data/dataset/train_balanced_augmented.parquet",
                        help="Balanced parquet to append hard negatives to")
    parser.add_argument("--scaler-path",      default="models/feature_scaler_19.pkl")
    parser.add_argument("--tcn-out",          default="models/tcn_v1.pt")
    parser.add_argument("--top-k",   type=int, default=2000,
                        help="Number of hardest QUIET windows to add")
    parser.add_argument("--dup",     type=int, default=1,
                        help="Duplicate each mined negative this many times (up-weight)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Report how many QUIET windows exceed this M+X score")
    parser.add_argument("--out",     default="data/dataset/train_balanced_augmented.parquet")
    args = parser.parse_args()

    scaler = None
    if Path(args.scaler_path).exists():
        with open(args.scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
        logger.info("Loaded scaler %s (n_features_in_=%s)",
                    args.scaler_path, getattr(scaler, "n_features_in_", "?"))

    logger.info("Loading train split %s ...", args.train_parquet)
    train_df = pd.read_parquet(args.train_parquet)
    quiet_df = train_df[train_df["label"] == 0].reset_index(drop=True)
    logger.info("Train QUIET windows: %d", len(quiet_df))

    scores = _tcn_mx_scores(quiet_df, Path(args.tcn_out), scaler)

    n_false_alarm = int((scores >= args.threshold).sum())
    logger.info("QUIET windows scoring M+X >= %.2f (current false alarms): %d (%.2f%%)",
                args.threshold, n_false_alarm, 100.0 * n_false_alarm / max(len(quiet_df), 1))

    # Take the top-K hardest (highest false-alarm score).
    k = min(args.top_k, len(quiet_df))
    hardest_idx = np.argsort(scores)[::-1][:k]
    hard_df = quiet_df.iloc[hardest_idx].copy()
    logger.info("Mining top %d hardest negatives — M+X score range [%.3f, %.3f]",
                k, float(scores[hardest_idx].min()), float(scores[hardest_idx].max()))

    # Duplicate for up-weighting.
    mined = pd.concat([hard_df] * args.dup, ignore_index=True) if args.dup > 1 else hard_df

    logger.info("Loading balanced parquet %s ...", args.balanced_parquet)
    balanced = pd.read_parquet(args.balanced_parquet)

    # Keep only the columns the balanced parquet uses (defensive).
    cols = balanced.columns.tolist()
    mined = mined[[c for c in cols if c in mined.columns]]

    out_df = pd.concat([balanced, mined], ignore_index=True)
    out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)

    _CLASS = {0: "QUIET", 1: "B", 2: "C", 3: "M", 4: "X"}
    logger.info("New balanced distribution (%d rows, +%d hard negatives):",
                len(out_df), len(mined))
    for lbl, name in _CLASS.items():
        logger.info("  %-5s: %d", name, int((out_df["label"] == lbl).sum()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)

    print("\n" + "=" * 55)
    print("HARD-NEGATIVE MINING COMPLETE")
    print(f"  Current false alarms (M+X>={args.threshold:.2f}) : {n_false_alarm}")
    print(f"  Hard negatives added                  : {len(mined)}")
    print(f"  New balanced parquet rows             : {len(out_df)}")
    print(f"  Saved                                 : {args.out}")
    print("\nNext: refit the scaler on the new parquet, then retrain with --no-sampler.")


if __name__ == "__main__":
    main()
