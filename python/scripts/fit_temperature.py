#!/usr/bin/env python3
"""
fit_temperature.py — Post-hoc temperature scaling for the trained DualStreamTCN.

Fits a single scalar T and divides the logits by T before softmax. This is the
standard route to low ECE (Guo et al. 2017, "On Calibration of Modern Neural
Networks"). It is monotone in the logits, so argmax and the M+X ranking are
unchanged — TSS and per-class recall at any given threshold are preserved.
Only the *confidence* (and therefore ECE) changes.

Fitted on the VALIDATION set only (val is for model selection/calibration);
the test split is never touched, so this is leak-free.

Usage (from project root):
    python python/scripts/fit_temperature.py \
        --dataset-dir data/dataset \
        --scaler-path models/feature_scaler_19.pkl \
        --tcn-out     models/tcn_v1.pt \
        --out         models/tcn_temperature.npy
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
logger = logging.getLogger("fit_temperature")

SEQUENCE_LENGTH = 600
N_FEATURES_BASE = 19


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    conf  = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    correct = (preds == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (conf > bins[i]) & (conf <= bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(labels) * abs(conf[mask].mean() - correct[mask].mean())
    return float(ece)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _tcn_logits(df: pd.DataFrame, weights_path: Path, scaler) -> np.ndarray:
    """Run the TCN over df and return raw (N, 5) logits (no softmax)."""
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
    out = []
    for s in range(0, N, BATCH):
        prepared = [prep(r) for r in df.iloc[s:s + BATCH].itertuples(index=False)]
        bt = torch.tensor(np.stack([p[0] for p in prepared]), dtype=torch.float32).to(device)
        bn = torch.tensor(np.stack([p[1] for p in prepared]), dtype=torch.float32).to(device)
        bs = torch.tensor(np.stack([p[2] for p in prepared]), dtype=torch.float32).to(device)
        with torch.no_grad():
            out.append(model(bt, bn, bs).cpu().numpy())
        if (s + BATCH) % 5000 < BATCH:
            logger.info("  logits: %d / %d", min(s + BATCH, N), N)
    return np.concatenate(out, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit temperature scaling for ARUNA TCN")
    parser.add_argument("--dataset-dir", default="data/dataset")
    parser.add_argument("--scaler-path", default="models/feature_scaler_19.pkl")
    parser.add_argument("--tcn-out",     default="models/tcn_v1.pt")
    parser.add_argument("--out",         default="models/tcn_temperature.npy")
    args = parser.parse_args()

    import torch

    val_pq = Path(args.dataset_dir) / "val.parquet"
    if not val_pq.exists():
        print(f"Val split not found: {val_pq}")
        sys.exit(1)

    scaler = None
    if Path(args.scaler_path).exists():
        with open(args.scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
        logger.info("Loaded scaler %s (n_features_in_=%s)",
                    args.scaler_path, getattr(scaler, "n_features_in_", "?"))

    logger.info("Loading val split ...")
    val_df = pd.read_parquet(val_pq)
    labels = val_df["label"].to_numpy(dtype=np.int64)
    logger.info("Val windows: %d", len(val_df))

    logits = _tcn_logits(val_df, Path(args.tcn_out), scaler)

    # Fit T by minimising NLL on val (single scalar, LBFGS).
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_T    = torch.nn.Parameter(torch.zeros(1))   # optimise log T so T stays > 0
    optimizer = torch.optim.LBFGS([log_T], lr=0.05, max_iter=200)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits_t / torch.exp(log_T), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(torch.exp(log_T).detach().item())

    ece_before = _ece(_softmax(logits),     labels)
    ece_after  = _ece(_softmax(logits / T), labels)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, np.array([T], dtype=np.float32))

    print("\n" + "=" * 55)
    print("TEMPERATURE SCALING COMPLETE")
    print(f"  Fitted T        : {T:.4f}   (>1 = model was over-confident)")
    print(f"  Val ECE before  : {ece_before:.4f}")
    print(f"  Val ECE after   : {ece_after:.4f}   (target < 0.02)")
    print(f"  Saved           : {args.out}")
    print("\neval_models.py will auto-apply this T if the file is present.")


if __name__ == "__main__":
    main()
