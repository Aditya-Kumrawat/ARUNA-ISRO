#!/usr/bin/env python3
"""
train_tcn.py — Train DualStreamTCN on the pre-built windowed dataset,
run MAPIE conformal calibration on the validation set, and save all
model artifacts.

Usage:
    python python/scripts/train_tcn.py \
        --dataset-dir  data/dataset/ \
        --scaler-path  models/feature_scaler.pkl \
        --tcn-out      models/tcn_v1.pt \
        --mapie-out    models/mapie_calibration.pkl \
        --epochs       100 \
        --batch-size   64 \
        --lr           1e-3 \
        --device       cpu

Outputs:
    models/tcn_v1.pt            — best checkpoint by validation TSS
    models/mapie_calibration.pkl — MAPIE conformal calibrator
    models/training_log.csv     — per-epoch metrics

The script is resumable: if tcn_v1.pt already exists it will be loaded
as the starting checkpoint (use --no-resume to train from scratch).
"""
from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("train_tcn")

# ── Constants matching build_dataset.py ───────────────────────────────────────

SEQUENCE_LENGTH = 600
N_THERMAL    = 9    # must match DualStreamTCN defaults
N_NONTHERMAL = 8
N_SHARED     = 10   # 3 base cols + 5 one-hot phase cols = 8... but model uses 10
                    # features.py adds suit_uv_flux, suit_uv_brightening_rate (→ 0 if absent)
N_FEATURES_BASE = 22   # 8 thermal + 8 nonthermal + 3 shared_base (no phase yet)

_PHASE_INDEX = {"QUIET": 0, "HOPE": 1, "IMPULSIVE": 2, "GRADUAL": 3, "DECAY": 4}
_CLASS_NAMES = ["QUIET", "B", "C", "M", "X"]
_CLASS_WEIGHTS_OVERSAMPLE = [1.0, 3.0, 5.0, 10.0, 20.0]   # over-sample rare classes


# ── Dataset ────────────────────────────────────────────────────────────────────

class SolarFlareDataset:
    """
    PyTorch Dataset backed by a Parquet file of pre-built windows.

    Each row contains:
      features_bytes  — float32 bytes, shape (SEQUENCE_LENGTH, N_FEATURES_BASE)
      phases_bytes    — str bytes, shape (SEQUENCE_LENGTH,)  for one-hot encoding
      label           — int 0-4
    """

    def __init__(self, parquet_path: str, scaler=None) -> None:
        import torch
        self._torch = torch
        self._df = pd.read_parquet(parquet_path)
        self._scaler = scaler

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        import torch
        row = self._df.iloc[idx]

        # Decode feature bytes → (T, N_FEATURES_BASE)
        feats = np.frombuffer(row["features_bytes"], dtype=np.float32).copy()
        feats = feats.reshape(SEQUENCE_LENGTH, N_FEATURES_BASE)

        # Decode phase bytes → (T,) uint8 integers (0=QUIET..4=DECAY) → one-hot (T, 5)
        # build_dataset.py stores phases as uint8 ints, one byte per timestep.
        phases_int = np.frombuffer(row["phases_bytes"], dtype=np.uint8)
        phase_onehot = np.zeros((SEQUENCE_LENGTH, 5), dtype=np.float32)
        for t in range(min(SEQUENCE_LENGTH, len(phases_int))):
            idx_p = int(phases_int[t])
            if 0 <= idx_p <= 4:
                phase_onehot[t, idx_p] = 1.0

        # Apply scaler to base features
        if self._scaler is not None:
            feats = self._scaler.transform(feats).astype(np.float32)

        # Split into streams matching DualStreamTCN input
        # Thermal:    cols 0:8  (T_MK, EM, DNI, tnpsv_mag, tnpsv_ang, dT, dEM, d2T)
        # Nonthermal: cols 8:16 (SHEI, dSHEI, FeKa, hel bands ×3, QPP×2)
        # Shared base: cols 16:19 (HFD, solexs_3_6, solexs_6_10)
        # → pad to (T, 9) thermal by appending zeros for tnci (PRD §13.4)
        # → pad shared to (T, 10) by appending suit_uv_flux, suit_uv_brightening_rate=0
        # → concat shared_base + phase_onehot → (T, 8) then pad to 10

        thermal_base = feats[:, :8]                             # (T, 8)
        tnci_col     = np.zeros((SEQUENCE_LENGTH, 1), dtype=np.float32)
        x_thermal    = np.concatenate([thermal_base, tnci_col], axis=1)  # (T, 9)

        x_nonthermal = feats[:, 8:16]                          # (T, 8)

        shared_base  = feats[:, 16:19]                         # (T, 3)
        suit_zeros   = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)  # suit UV → 0
        x_shared     = np.concatenate([shared_base, suit_zeros, phase_onehot], axis=1)  # (T, 10)

        label = int(row["label"])

        # Soft label with ε=0.1 label smoothing
        soft = np.full(5, 0.1 / 4, dtype=np.float32)
        soft[label] = 0.9

        return (
            torch.tensor(x_thermal,    dtype=torch.float32),
            torch.tensor(x_nonthermal, dtype=torch.float32),
            torch.tensor(x_shared,     dtype=torch.float32),
            torch.tensor(soft,         dtype=torch.float32),
            label,
        )


def _make_sampler(dataset: SolarFlareDataset):
    """WeightedRandomSampler: over-sample M and X class windows."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    labels = dataset._df["label"].to_numpy()
    weights = np.array([_CLASS_WEIGHTS_OVERSAMPLE[l] for l in labels], dtype=np.float32)
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights),
        num_samples=len(weights),
        replacement=True,
    )


# ── Metrics ────────────────────────────────────────────────────────────────────

def _compute_tss(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    """
    Binary TSS for M+X class (label >= 3).
    TSS = POD - POFD = TP/(TP+FN) - FP/(FP+TN)
    """
    pred_active = (probs[:, 3] + probs[:, 4]) >= threshold
    true_active = labels >= 3

    tp = (pred_active &  true_active).sum()
    fn = (~pred_active & true_active).sum()
    fp = (pred_active & ~true_active).sum()
    tn = (~pred_active & ~true_active).sum()

    pod  = tp / (tp + fn + 1e-9)
    pofd = fp / (fp + tn + 1e-9)
    return float(pod - pofd)


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error on the predicted class probability."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct     = (predictions == labels).astype(float)

    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean()
        avg_acc  = correct[mask].mean()
        ece += mask.sum() / len(labels) * abs(avg_conf - avg_acc)
    return float(ece)


# ── Training loop ──────────────────────────────────────────────────────────────

def train(
    dataset_dir: str,
    scaler_path: str,
    tcn_out: str,
    mapie_out: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device_str: str,
    no_resume: bool,
) -> None:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
    from aruna.l4_forecasting.tcn_model import DualStreamTCN, asymmetric_focal_loss

    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu"
                          else "cpu")
    logger.info("Using device: %s", device)

    # Load scaler
    scaler = None
    if Path(scaler_path).exists():
        with open(scaler_path, "rb") as fh:
            scaler = pickle.load(fh)
        logger.info("Loaded feature scaler from %s", scaler_path)
    else:
        logger.warning("No scaler found at %s — using raw values", scaler_path)

    # Datasets
    train_pq = Path(dataset_dir) / "train.parquet"
    val_pq   = Path(dataset_dir) / "val.parquet"
    if not train_pq.exists() or not val_pq.exists():
        print(f"Dataset not found in {dataset_dir}. Run build_dataset.py first.")
        sys.exit(1)

    train_ds = SolarFlareDataset(str(train_pq), scaler)
    val_ds   = SolarFlareDataset(str(val_pq),   scaler)
    logger.info("Train windows: %d  |  Val windows: %d", len(train_ds), len(val_ds))

    sampler = _make_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=(device_str != "cpu"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0, pin_memory=(device_str != "cpu"))

    # Model
    model = DualStreamTCN(n_thermal=N_THERMAL, n_nonthermal=N_NONTHERMAL, n_shared=N_SHARED)
    model.to(device)

    if not no_resume and Path(tcn_out).exists():
        state = torch.load(tcn_out, map_location=device)
        model.load_state_dict(state)
        logger.info("Resumed from existing checkpoint %s", tcn_out)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    Path(tcn_out).parent.mkdir(parents=True, exist_ok=True)
    log_path = Path(tcn_out).parent / "training_log.csv"

    best_tss   = -999.0
    patience   = 10
    no_improve = 0

    with open(log_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_tss", "val_ece", "lr"])

        for epoch in range(1, epochs + 1):
            # ── Train ─────────────────────────────────────────────────────────
            model.train()
            epoch_loss = 0.0
            t0 = time.monotonic()

            for x_th, x_nth, x_sh, soft_labels, _ in train_loader:
                x_th   = x_th.to(device)
                x_nth  = x_nth.to(device)
                x_sh   = x_sh.to(device)
                soft_labels = soft_labels.to(device)

                optimizer.zero_grad()
                logits = model(x_th, x_nth, x_sh)
                loss   = asymmetric_focal_loss(logits, soft_labels, alpha=0.2, gamma=2.0)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

            train_loss = epoch_loss / len(train_loader)
            scheduler.step()

            # ── Validate ──────────────────────────────────────────────────────
            model.eval()
            val_loss = 0.0
            all_probs: list[np.ndarray] = []
            all_labels: list[int] = []

            with torch.no_grad():
                for x_th, x_nth, x_sh, soft_labels, hard_labels in val_loader:
                    x_th  = x_th.to(device)
                    x_nth = x_nth.to(device)
                    x_sh  = x_sh.to(device)
                    soft_labels = soft_labels.to(device)

                    logits = model(x_th, x_nth, x_sh)
                    loss   = asymmetric_focal_loss(logits, soft_labels, alpha=0.2, gamma=2.0)
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    all_probs.append(probs)
                    all_labels.extend(hard_labels.tolist())

            val_loss  /= len(val_loader)
            all_probs_np = np.concatenate(all_probs, axis=0)
            all_labels_np = np.array(all_labels)

            val_tss = _compute_tss(all_probs_np, all_labels_np)
            val_ece = _compute_ece(all_probs_np, all_labels_np)
            cur_lr  = scheduler.get_last_lr()[0]
            elapsed = time.monotonic() - t0

            logger.info(
                "Epoch %3d/%d  loss=%.4f  val_loss=%.4f  TSS=%.4f  ECE=%.4f  lr=%.2e  %.1fs",
                epoch, epochs, train_loss, val_loss, val_tss, val_ece, cur_lr, elapsed,
            )
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                             f"{val_tss:.4f}", f"{val_ece:.4f}", f"{cur_lr:.2e}"])
            csvfile.flush()

            # ── Checkpoint ────────────────────────────────────────────────────
            if val_tss > best_tss:
                best_tss   = val_tss
                no_improve = 0
                torch.save(model.state_dict(), tcn_out)
                logger.info("  → New best TSS=%.4f — saved %s", best_tss, tcn_out)
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info("Early stopping: no TSS improvement for %d epochs", patience)
                    break

    logger.info("Training complete. Best val TSS = %.4f", best_tss)
    logger.info("Checkpoint saved → %s", tcn_out)

    # ── MAPIE calibration ──────────────────────────────────────────────────────
    logger.info("Running MAPIE conformal calibration on validation set ...")
    try:
        from mapie.classification import MapieClassifier
        from sklearn.base import BaseEstimator, ClassifierMixin

        # Reload best checkpoint for final val pass
        model.load_state_dict(torch.load(tcn_out, map_location=device))
        model.eval()

        val_probs: list[np.ndarray] = []
        val_labels_list: list[int] = []

        with torch.no_grad():
            for x_th, x_nth, x_sh, _, hard_labels in DataLoader(
                val_ds, batch_size=batch_size, shuffle=False
            ):
                logits = model(x_th.to(device), x_nth.to(device), x_sh.to(device))
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                val_probs.append(probs)
                val_labels_list.extend(hard_labels.tolist())

        val_probs_np  = np.concatenate(val_probs, axis=0)
        val_labels_np = np.array(val_labels_list)

        # MAPIE needs a fitted sklearn estimator. Wrap pre-computed probabilities.
        class _ProbaPassthrough(BaseEstimator, ClassifierMixin):
            def __init__(self, probs_):
                self.probs_ = probs_
                self.classes_ = np.arange(5)
            def fit(self, X, y): return self
            def predict(self, X): return self.probs_.argmax(axis=1)
            def predict_proba(self, X): return self.probs_

        mapie = MapieClassifier(
            estimator=_ProbaPassthrough(val_probs_np),
            method="score",
            cv="prefit",
        )
        mapie.fit(val_probs_np, val_labels_np)

        Path(mapie_out).parent.mkdir(parents=True, exist_ok=True)
        with open(mapie_out, "wb") as fh:
            pickle.dump(mapie, fh)
        logger.info("MAPIE calibrator saved -> %s", mapie_out)

        # ECE uses raw TCN probabilities; MAPIE provides conformal prediction SETS
        # (not recalibrated probabilities), so compute ECE on raw val_probs_np.
        ece_final = _compute_ece(val_probs_np, val_labels_np)
        tss_final = _compute_tss(val_probs_np, val_labels_np)
        logger.info("Final val  TSS=%.4f  ECE=%.4f  (target ECE < 0.05)", tss_final, ece_final)

        if ece_final > 0.05:
            logger.warning("ECE %.4f exceeds PRD target 0.05", ece_final)

    except Exception as exc:
        logger.warning("MAPIE calibration skipped: %s", exc)
        logger.warning("TCN weights are still saved. Run 'pip install mapie scikit-learn' and re-run to add calibration.")

    # ── Final summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Best validation TSS : {best_tss:.4f}  (target > 0.60)")
    print(f"  TCN weights         : {tcn_out}")
    print(f"  MAPIE calibrator    : {mapie_out}")
    print(f"  Training log        : {log_path}")
    print("\nTo evaluate on the test set:")
    print("  python python/scripts/train_tcn.py --eval-only --dataset-dir data/dataset/")
    print("\nTo run the live pipeline with trained weights:")
    print("  python -m aruna.pipeline.run --fits-dir data/solexs/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DualStreamTCN on ARUNA dataset")
    parser.add_argument("--dataset-dir",  default="data/dataset")
    parser.add_argument("--scaler-path",  default="models/feature_scaler.pkl")
    parser.add_argument("--tcn-out",      default="models/tcn_v1.pt")
    parser.add_argument("--mapie-out",    default="models/mapie_calibration.pkl")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--device",       default="cuda",
                        help="'cuda' or 'cpu'. Falls back to cpu if cuda unavailable.")
    parser.add_argument("--no-resume",    action="store_true",
                        help="Train from scratch even if checkpoint exists")
    args = parser.parse_args()
    train(
        args.dataset_dir, args.scaler_path, args.tcn_out, args.mapie_out,
        args.epochs, args.batch_size, args.lr, args.device, args.no_resume,
    )


if __name__ == "__main__":
    main()
