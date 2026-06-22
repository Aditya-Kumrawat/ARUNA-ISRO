#!/usr/bin/env python3
"""
train_xgb.py — Train XGBoost classifier as ensemble member for L4 forecasting.

Reads the same windowed parquet dataset built by build_dataset.py.
Extracts 43 statistical features per window, trains XGBClassifier with
asymmetric class weights (M/X weighted 4× — matches TCN asymmetric loss).

Usage:
    python python/scripts/train_xgb.py \
        --dataset-dir data/dataset/ \
        --out         models/xgb_v1.json

Outputs:
    models/xgb_v1.json   — trained XGBoost model (portable JSON format)

Training time on i5/Ryzen CPU: ~5-10 minutes.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

# Add python/ directory to path so `aruna` package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("train_xgb")

SEQUENCE_LENGTH  = 600
N_BASE_FEATURES  = 19

_CLASS_NAMES = ["QUIET", "B", "C", "M", "X"]


def _extract_row(features_bytes: bytes, phases_bytes: bytes) -> np.ndarray:
    """Extract 43 statistical features from one parquet row."""
    from aruna.l4_forecasting.xgb_model import extract_statistical_features

    feats  = np.frombuffer(features_bytes, dtype=np.float32).copy()
    feats  = feats.reshape(SEQUENCE_LENGTH, N_BASE_FEATURES)
    phases = np.frombuffer(phases_bytes,   dtype=np.uint8)

    return extract_statistical_features(feats, phases)


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Convert full parquet dataframe to (X, y) arrays."""
    rows = []
    labels = df["label"].to_numpy()

    logger.info("Extracting features from %d windows ...", len(df))
    for i, row in enumerate(df.itertuples(index=False)):
        rows.append(_extract_row(row.features_bytes, row.phases_bytes))
        if (i + 1) % 1000 == 0:
            logger.info("  %d / %d", i + 1, len(df))

    X = np.vstack(rows).astype(np.float32)
    y = labels.astype(np.int32)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XGBoost ensemble member for ARUNA L4")
    parser.add_argument("--dataset-dir", default="data/dataset/",
                        help="Directory with train.parquet and val.parquet")
    parser.add_argument("--out",         default="models/xgb_v1.json",
                        help="Output path for trained model (JSON format)")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth",    type=int, default=6)
    parser.add_argument("--lr",           type=float, default=0.05)
    parser.add_argument("--binary", action="store_true",
                        help="Train binary QUIET vs M+X classifier (simpler, often higher TSS)")
    parser.add_argument("--train-parquet", default=None,
                        help="Override training parquet (default: <dataset-dir>/train.parquet)")
    parser.add_argument("--device", default="cuda",
                        help="XGBoost device: 'cuda' or 'cpu'")
    args = parser.parse_args()

    try:
        import xgboost as xgb
    except ImportError:
        print("xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    dataset_dir = Path(args.dataset_dir)
    train_pq    = Path(args.train_parquet) if args.train_parquet else dataset_dir / "train.parquet"
    val_pq      = dataset_dir / "val.parquet"

    if not train_pq.exists():
        print(f"Dataset not found: {train_pq}. Run build_dataset.py first.")
        sys.exit(1)

    logger.info("Train parquet: %s", train_pq)
    train_df = pd.read_parquet(train_pq)
    logger.info("Loading val split ...")
    val_df   = pd.read_parquet(val_pq)

    logger.info("Building train feature matrix (%d windows) ...", len(train_df))
    X_train, y_train = build_feature_matrix(train_df)

    logger.info("Building val feature matrix (%d windows) ...", len(val_df))
    X_val, y_val = build_feature_matrix(val_df)

    if args.binary:
        # Binary mode: collapse to QUIET (0) vs M+X (1). Much simpler problem.
        # B and C become QUIET (not operationally relevant to alert).
        y_train_bin = (y_train >= 3).astype(np.int32)
        y_val_bin   = (y_val   >= 3).astype(np.int32)
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_bin).astype(np.float32)
        logger.info(
            "BINARY mode — QUIET=%d  M+X=%d  weight_quiet=%.2f  weight_mx=%.2f",
            (y_train_bin == 0).sum(), (y_train_bin == 1).sum(),
            sample_weights[y_train_bin == 0].mean(), sample_weights[y_train_bin == 1].mean(),
        )
        X_val_es, y_val_es = X_val, y_val_bin
        present_classes = np.array([-1], dtype=np.int32)  # sentinel: binary mode
        model = xgb.XGBClassifier(
            n_estimators          = args.n_estimators,
            max_depth             = args.max_depth,
            learning_rate         = args.lr,
            objective             = "binary:logistic",
            eval_metric           = ["logloss", "aucpr"],
            early_stopping_rounds = 40,
            scale_pos_weight      = float((y_train_bin == 0).sum()) / max((y_train_bin == 1).sum(), 1),
            subsample             = 0.8,
            colsample_bytree      = 0.8,
            min_child_weight      = 1,
            device                = args.device,
            n_jobs                = -1,
            verbosity             = 1,
        )
        y_train_enc = y_train_bin
        val_mask    = np.ones(len(y_val), dtype=bool)
    else:
        # Balanced sample weights — inverse class frequency from actual training distribution.
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train).astype(np.float32)
        logger.info(
            "Sample weight summary — min=%.2f  max=%.2f  M≈%.1f  X≈%.1f",
            sample_weights.min(), sample_weights.max(),
            sample_weights[y_train == 3].mean() if (y_train == 3).any() else 0.0,
            sample_weights[y_train == 4].mean() if (y_train == 4).any() else 0.0,
        )

        # Remap labels to contiguous integers — B-class (1) may be absent from dataset
        present_classes = np.array(sorted(np.unique(y_train)), dtype=np.int32)
        label_to_enc    = {int(c): i for i, c in enumerate(present_classes)}
        y_train_enc     = np.array([label_to_enc[int(l)] for l in y_train], dtype=np.int32)
        n_classes       = len(present_classes)
        logger.info("Classes present in train: %s  (mapped to 0..%d)", present_classes.tolist(), n_classes - 1)

        # Map val labels for early stopping (skip samples with classes not in train)
        val_mask    = np.array([int(l) in label_to_enc for l in y_val])
        X_val_es    = X_val[val_mask]
        y_val_es    = np.array([label_to_enc[int(l)] for l in y_val[val_mask]], dtype=np.int32)

        model = xgb.XGBClassifier(
            n_estimators          = args.n_estimators,
            max_depth             = args.max_depth,
            learning_rate         = args.lr,
            objective             = "multi:softprob",
            num_class             = n_classes,
            eval_metric           = ["mlogloss"],
            early_stopping_rounds = 40,
            subsample             = 0.8,
            colsample_bytree      = 0.8,
            min_child_weight      = 1,
            device                = args.device,
            n_jobs                = -1,
            verbosity             = 1,
        )


    model.fit(
        X_train, y_train_enc,
        sample_weight    = sample_weights,
        eval_set         = [(X_val_es, y_val_es)],
        verbose          = 20,
    )

    # Save label mapping so xgb_model.py can map back to 5-class probabilities
    out_path = Path(args.out)
    classes_path = out_path.parent / "xgb_label_classes.npy"
    np.save(str(classes_path), present_classes)
    logger.info("Label mapping saved → %s", classes_path)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_val_orig = y_val[val_mask]
    probs_enc  = model.predict_proba(X_val_es).astype(np.float32)

    if args.binary:
        # Binary: probs_enc[:,1] = P(M+X). Expand to 5-class: put it all in col 3 (M proxy).
        val_probs = np.zeros((len(probs_enc), 5), dtype=np.float32)
        val_probs[:, 0] = probs_enc[:, 0]   # QUIET
        val_probs[:, 3] = probs_enc[:, 1]   # M+X probability → M column
    else:
        val_probs = np.zeros((len(probs_enc), 5), dtype=np.float32)
        for enc_idx, orig_label in enumerate(present_classes):
            val_probs[:, orig_label] = probs_enc[:, enc_idx]

    val_preds = val_probs.argmax(axis=1)

    # TSS for M+X — sweep thresholds to find best (imbalanced data makes 0.5 wrong)
    true_active = y_val_orig >= 3
    best_tss, best_t = -1.0, 0.5
    for _t in np.arange(0.02, 0.51, 0.02):
        _pred = (val_probs[:, 3] + val_probs[:, 4]) >= _t
        _tp = (_pred & true_active).sum()
        _fn = (~_pred & true_active).sum()
        _fp = (_pred & ~true_active).sum()
        _tn = (~_pred & ~true_active).sum()
        _tss = _tp / max(_tp + _fn, 1) - _fp / max(_fp + _tn, 1)
        if _tss > best_tss:
            best_tss, best_t = _tss, _t

    pred_active = (val_probs[:, 3] + val_probs[:, 4]) >= best_t
    tp = (pred_active &  true_active).sum()
    fn = (~pred_active & true_active).sum()
    fp = (pred_active & ~true_active).sum()
    tn = (~pred_active & ~true_active).sum()
    tss  = best_tss
    acc  = (val_preds == y_val_orig).mean()

    logger.info("Val  TSS=%.4f  Acc=%.4f  threshold=%.2f  (target TSS > 0.60)", tss, acc, best_t)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(out_path))
    logger.info("Saved → %s", out_path)

    print("\n" + "=" * 55)
    print("XGB TRAINING COMPLETE")
    print(f"  Val TSS        : {tss:.4f}  (target > 0.60)  [threshold={best_t:.2f}]")
    print(f"  Val Accuracy   : {acc:.4f}")
    print(f"  Model saved    : {out_path}")
    print("\nNow run the full pipeline:")
    print("  python -m aruna.pipeline.run --fits-dir python/tests/fixtures/x63/")


if __name__ == "__main__":
    main()
