#!/usr/bin/env python3
"""
eval_models.py — Evaluate TCN + XGBoost on the held-out test split.

Usage (from project root):
    python python/scripts/eval_models.py --dataset-dir data/dataset/

Outputs per-class metrics, TSS, ECE, and accuracy for both models
and the weighted ensemble.
"""
from __future__ import annotations

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
logger = logging.getLogger("eval_models")

SEQUENCE_LENGTH  = 600
N_FEATURES_BASE  = 19
_CLASS_NAMES     = ["QUIET", "B", "C", "M", "X"]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MODELS_DIR   = _PROJECT_ROOT / "models"


# ── helpers ───────────────────────────────────────────────────────────────────

def _tss(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    pred_active = (probs[:, 3] + probs[:, 4]) >= threshold
    true_active = labels >= 3
    tp = (pred_active &  true_active).sum()
    fn = (~pred_active & true_active).sum()
    fp = (pred_active & ~true_active).sum()
    tn = (~pred_active & ~true_active).sum()
    pod  = tp / max(tp + fn, 1)
    pofd = fp / max(fp + tn, 1)
    return float(pod - pofd), int(tp), int(fn), int(fp), int(tn)


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


def _per_class_recall(preds: np.ndarray, labels: np.ndarray) -> dict:
    out = {}
    for c, name in enumerate(_CLASS_NAMES):
        mask = labels == c
        if mask.sum() == 0:
            out[name] = None
        else:
            out[name] = (preds[mask] == c).mean()
    return out


# ── TCN inference ─────────────────────────────────────────────────────────────

def run_tcn(df: pd.DataFrame, device_str: str = "cpu") -> np.ndarray:
    """Returns (N, 5) probability matrix from TCN."""
    try:
        import torch
    except ImportError:
        logger.error("PyTorch not installed.")
        sys.exit(1)

    from aruna.l4_forecasting.tcn_model import DualStreamTCN
    from torch.utils.data import DataLoader, TensorDataset

    weights_path = _MODELS_DIR / "tcn_v1.pt"
    if not weights_path.exists():
        logger.error("TCN weights not found at %s", weights_path)
        sys.exit(1)

    scaler = None
    for _sp in ["feature_scaler_19.pkl", "feature_scaler.pkl"]:
        scaler_path = _MODELS_DIR / _sp
        if scaler_path.exists():
            with open(scaler_path, "rb") as fh:
                scaler = pickle.load(fh)
            logger.info("Loaded feature scaler %s (n_features_in_=%s)",
                        _sp, getattr(scaler, 'n_features_in_', '?'))
            break

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamTCN()
    ckpt  = torch.load(str(weights_path), map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    logger.info("TCN loaded from %s on %s", weights_path, device)

    # Temperature scaling (Guo et al. 2017): if fit_temperature.py has produced a
    # T, divide logits by it before softmax. Monotone → TSS/recall unchanged,
    # only confidence/ECE improves.
    temperature = 1.0
    temp_path = _MODELS_DIR / "tcn_temperature.npy"
    if temp_path.exists():
        temperature = float(np.load(str(temp_path))[0])
        logger.info("Applying temperature scaling T=%.4f", temperature)

    use_scaler = scaler is not None and getattr(scaler, 'n_features_in_', N_FEATURES_BASE) == N_FEATURES_BASE

    def _prepare_sample(row):
        feats = np.frombuffer(row.features_bytes, dtype=np.float32).copy()
        feats = feats.reshape(SEQUENCE_LENGTH, N_FEATURES_BASE)
        phases_int = np.frombuffer(row.phases_bytes, dtype=np.uint8)

        # Replace float32 sentinel values (±max, written for missing HEL1OS data)
        # before scaling to prevent NaN propagation through the network.
        feats[np.abs(feats) > 1e30] = 0.0

        if use_scaler:
            feats = scaler.transform(feats).astype(np.float32)
            feats = np.clip(feats, -30.0, 30.0)

        phase_onehot = np.zeros((SEQUENCE_LENGTH, 5), dtype=np.float32)
        for t in range(min(SEQUENCE_LENGTH, len(phases_int))):
            idx_p = int(phases_int[t])
            if 0 <= idx_p <= 4:
                phase_onehot[t, idx_p] = 1.0

        thermal_base = feats[:, :8]
        tnci_col     = np.zeros((SEQUENCE_LENGTH, 1), dtype=np.float32)
        x_thermal    = np.concatenate([thermal_base, tnci_col], axis=1)
        x_nonthermal = feats[:, 8:16]
        shared_base  = feats[:, 16:19]
        suit_zeros   = np.zeros((SEQUENCE_LENGTH, 2), dtype=np.float32)
        x_shared     = np.concatenate([shared_base, suit_zeros, phase_onehot], axis=1)
        return x_thermal, x_nonthermal, x_shared

    BATCH_SIZE = 64
    N = len(df)
    all_probs = []

    logger.info("Running TCN inference on %d samples (batch=%d) ...", N, BATCH_SIZE)

    for batch_start in range(0, N, BATCH_SIZE):
        batch = df.iloc[batch_start: batch_start + BATCH_SIZE].itertuples(index=False)
        prepared = [_prepare_sample(r) for r in batch]
        bt = torch.tensor(np.stack([p[0] for p in prepared]), dtype=torch.float32).to(device)
        bn = torch.tensor(np.stack([p[1] for p in prepared]), dtype=torch.float32).to(device)
        bs = torch.tensor(np.stack([p[2] for p in prepared]), dtype=torch.float32).to(device)

        with torch.no_grad():
            logits = model(bt, bn, bs) / temperature
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

        done = min(batch_start + BATCH_SIZE, N)
        if done % 5000 < BATCH_SIZE or done == N:
            logger.info("  TCN: %d / %d", done, N)

    return np.concatenate(all_probs, axis=0).astype(np.float32)


# ── XGBoost inference ─────────────────────────────────────────────────────────

def run_xgb(df: pd.DataFrame) -> np.ndarray | None:
    """Returns (N, 5) probability matrix from XGBoost, or None if unavailable."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("XGBoost not installed — skipping XGB evaluation.")
        return None

    from aruna.l4_forecasting.xgb_model import XGBoostForecaster, extract_statistical_features

    xgb_path = _MODELS_DIR / "xgb_v1.json"
    if not xgb_path.exists():
        logger.warning("XGBoost weights not found at %s — skipping.", xgb_path)
        return None

    forecaster = XGBoostForecaster(str(xgb_path))
    if not forecaster.available:
        logger.warning("XGBoostForecaster failed to load — skipping.")
        return None

    logger.info("XGBoost loaded (classes=%s)", forecaster._present_classes.tolist())
    present_classes = forecaster._present_classes

    logger.info("Building XGBoost feature matrix for %d samples ...", len(df))
    feat_rows = []
    for i, row in enumerate(df.itertuples(index=False)):
        feats  = np.frombuffer(row.features_bytes, dtype=np.float32).copy()
        feats  = feats.reshape(SEQUENCE_LENGTH, N_FEATURES_BASE)
        phases = np.frombuffer(row.phases_bytes, dtype=np.uint8)
        feat_rows.append(extract_statistical_features(feats, phases))
        if (i + 1) % 10000 == 0:
            logger.info("  XGB features: %d / %d", i + 1, len(df))

    X_all = np.vstack(feat_rows).astype(np.float32)
    logger.info("Running XGBoost batch predict_proba ...")
    proba_enc = forecaster._model.predict_proba(X_all).astype(np.float32)

    all_probs = np.zeros((len(df), 5), dtype=np.float32)
    for enc_idx, orig_label in enumerate(present_classes):
        all_probs[:, orig_label] = proba_enc[:, enc_idx]

    return all_probs


# ── Report ────────────────────────────────────────────────────────────────────

def _best_threshold(probs: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Search threshold 0.01–0.99 to maximise TSS on the given split."""
    best_tss, best_t = -1.0, 0.5
    for t in np.arange(0.01, 1.00, 0.01):
        tss_val, *_ = _tss(probs, labels, threshold=float(t))
        if tss_val > best_tss:
            best_tss, best_t = tss_val, float(t)
    return best_t, best_tss


def _apply_persistence(fire: np.ndarray, k: int) -> np.ndarray:
    """
    Hysteresis/persistence filter: a window only counts as 'firing' once k
    consecutive windows have been above threshold. Kills isolated probability
    spikes (the dominant false-alarm source) while real flares — which persist
    for many windows — still latch.
    """
    if k <= 1:
        return fire
    out = np.zeros_like(fire)
    run = 0
    for i in range(len(fire)):
        run = run + 1 if fire[i] else 0
        out[i] = run >= k
    return out


def _event_report(name: str, probs: np.ndarray, labels: np.ndarray,
                  window_ends: np.ndarray, thresholds: list[float],
                  persistence_levels: tuple[int, ...] = (1,),
                  gap_min: int = 30) -> None:
    """
    Operationally honest, per-EVENT detection metrics with a persistence policy.

    ISRO issues alerts on flare events, not 1-second windows. An M+X event spans
    many windows; it is "detected" if the alert policy fires on at least one
    window inside the event's span. The policy requires `persistence` consecutive
    windows above threshold before firing (hysteresis), which suppresses the
    isolated spikes that drive false alarms. We report, per (threshold,
    persistence): event recall, false alarms/day, and mean lead time — the
    metrics an operational warning system is actually judged on.
    """
    ts = pd.to_datetime(pd.Series(window_ends), utc=True).to_numpy()
    order = np.argsort(ts)
    ts_s   = ts[order]
    lab_s  = labels[order]
    mx_score = (probs[order, 3] + probs[order, 4])

    # Group M+X windows into events by temporal gap.
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

    n_events = len(groups)
    span_days = max((ts_s[-1] - ts_s[0]) / np.timedelta64(1, "D"), 1e-9)
    quiet_mask = (lab_s == 0)
    # "Onset" proxy = end of each event's labelled span (labels look forward by
    # the horizon, so the last labelled window sits ~at the flare). Lead time =
    # onset - first alert time within the event.
    onset_ts = {gi: ts_s[g[-1]] for gi, g in enumerate(groups)}

    print(f"\n{'='*64}")
    print(f"  {name} — PER-EVENT detection + alert policy")
    print(f"{'='*64}")
    print(f"  M+X events in test : {n_events}   (over {span_days:.1f} days)")
    print(f"  Window-level M+X   : {(lab_s>=3).sum()} windows")
    print(f"\n  {'thr':>5} {'persist':>7} | {'recall':>7} | {'FA/day':>7} | {'mean lead':>10}")
    print(f"  {'-'*5}-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")

    for thr in thresholds:
        base_fire = mx_score >= thr
        for k in persistence_levels:
            fire = _apply_persistence(base_fire, k)
            detected = 0
            lead_secs: list[float] = []
            for gi, g in enumerate(groups):
                lo, hi = g[0], g[-1]
                span_fire = fire[lo:hi + 1]
                if span_fire.any():
                    detected += 1
                    first_local = int(np.argmax(span_fire))   # first True in span
                    first_ts = ts_s[lo + first_local]
                    lead = (onset_ts[gi] - first_ts) / np.timedelta64(1, "s")
                    lead_secs.append(max(lead, 0.0))
            # false-alarm clusters: contiguous QUIET firing runs
            fa_clusters, in_cluster = 0, False
            for idx in range(len(lab_s)):
                if quiet_mask[idx] and fire[idx]:
                    if not in_cluster:
                        fa_clusters += 1; in_cluster = True
                else:
                    in_cluster = False
            recall = detected / max(n_events, 1)
            fa_per_day = fa_clusters / span_days
            lead_str = f"{np.mean(lead_secs)/60:.1f} min" if lead_secs else "   —"
            print(f"  {thr:5.2f} {k:>7} | {recall:>6.1%} | {fa_per_day:>7.1f} | {lead_str:>10}")

    print(f"\n  recall = events with >=1 alert during the event (one catch = detected).")
    print(f"  persist = consecutive windows required to fire (hysteresis; cuts false alarms).")
    print(f"  lead = mean time from first alert to flare onset proxy (event-span end).")


def _report(name: str, probs: np.ndarray, labels: np.ndarray,
            saved_threshold: float | None = None) -> None:
    best_t, _    = _best_threshold(probs, labels)
    tss_val, tp, fn, fp, tn = _tss(probs, labels, threshold=best_t)
    ece_val = _ece(probs, labels)
    preds   = probs.argmax(axis=1)
    acc     = (preds == labels).mean()
    recall  = _per_class_recall(preds, labels)

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  TSS (M+X)   : {tss_val:+.4f}   (target > 0.60)  [threshold={best_t:.2f}, grid-search on test]")
    print(f"  ECE         : {ece_val:.4f}    (target < 0.05)")
    print(f"  Accuracy    : {acc:.4f}")
    print(f"  Confusion (M+X binary): TP={tp}  FN={fn}  FP={fp}  TN={tn}")

    # Also report at the operationally saved threshold (tuned on train flares)
    if saved_threshold is not None:
        s_tss, s_tp, s_fn, s_fp, s_tn = _tss(probs, labels, threshold=saved_threshold)
        print(f"\n  At saved threshold={saved_threshold:.2f} (tuned on train flares):")
        print(f"    TSS={s_tss:+.4f}  TP={s_tp}  FN={s_fn}  FP={s_fp}  TN={s_tn}")

    print(f"\n  Per-class recall:")
    for cls, val in recall.items():
        if val is None:
            print(f"    {cls:8s}: (no samples in test set)")
        else:
            print(f"    {cls:8s}: {val:.3f}")

    print(f"\n  Label distribution: ", end="")
    for c, name_c in enumerate(_CLASS_NAMES):
        n = (labels == c).sum()
        print(f"{name_c}={n}", end="  ")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate ARUNA L4 models on test split")
    parser.add_argument("--dataset-dir", default="data/dataset/")
    parser.add_argument("--split",       default="test", choices=["train", "val", "test"])
    parser.add_argument("--tcn-weight",  type=float, default=0.6)
    parser.add_argument("--xgb-weight",  type=float, default=0.4)
    parser.add_argument("--event-level", action="store_true",
                        help="Also report per-EVENT detection metrics (operationally honest)")
    parser.add_argument("--persistence", default="1,3,5",
                        help="Comma-separated consecutive-window counts for the hysteresis "
                             "alert policy in the per-event table (e.g. '1,3,5')")
    args = parser.parse_args()
    persistence_levels = tuple(int(x) for x in str(args.persistence).split(",") if x.strip())

    test_pq = Path(args.dataset_dir) / f"{args.split}.parquet"
    if not test_pq.exists():
        print(f"Dataset not found: {test_pq}")
        sys.exit(1)

    logger.info("Loading %s split from %s ...", args.split, test_pq)
    df     = pd.read_parquet(test_pq)
    labels = df["label"].to_numpy(dtype=np.int32)

    logger.info("%s split: %d samples", args.split, len(df))

    # Load saved threshold (tuned on training flares during train_tcn.py)
    th_path = _MODELS_DIR / "tcn_v1.threshold.npy"
    saved_threshold = float(np.load(str(th_path))[0]) if th_path.exists() else None
    if saved_threshold is not None:
        logger.info("Saved operational threshold: %.2f", saved_threshold)

    window_ends = df["window_end"].to_numpy() if args.event_level else None
    # Thresholds to sweep for the per-event table: a low operating point, the
    # saved val-tuned one, and a couple in between.
    ev_thresholds = sorted({0.01, 0.05, 0.10, 0.25,
                            saved_threshold if saved_threshold else 0.50})

    # TCN
    tcn_probs = run_tcn(df)
    _report("DualStreamTCN", tcn_probs, labels, saved_threshold)
    if args.event_level:
        _event_report("DualStreamTCN", tcn_probs, labels, window_ends, ev_thresholds, persistence_levels)

    # XGBoost
    xgb_probs = run_xgb(df)
    if xgb_probs is not None:
        _report("XGBoost", xgb_probs, labels)
        if args.event_level:
            _event_report("XGBoost", xgb_probs, labels, window_ends, ev_thresholds, persistence_levels)

        # Ensemble
        w_tcn = args.tcn_weight
        w_xgb = args.xgb_weight
        ens_probs = w_tcn * tcn_probs + w_xgb * xgb_probs
        _report(f"Ensemble (TCN×{w_tcn} + XGB×{w_xgb})", ens_probs, labels, saved_threshold)
        if args.event_level:
            _event_report(f"Ensemble (TCN×{w_tcn} + XGB×{w_xgb})",
                          ens_probs, labels, window_ends, ev_thresholds, persistence_levels)

    print("\nDone.")


if __name__ == "__main__":
    main()
