"""
Forecast skill metrics for ARUNA — PRD §19, §20.
All metric functions operate on List[MatchedEvent].
All metrics follow WMO Space Weather conventions.

Binary metrics are computed on M+X vs rest:
  positive class  = M or X (operationally significant flare)
  negative class  = QUIET, B, C (no significant threat)

TSS = POD - POFD  (Youden's J statistic)
  POD  = TP / (TP + FN)   [also called TPR / recall / sensitivity]
  POFD = FP / (FP + TN)   [False Detection Rate, aka FPR]
  FAR  = FP / (TP + FP)   [False Alarm Ratio = 1 - precision]
"""
from __future__ import annotations

import logging
import random
from typing import Callable, Dict, List, Tuple

import numpy as np

from aruna.validation.models import MatchedEvent, ValidationResult
from aruna.validation.reliability import compute_ece

logger = logging.getLogger(__name__)

CLASS_ORDER = ["QUIET", "B", "C", "M", "X"]
_M_PLUS = {"M", "X"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _binary_counts(events: List[MatchedEvent]) -> Tuple[int, int, int, int]:
    """Return (TP, FP, FN, TN) for M+X binary classification."""
    tp = fp = fn = tn = 0
    for e in events:
        pred_pos = e.predicted_class in _M_PLUS
        true_pos = e.actual_class in _M_PLUS
        if pred_pos and true_pos:
            tp += 1
        elif pred_pos and not true_pos:
            fp += 1
        elif not pred_pos and true_pos:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


# ── Binary detection metrics ──────────────────────────────────────────────────

def compute_pod(events: List[MatchedEvent]) -> float:
    """Probability of Detection = TP / (TP + FN). Returns 0.0 if no positives."""
    tp, _, fn, _ = _binary_counts(events)
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def compute_far(events: List[MatchedEvent]) -> float:
    """False Alarm Ratio = FP / (TP + FP). Returns 0.0 if no predicted positives."""
    tp, fp, _, _ = _binary_counts(events)
    denom = tp + fp
    return fp / denom if denom > 0 else 0.0


def compute_pofd(events: List[MatchedEvent]) -> float:
    """Probability of False Detection = FP / (FP + TN). Returns 0.0 if no negatives."""
    _, fp, _, tn = _binary_counts(events)
    denom = fp + tn
    return fp / denom if denom > 0 else 0.0


def compute_tss(events: List[MatchedEvent]) -> float:
    """True Skill Statistic = POD - POFD. Range [-1, 1]; > 0 better than random."""
    return compute_pod(events) - compute_pofd(events)


def compute_csi(events: List[MatchedEvent]) -> float:
    """Critical Success Index = TP / (TP + FP + FN). Range [0, 1]."""
    tp, fp, fn, _ = _binary_counts(events)
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


# ── Multi-class metrics ───────────────────────────────────────────────────────

def compute_confusion_matrix(events: List[MatchedEvent]) -> List[List[int]]:
    """5×5 confusion matrix. Rows = actual, columns = predicted. Order: QUIET B C M X."""
    idx = {c: i for i, c in enumerate(CLASS_ORDER)}
    mat = [[0] * 5 for _ in range(5)]
    for e in events:
        r = idx.get(e.actual_class, 0)
        c = idx.get(e.predicted_class, 0)
        mat[r][c] += 1
    return mat


def compute_kappa(events: List[MatchedEvent]) -> float:
    """Cohen's Kappa. Returns 0.0 if only one class present."""
    try:
        from sklearn.metrics import cohen_kappa_score
        actual = [e.actual_class for e in events]
        predicted = [e.predicted_class for e in events]
        return float(cohen_kappa_score(actual, predicted, labels=CLASS_ORDER))
    except Exception:
        return 0.0


def compute_hss(events: List[MatchedEvent]) -> float:
    """
    Heidke Skill Score for binary M+X classification.
    HSS = 2(TP*TN - FP*FN) / ((TP+FN)(FN+TN) + (TP+FP)(FP+TN))
    """
    tp, fp, fn, tn = _binary_counts(events)
    num = 2 * (tp * tn - fp * fn)
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return num / denom if denom != 0 else 0.0


# ── Probabilistic metrics ─────────────────────────────────────────────────────

def compute_brier(events: List[MatchedEvent]) -> float:
    """Brier Score for M+X binary probability. Lower is better."""
    if not events:
        return 0.0
    probs = np.array([e.predicted_prob_M_plus for e in events])
    actuals = np.array([1.0 if e.actual_class in _M_PLUS else 0.0 for e in events])
    return float(np.mean((probs - actuals) ** 2))


def compute_roc_auc(events: List[MatchedEvent]) -> float:
    """ROC-AUC for binary M+X detection. Returns 0.5 if only one class present."""
    try:
        from sklearn.metrics import roc_auc_score
        probs = [e.predicted_prob_M_plus for e in events]
        labels = [1 if e.actual_class in _M_PLUS else 0 for e in events]
        if len(set(labels)) < 2:
            return 0.5
        return float(roc_auc_score(labels, probs))
    except Exception:
        return 0.5


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    metric_fn: Callable[[List[MatchedEvent]], float],
    events: List[MatchedEvent],
    n: int = 2000,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI (n=2000, 95% by default — PRD §20.4).
    Returns (low, high).
    """
    if len(events) < 2:
        val = metric_fn(events)
        return (val, val)

    scores = []
    for _ in range(n):
        sample = random.choices(events, k=len(events))
        try:
            scores.append(metric_fn(sample))
        except Exception:
            continue

    if not scores:
        val = metric_fn(events)
        return (val, val)

    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(scores, alpha * 100))
    high = float(np.percentile(scores, (1.0 - alpha) * 100))
    return (low, high)


# ── Full suite ────────────────────────────────────────────────────────────────

def compute_all(
    events: List[MatchedEvent],
    period_start=None,
    period_end=None,
    bootstrap_n: int = 2000,
) -> ValidationResult:
    """Compute all metrics + bootstrap CIs over a list of MatchedEvents."""
    from datetime import datetime, timezone

    tp, fp, fn, tn = _binary_counts(events)
    pod   = compute_pod(events)
    far   = compute_far(events)
    pofd  = compute_pofd(events)
    tss   = compute_tss(events)
    csi   = compute_csi(events)
    hss   = compute_hss(events)
    kappa = compute_kappa(events)
    brier = compute_brier(events)
    roc   = compute_roc_auc(events)

    # ECE over full batch (not per-frame)
    probs  = np.array([e.predicted_prob_M_plus for e in events])
    labels = np.array([1.0 if e.actual_class in _M_PLUS else 0.0 for e in events])
    if len(probs) > 0:
        ece, bin_centres, bin_accs, _ = compute_ece(probs, labels)
    else:
        ece = 0.0
        bin_centres = bin_accs = np.array([])

    # Lead time (for matched HOPE events — use delta_sec as proxy)
    lead_times = [abs(e.delta_sec) for e in events if e.matched and e.delta_sec != 0]
    hope_median = float(np.median(lead_times)) if lead_times else 0.0
    hope_mae    = float(np.mean(lead_times))   if lead_times else 0.0

    # Bootstrap CIs for key metrics
    ci: Dict[str, Tuple[float, float]] = {}
    if len(events) >= 5:
        for name, metric_fn in [
            ("pod", compute_pod), ("far", compute_far), ("tss", compute_tss),
            ("csi", compute_csi), ("hss", compute_hss), ("kappa", compute_kappa),
            ("brier", compute_brier),
        ]:
            ci[name] = bootstrap_ci(metric_fn, events, n=bootstrap_n)
    else:
        for name in ("pod", "far", "tss", "csi", "hss", "kappa", "brier"):
            val = locals()[name] if name in ("pod", "far", "tss", "csi", "hss", "kappa", "brier") else 0.0
            ci[name] = (val, val)

    now = datetime.now(timezone.utc)
    return ValidationResult(
        confusion_matrix=compute_confusion_matrix(events),
        tp=tp, fp=fp, fn=fn, tn=tn,
        pod=pod, far=far, pofd=pofd, tss=tss, csi=csi,
        hss=hss, kappa=kappa,
        brier_score=brier, roc_auc=roc,
        ece=ece,
        reliability_bins=list(bin_centres) if len(bin_centres) else [],
        reliability_fracs=list(bin_accs)   if len(bin_accs) else [],
        hope_lead_time_median_sec=hope_median,
        hope_lead_time_mae_sec=hope_mae,
        ci=ci,
        n_events=len(events),
        n_matched=sum(1 for e in events if e.matched),
        period_start=period_start or now,
        period_end=period_end or now,
        bootstrap_resamples=bootstrap_n,
    )
