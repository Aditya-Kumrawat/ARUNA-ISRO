from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple


@dataclass
class MatchedEvent:
    """One ARUNA detection matched (or not) against the GOES catalog."""
    predicted_class: str          # "QUIET" | "B" | "C" | "M" | "X"
    actual_class: str             # from GOES catalog; "QUIET" if no match
    predicted_prob_M_plus: float  # P(M) + P(X) for probabilistic metrics
    confidence: float             # NowcastEvent.confidence
    onset_time: datetime
    delta_sec: float              # signed onset error vs catalog (0 if no match)
    matched: bool                 # True if within ±3 min AND ±1 subclass


@dataclass
class ValidationResult:
    # ── Confusion matrix ─────────────────────────────────────────────────────
    confusion_matrix: List[List[int]]    # 5×5 rows=actual cols=predicted [QUIET,B,C,M,X]

    # ── Binary M+X detection metrics ─────────────────────────────────────────
    tp: int
    fp: int
    fn: int
    tn: int
    pod: float    # Probability of Detection = TP / (TP + FN)
    far: float    # False Alarm Ratio        = FP / (TP + FP)
    pofd: float   # Prob. of False Detection = FP / (FP + TN)  [used in TSS]
    tss: float    # True Skill Statistic     = POD - POFD
    csi: float    # Critical Success Index   = TP / (TP + FP + FN)

    # ── Multi-class metrics ───────────────────────────────────────────────────
    hss: float    # Heidke Skill Score
    kappa: float  # Cohen's Kappa

    # ── Probabilistic metrics ─────────────────────────────────────────────────
    brier_score: float     # mean((p_M_plus - o_M_plus)^2)
    roc_auc: float         # sklearn roc_auc_score on binary M+

    # ── Calibration ──────────────────────────────────────────────────────────
    ece: float                      # Expected Calibration Error (batch)
    reliability_bins: List[float]   # bin centre confidences
    reliability_fracs: List[float]  # observed fractions per bin

    # ── Lead time ─────────────────────────────────────────────────────────────
    hope_lead_time_median_sec: float
    hope_lead_time_mae_sec: float

    # ── Bootstrap CIs (95%, n=2000) ──────────────────────────────────────────
    ci: Dict[str, Tuple[float, float]]  # e.g. {"tss": (0.61, 0.74)}

    # ── Coverage ──────────────────────────────────────────────────────────────
    n_events: int
    n_matched: int
    period_start: datetime
    period_end: datetime
    bootstrap_resamples: int = 2000
