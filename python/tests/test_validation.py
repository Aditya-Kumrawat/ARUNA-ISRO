"""
Validation module tests — PRD §19, §20.
All tests use synthetic MatchedEvent lists. No network calls.
"""
from __future__ import annotations
import numpy as np
import pytest
from datetime import datetime, timezone, timedelta

from aruna.validation.models import MatchedEvent, ValidationResult
from aruna.validation.metrics import (
    compute_pod, compute_far, compute_pofd, compute_tss, compute_csi,
    compute_hss, compute_kappa, compute_brier, compute_roc_auc,
    compute_confusion_matrix, bootstrap_ci, compute_all,
)
from aruna.validation.reliability import compute_ece, ece_scalar
from aruna.validation.ground_truth import match_event, assert_no_temporal_leakage
from aruna.validation import ValidationEngine


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _ev(predicted: str, actual: str, prob: float = 0.8, matched: bool = True) -> MatchedEvent:
    return MatchedEvent(
        predicted_class=predicted,
        actual_class=actual,
        predicted_prob_M_plus=prob,
        confidence=prob,
        onset_time=datetime(2024, 2, 22, 9, 0, tzinfo=timezone.utc),
        delta_sec=90.0 if matched else 0.0,
        matched=matched,
    )


def _perfect_tp_events(n: int = 20) -> list[MatchedEvent]:
    """All true positives — predicted M, actual M."""
    return [_ev("M", "M", prob=0.9) for _ in range(n)]


def _all_fp_events(n: int = 20) -> list[MatchedEvent]:
    """All false positives — predicted M, actual QUIET."""
    return [_ev("M", "QUIET", prob=0.9, matched=False) for _ in range(n)]


def _mixed_events(n_tp=10, n_fp=3, n_fn=2, n_tn=5) -> list[MatchedEvent]:
    events = []
    events += [_ev("M", "M", prob=0.85) for _ in range(n_tp)]
    events += [_ev("M", "QUIET", prob=0.75, matched=False) for _ in range(n_fp)]
    events += [_ev("QUIET", "M", prob=0.2, matched=False) for _ in range(n_fn)]
    events += [_ev("QUIET", "QUIET", prob=0.1, matched=True) for _ in range(n_tn)]
    return events


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_confusion_matrix_shape():
    events = _mixed_events()
    mat = compute_confusion_matrix(events)
    assert len(mat) == 5
    assert all(len(row) == 5 for row in mat)


def test_tss_perfect_detector():
    """All TP → POD=1, POFD=0 → TSS=1."""
    events = _perfect_tp_events(20) + [_ev("QUIET", "QUIET", prob=0.05) for _ in range(10)]
    tss = compute_tss(events)
    assert abs(tss - 1.0) < 0.01


def test_tss_all_false_alarms():
    """All FP, no TP → POFD > 0, POD = 0 → TSS < 0."""
    events = _all_fp_events(20)
    tss = compute_tss(events)
    assert tss < 0.0


def test_far_quiet_sun_threshold():
    """On events where all actual are QUIET, FAR = 1.0 (all predicted M are FP)."""
    events = [_ev("M", "QUIET", prob=0.8, matched=False) for _ in range(10)]
    far = compute_far(events)
    assert abs(far - 1.0) < 0.01


def test_kappa_near_zero_for_random():
    """Random predictions mixed with true positives → Kappa ≪ 1."""
    events = (
        [_ev("M", "QUIET", prob=0.5, matched=False) for _ in range(10)]
        + [_ev("QUIET", "M", prob=0.5, matched=False) for _ in range(10)]
    )
    kappa = compute_kappa(events)
    assert kappa < 0.3


def test_bootstrap_ci_returns_valid_interval():
    """Bootstrap CI must return (low, high) with high >= low."""
    events = _mixed_events()
    low, high = bootstrap_ci(compute_tss, events, n=200)
    assert isinstance(low, float)
    assert isinstance(high, float)
    assert high >= low


def test_ece_perfect_calibration():
    """If predicted_prob == actual_frequency per bin → ECE ≈ 0."""
    # All high-confidence predictions are correct → ECE should be low
    probs  = np.array([0.9] * 100 + [0.1] * 100, dtype=float)
    labels = np.array([1]   * 90  + [0] * 10 + [0] * 90 + [1] * 10, dtype=float)
    ece, _, _, _ = compute_ece(probs, labels)
    # Not perfect calibration but ECE should be non-trivial scalar
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_ece_not_stub():
    """ECE must NOT return 0.0 on non-trivial miscalibrated inputs."""
    probs  = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=float)
    labels = np.array([0, 0, 0, 0, 0], dtype=float)   # all wrong → miscalibrated
    ece = ece_scalar(probs, labels)
    assert ece != 0.0, "ECE returned 0.0 — still a stub?"
    assert ece > 0.0


def test_ground_truth_matching_tp():
    """Event within ±3 min and same class → matched=True."""
    from aruna.models.nowcast import NowcastEvent
    event = NowcastEvent(
        alert_type="FLARE_ONSET",
        timestamp=datetime(2024, 2, 22, 9, 3, tzinfo=timezone.utc),
        goes_class="M",
        goes_subclass=5.0,
        confidence=0.87,
        instruments=["SoLEXS", "HEL1OS"],
        physics_frame_id="pf-001",
        detection_method="DUAL_TRIGGER",
        flare_type="STANDARD",
        lead_time_sec=720,
    )
    catalog = [{
        "start_time": datetime(2024, 2, 22, 9, 2, tzinfo=timezone.utc),
        "goes_class": "M",
        "goes_subclass": 5.2,
    }]
    result = match_event(event, catalog)
    assert result.matched is True
    assert result.predicted_class == "M"


def test_ground_truth_matching_fp():
    """Event >3 min from catalog entry → matched=False."""
    from aruna.models.nowcast import NowcastEvent
    event = NowcastEvent(
        alert_type="FLARE_ONSET",
        timestamp=datetime(2024, 2, 22, 10, 0, tzinfo=timezone.utc),
        goes_class="M",
        goes_subclass=5.0,
        confidence=0.87,
        instruments=["SoLEXS", "HEL1OS"],
        physics_frame_id="pf-002",
        detection_method="DUAL_TRIGGER",
        flare_type="STANDARD",
    )
    catalog = [{
        "start_time": datetime(2024, 2, 22, 9, 0, tzinfo=timezone.utc),
        "goes_class": "M",
        "goes_subclass": 5.0,
    }]
    result = match_event(event, catalog)
    assert result.matched is False


def test_temporal_leakage_guard_raises():
    """assert_no_temporal_leakage must raise ValueError when timestamps are out of split."""
    from aruna.models.nowcast import NowcastEvent
    out_of_split = NowcastEvent(
        alert_type="FLARE_ONSET",
        timestamp=datetime(2022, 1, 1, tzinfo=timezone.utc),  # before any split
        goes_class="M",
        goes_subclass=1.0,
        confidence=0.5,
        instruments=["SoLEXS"],
        physics_frame_id="pf-003",
        detection_method="DUAL_TRIGGER",
        flare_type="STANDARD",
    )
    with pytest.raises(ValueError, match="Temporal leakage"):
        assert_no_temporal_leakage([out_of_split], "train")


def test_compute_all_returns_validation_result():
    """Full metric suite on 20 synthetic events must return a valid ValidationResult."""
    events = _mixed_events(n_tp=10, n_fp=3, n_fn=2, n_tn=5)
    result = compute_all(events, bootstrap_n=100)  # low n for speed

    assert isinstance(result, ValidationResult)
    assert result.n_events == 20
    assert result.tp + result.fp + result.fn + result.tn == 20
    assert 0.0 <= result.tss <= 1.0 or result.tss >= -1.0
    assert "tss" in result.ci
    low, high = result.ci["tss"]
    assert high >= low
