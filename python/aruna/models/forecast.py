"""
Global data contract: L4 output.
ForecastFrame carries calibrated flare probability for the next N minutes
with MAPIE conformal prediction intervals.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

FORECAST_HORIZONS = (5, 10, 15, 30, 60)   # minutes
CLASS_LABELS = ("QUIET", "B", "C", "M", "X")


@dataclass
class ForecastFrame:
    """
    Calibrated flare probability forecast.
    Published to: aruna.forecast.frames
    """

    timestamp: datetime
    horizon_minutes: int               # one of FORECAST_HORIZONS

    # ── Probability vector (5 classes) ───────────────────────────────────────
    prob_quiet: float
    prob_B: float
    prob_C: float
    prob_M: float
    prob_X: float
    predicted_class: str               # argmax class label
    peak_class_prob: float             # probability of predicted class

    # ── MAPIE conformal uncertainty (90% coverage guaranteed) ────────────────
    confidence_interval_low: float
    confidence_interval_high: float
    prediction_set: List[str]          # conformal prediction set (≥1 class)
    calibration_ece: float             # expected calibration error

    # ── Lead time estimate ───────────────────────────────────────────────────
    lead_time_estimate_sec: int
    lead_time_uncertainty_sec: int

    # ── Attribution ──────────────────────────────────────────────────────────
    tcn_weight: float
    moirai_weight: float
    hope_flag_active: bool
    physics_state: str

    # ── Input window provenance ──────────────────────────────────────────────
    input_window_start: datetime
    input_window_end: datetime
    model_version: str

    # ── Auto-generated ───────────────────────────────────────────────────────
    forecast_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ── Message bus metadata ─────────────────────────────────────────────────
    schema_version: str = "1.0"
    pipeline_version: str = "2.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.horizon_minutes not in FORECAST_HORIZONS:
            raise ValueError(f"horizon_minutes must be one of {FORECAST_HORIZONS}")
        if self.predicted_class not in CLASS_LABELS:
            raise ValueError(f"predicted_class must be one of {CLASS_LABELS}")
        total = self.prob_quiet + self.prob_B + self.prob_C + self.prob_M + self.prob_X
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Probabilities must sum to 1.0, got {total:.4f}")

    @property
    def prob_vector(self) -> tuple[float, float, float, float, float]:
        return (self.prob_quiet, self.prob_B, self.prob_C, self.prob_M, self.prob_X)

    def prob_major_flare(self) -> float:
        """P(M or X class flare) — primary operational metric."""
        return self.prob_M + self.prob_X
