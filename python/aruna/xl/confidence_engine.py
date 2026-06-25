from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Tuple

from aruna.models.forecast import ForecastFrame
from aruna.models.frame_header import FrameHeader
from aruna.xl.mpci import MPCIResult


@dataclass
class StackedForecastFrame:
    """Output of the Confidence Engine — adjusted forecast with contributor breakdown."""
    header: FrameHeader
    source_forecast_id: str
    adjusted_probabilities: Tuple[float, float, float, float, float]  # QUIET,B,C,M,X
    decision_confidence: float          # clipped to [0, 1]
    contributor_breakdown: Dict[str, float]  # name → delta applied
    schema_version: str = "1.0"
    pipeline_version: str = "2.0.0"


# P(M+X) prior per GOES subclass — rough historical base rates
_HISTORICAL_BASE: Dict[str, float] = {
    "QUIET": 0.02, "B": 0.04, "C": 0.08, "M": 0.45, "X": 0.80,
}


class ConfidenceEngine:
    """
    Composable confidence pipeline. Each contributor adds a signed delta
    to the raw P(M+X) from ForecastFrame. Final decision_confidence is
    clipped to [0, 1] and probabilities are renormalised to sum to 1.
    """

    def compute(
        self,
        forecast: ForecastFrame,
        mpci: MPCIResult,
        source: str = "LIVE",
    ) -> StackedForecastFrame:
        base = forecast.prob_M + forecast.prob_X
        breakdown: Dict[str, float] = {}

        # Contributor 1: MPCI coherence modifier
        mpci_delta = mpci.confidence_modifier * base
        breakdown["mpci"] = mpci_delta

        # Contributor 2: Physics gate — penalise if physics_state is QUIET but hope active
        if getattr(forecast, "physics_state", "QUIET") == "QUIET" and getattr(forecast, "hope_flag_active", False):
            physics_delta = -0.05
        else:
            physics_delta = 0.0
        breakdown["physics_gate"] = physics_delta

        # Contributor 3: Historical base rate bonus/penalty
        hist_base = _HISTORICAL_BASE.get(getattr(forecast, "predicted_class", "QUIET"), 0.05)
        hist_delta = (hist_base - base) * 0.10  # 10% pull toward historical mean
        breakdown["historical_base"] = hist_delta

        confidence = max(0.0, min(1.0, base + mpci_delta + physics_delta + hist_delta))

        # Redistribute the delta proportionally across the 5-class vector
        raw = [forecast.prob_quiet, forecast.prob_B, forecast.prob_C, forecast.prob_M, forecast.prob_X]
        delta_total = confidence - base
        mx_sum = raw[3] + raw[4]
        if mx_sum > 0:
            raw[3] += delta_total * (raw[3] / mx_sum)
            raw[4] += delta_total * (raw[4] / mx_sum)
        total = sum(raw)
        normalised = tuple(max(0.0, p / total) for p in raw)

        return StackedForecastFrame(
            header=FrameHeader(timestamp=datetime.utcnow(), source=source),
            source_forecast_id=str(forecast.forecast_id),
            adjusted_probabilities=normalised,
            decision_confidence=confidence,
            contributor_breakdown=breakdown,
        )
