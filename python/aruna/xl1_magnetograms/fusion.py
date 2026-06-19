"""
XL-1 Magnetogram Fusion — applies SHARP-derived magnetic context as a soft
prior on top of the Aditya-L1 forecast output.

Design constraints (from spec):
  • NEVER replaces the core SoLEXS + HEL1OS pipeline
  • Pipeline continues with no modification if MagneticContext is None
  • Fusion is additive/multiplicative on prob_M + prob_X only
  • Confidence modifier is bounded to avoid runaway boosting

Fusion logic:
  24h probability > 0.75 → +0.20 shift toward M/X classes (strong AR)
  24h probability > 0.50 → +0.10 shift toward M/X classes (moderate AR)
   6h probability > 0.60 → additional +0.10 (rapid flux emergence detected)
  24h probability < 0.10 → −0.08 dampening (magnetically quiet AR)
  No SHARP data         →  0.00 (forecast unchanged)

All probability vectors are re-normalised to sum=1 after modification.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from aruna.models.forecast import ForecastFrame
from .models import MagneticContext, SHARPConfig

logger = logging.getLogger(__name__)

_CLASS_LABELS = ("QUIET", "B", "C", "M", "X")
_M_IDX = 3
_X_IDX = 4


def _prob_vector(ff: ForecastFrame) -> np.ndarray:
    return np.array([
        ff.prob_quiet, ff.prob_B, ff.prob_C, ff.prob_M, ff.prob_X
    ], dtype=np.float64)


def _apply_boost(
    probs: np.ndarray,
    boost: float,
) -> np.ndarray:
    """
    Shift probability mass toward M+X classes by `boost` amount.

    Strategy: redistribute `boost` from quiet/B/C (proportional) to M/X (equal split).
    After redistribution, re-normalise to sum=1.
    """
    if abs(boost) < 1e-9:
        return probs

    p = probs.copy()
    low_classes = [0, 1, 2]   # QUIET, B, C — donor classes for positive boost
    high_classes = [3, 4]     # M, X — recipient classes

    if boost > 0:
        # Take from QUIET/B/C proportional to their current weight
        available = p[low_classes].sum()
        actual_boost = min(boost, available * 0.8)   # cap at 80% of available
        weights = p[low_classes] / (available + 1e-15)
        p[low_classes] -= weights * actual_boost
        p[high_classes] += actual_boost / len(high_classes)
    else:
        # Take from M/X and return to QUIET/B/C
        available = p[high_classes].sum()
        actual_dampen = min(abs(boost), available * 0.5)
        hweights = p[high_classes] / (available + 1e-15)
        p[high_classes] -= hweights * actual_dampen
        p[low_classes] += actual_dampen / len(low_classes)

    # Clip negatives and re-normalise
    p = np.clip(p, 0.0, 1.0)
    p /= p.sum() + 1e-15
    return p


class SHARPFusionLayer:
    """
    Applies MagneticContext as a soft prior on ForecastFrames.

    Call apply() with the list of ForecastFrames produced by ForecastOrchestrator
    and the most recent MagneticContext. Returns modified ForecastFrames.

    When ctx is None (no SHARP data), returns frames unchanged.
    """

    def __init__(self, config: Optional[SHARPConfig] = None) -> None:
        self._cfg = config or SHARPConfig()

    def apply(
        self,
        frames: List[ForecastFrame],
        ctx: Optional[MagneticContext],
    ) -> List[ForecastFrame]:
        """
        Apply SHARP-derived magnetic prior to a list of ForecastFrames.

        Parameters
        ----------
        frames : list of ForecastFrame from ForecastOrchestrator (one per horizon)
        ctx    : current MagneticContext, or None if SHARP unavailable

        Returns
        -------
        Modified list of ForecastFrames (same length, same horizons)
        Frames are returned unchanged if ctx is None.
        """
        if ctx is None:
            return frames

        boost = self._compute_boost(ctx)
        if abs(boost) < 1e-9:
            return frames   # nothing to do

        modified = []
        for ff in frames:
            probs = _prob_vector(ff)
            probs_new = _apply_boost(probs, boost)

            # Rebuild the ForecastFrame with updated probabilities
            import dataclasses
            ff_new = dataclasses.replace(
                ff,
                prob_quiet=float(probs_new[0]),
                prob_B=float(probs_new[1]),
                prob_C=float(probs_new[2]),
                prob_M=float(probs_new[3]),
                prob_X=float(probs_new[4]),
            )
            modified.append(ff_new)

        logger.debug(
            "SHARP fusion applied: AR=%s p24h=%.2f p6h=%.2f boost=%.3f",
            ctx.active_region_id,
            ctx.flare_probability_24h or 0.0,
            ctx.flare_probability_6h or 0.0,
            boost,
        )
        return modified

    def _compute_boost(self, ctx: MagneticContext) -> float:
        """Return net probability shift toward M/X classes."""
        p24 = ctx.flare_probability_24h or 0.0
        p6  = ctx.flare_probability_6h or 0.0

        if p24 >= 0.75:
            boost = self._cfg.boost_high
        elif p24 >= 0.50:
            boost = self._cfg.boost_medium
        elif p24 < 0.10:
            boost = self._cfg.dampen
        else:
            boost = 0.0

        # Additional boost for imminent 6h risk
        if p6 >= 0.60:
            boost += self._cfg.boost_medium

        return boost
