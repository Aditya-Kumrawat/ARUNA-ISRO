"""
EnsembleWeighter — combines TCN and Moirai probability vectors into a
calibration-ready ensemble probability. Pure function module; no state.

Key behaviour:
- Default weights: tcn=0.5, moirai=0.5
- Moirai context clipped (>512): tcn_weight → 0.7, moirai_weight → 0.3
- Output always sums to 1.0 (float safety normalisation applied)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .models import ForecastConfig


class EnsembleWeighter:
    """
    Weighted combination of TCN and Moirai probability vectors.

    Parameters
    ----------
    config : ForecastConfig — provides tcn_weight, moirai_weight,
             moirai_fallback_tcn_weight from config.yaml.
    """

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self._cfg = config or ForecastConfig()

    def combine(
        self,
        tcn_probs: np.ndarray,           # shape (5,) — softmax of TCN logits
        moirai_probs: np.ndarray,         # shape (5,) — uniform if unavailable
        context_was_clipped: bool = False,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Combine TCN and Moirai probability vectors.

        Parameters
        ----------
        tcn_probs          : (5,) float32 — TCN class probabilities (must sum ≈ 1)
        moirai_probs       : (5,) float32 — Moirai class probabilities (must sum ≈ 1)
        context_was_clipped: bool — if True, shift weight toward TCN per PRD §8.6

        Returns
        -------
        ensemble_probs      : (5,) float32, sums to 1.0
        effective_tcn_weight: float — actual tcn weight used
        effective_moirai_weight: float — actual moirai weight used
        """
        if context_was_clipped:
            w_tcn = self._cfg.moirai_fallback_tcn_weight      # 0.7
            w_moirai = 1.0 - w_tcn                             # 0.3
        else:
            w_tcn = self._cfg.tcn_weight                       # 0.5
            w_moirai = self._cfg.moirai_weight                 # 0.5

        ensemble = w_tcn * np.asarray(tcn_probs, dtype=np.float32) \
                 + w_moirai * np.asarray(moirai_probs, dtype=np.float32)

        # Normalise — guards against floating-point drift
        total = ensemble.sum()
        if total > 0:
            ensemble = ensemble / total
        else:
            ensemble = np.full(5, 0.2, dtype=np.float32)

        return ensemble, float(w_tcn), float(w_moirai)
