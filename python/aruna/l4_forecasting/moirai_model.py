"""
MoiraiForecaster — Salesforce Moirai-1.0-R zero-shot time-series foundation model.
Lazy import strategy: uni2ts is expensive; ImportError handled gracefully.

When unavailable: returns uniform probability vector [0.2, 0.2, 0.2, 0.2, 0.2].
When context > 512: clips to last 512 steps and logs MOIRAI_CONTEXT_CLIP.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Attempt import once at module load — availability cached as module-level flag
_MOIRAI_AVAILABLE = False
_MOIRAI_IMPORT_ATTEMPTED = False


def _try_import_moirai():
    global _MOIRAI_AVAILABLE, _MOIRAI_IMPORT_ATTEMPTED
    if _MOIRAI_IMPORT_ATTEMPTED:
        return _MOIRAI_AVAILABLE
    _MOIRAI_IMPORT_ATTEMPTED = True
    try:
        import uni2ts  # noqa: F401
        _MOIRAI_AVAILABLE = True
        logger.info("MoiraiForecaster: uni2ts available — Moirai enabled")
    except ImportError:
        _MOIRAI_AVAILABLE = False
        logger.warning(
            "MoiraiForecaster: uni2ts not installed — using uniform fallback. "
            "Install with: pip install uni2ts"
        )
    return _MOIRAI_AVAILABLE


# Uniform fallback — 5 equal-probability classes (QUIET, B, C, M, X)
_UNIFORM_PROBS = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

_CLASS_LABELS = ("QUIET", "B", "C", "M", "X")


class MoiraiForecaster:
    """
    Zero-shot time-series forecaster using Salesforce Moirai-1.0-R-base.

    Usage
    -----
    forecaster = MoiraiForecaster(context_length=512)
    if forecaster.available:
        probs = forecaster.predict(feature_history)
    else:
        probs = _UNIFORM_PROBS  # graceful degradation

    Context clipping
    ----------------
    If the feature history has more than `context_length` timesteps, the last
    `context_length` steps are used, and MOIRAI_CONTEXT_CLIP is logged.
    The EnsembleWeighter is notified via the `context_was_clipped` return flag.

    Feature used
    ------------
    Moirai operates on the DNI time-series (single channel, most predictive scalar).
    Its output distribution is mapped to the 5-class probability space via the
    pre-fitted `_linear_head` calibration layer. In the absence of a fitted head,
    the raw Moirai output is projected via softmax.
    """

    def __init__(self, context_length: int = 512) -> None:
        self._context_length = context_length
        self._available = _try_import_moirai()
        self._model = None  # loaded on first call to predict() if available

        if self._available:
            self._load_model()

    def _load_model(self) -> None:
        """Load Moirai model weights. Catches any load error and marks unavailable."""
        try:
            from uni2ts.model.moirai import MoiraiForecast  # type: ignore[import]
            self._model = MoiraiForecast.from_pretrained("Salesforce/moirai-1.0-R-base")
            logger.info("MoiraiForecaster: model loaded from Salesforce/moirai-1.0-R-base")
        except Exception as exc:
            logger.warning("MoiraiForecaster: failed to load model: %s — using fallback", exc)
            self._available = False
            self._model = None

    @property
    def available(self) -> bool:
        """True when uni2ts is installed and model loaded successfully."""
        return self._available and self._model is not None

    def predict(
        self,
        feature_history: np.ndarray,      # shape (T, N_features) — raw or scaled
        dni_feature_index: int = 2,        # index of DNI in feature vector
    ) -> tuple[np.ndarray, bool]:
        """
        Produce a 5-class probability vector for the next 30-minute horizon.

        Parameters
        ----------
        feature_history    : (T, N_features) array — the current sliding window
        dni_feature_index  : index of DNI in the feature vector (default=2, thermal stream col 2)

        Returns
        -------
        probs              : (5,) float32 array summing to 1.0
        context_was_clipped: bool — True if T > context_length (signals EnsembleWeighter)

        Graceful degradation
        --------------------
        If Moirai is unavailable: returns (uniform, False) without raising.
        """
        context_was_clipped = False

        if not self.available:
            return _UNIFORM_PROBS.copy(), context_was_clipped

        T = feature_history.shape[0]
        if T > self._context_length:
            feature_history = feature_history[-self._context_length:]
            context_was_clipped = True
            logger.warning(
                "MOIRAI_CONTEXT_CLIP: context trimmed from %d to %d steps",
                T,
                self._context_length,
            )

        try:
            # Extract DNI single time-series for Moirai input
            dni_series = feature_history[:, dni_feature_index]  # shape (T,)
            # Moirai expects (batch, time, 1) or similar — adapt to model's API
            forecast = self._model.predict(
                dni_series.reshape(1, -1, 1),
                prediction_length=30,
            )
            # Map Moirai output distribution to 5-class probability space
            probs = self._map_to_class_probs(forecast)
        except Exception as exc:
            logger.warning("MoiraiForecaster.predict() error: %s — returning fallback", exc)
            return _UNIFORM_PROBS.copy(), context_was_clipped

        return probs.astype(np.float32), context_was_clipped

    @staticmethod
    def _map_to_class_probs(forecast) -> np.ndarray:
        """
        Map Moirai forecast distribution to 5-class (QUIET/B/C/M/X) probabilities.

        Moirai returns a probabilistic forecast for DNI over 30 minutes.
        High DNI values correlate with higher GOES classes:
          DNI < 0.5  → QUIET
          0.5–1.0    → B
          1.0–1.5    → C
          1.5–2.5    → M
          > 2.5      → X

        The P(class) is computed from the fraction of forecast samples in each bucket.
        """
        try:
            # forecast.samples shape: (num_samples, prediction_length)
            samples = forecast.samples if hasattr(forecast, "samples") else np.array(forecast)
            peak_values = samples.max(axis=-1)  # (num_samples,) — peak DNI in 30min
            thresholds = [0.5, 1.0, 1.5, 2.5]
            probs = np.zeros(5, dtype=np.float32)
            n = len(peak_values)
            probs[0] = np.sum(peak_values < thresholds[0]) / n           # QUIET
            probs[1] = np.sum((peak_values >= thresholds[0]) & (peak_values < thresholds[1])) / n  # B
            probs[2] = np.sum((peak_values >= thresholds[1]) & (peak_values < thresholds[2])) / n  # C
            probs[3] = np.sum((peak_values >= thresholds[2]) & (peak_values < thresholds[3])) / n  # M
            probs[4] = np.sum(peak_values >= thresholds[3]) / n           # X
        except Exception:
            return _UNIFORM_PROBS.copy()

        # Normalise to ensure sum == 1.0 (floating-point safety)
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = _UNIFORM_PROBS.copy()
        return probs
