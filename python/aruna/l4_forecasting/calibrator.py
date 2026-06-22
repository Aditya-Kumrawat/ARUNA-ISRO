"""
MapieCalibrator — MAPIE conformal prediction wrapper for L4 ensemble output.

Provides statistically guaranteed 90% coverage conformal prediction sets.
Graceful degradation: if no calibrator is loaded, passes probabilities through
unchanged and returns a singleton prediction set = [argmax class].
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .models import ForecastConfig, CLASS_LABELS

logger = logging.getLogger(__name__)


def _compute_ece(probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error on a single 5-class probability vector.

    For a single prediction we cannot compute ECE properly (it requires a dataset).
    This returns 0.0 as a placeholder — true ECE is computed during validation
    over the full validation set, not per-prediction. The field in ForecastFrame
    carries the ECE from the last calibration evaluation.
    """
    return 0.0


class MapieCalibrator:
    """
    Thin wrapper around MAPIE conformal classifiers (SplitConformalClassifier in v1.x).

    If MAPIE is not installed or no calibration pickle is loaded,
    the calibrator operates in passthrough mode:
    - calibrated_probs = input probs
    - prediction_set   = [argmax_class]
    - ece              = 0.0

    Usage
    -----
    calibrator = MapieCalibrator(config)
    calibrator.load("models/mapie_calibration.pkl")  # optional
    cal_probs, pred_set, ece = calibrator.calibrate(ensemble_probs)
    """

    def __init__(self, config: ForecastConfig | None = None) -> None:
        self._cfg = config or ForecastConfig()
        self._mapie = None         # fitted MapieClassifier or None
        self._mapie_available = self._check_mapie()
        self._last_ece: float = 0.0

    @staticmethod
    def _check_mapie() -> bool:
        try:
            import mapie  # noqa: F401
            return True
        except ImportError:
            logger.warning(
                "MapieCalibrator: mapie not installed — operating in passthrough mode. "
                "Install with: pip install mapie"
            )
            return False

    def load(self, path: str | None = None) -> None:
        """
        Load calibration data from a pickle file.

        Supports two formats:
        - dict with keys {quantiles, val_probs, val_labels, ece} — custom conformal format
        - fitted MAPIE classifier object (legacy)

        If path is None, uses config.mapie_calibration_path.
        """
        target = Path(path or self._cfg.mapie_calibration_path)
        if not target.exists():
            logger.warning(
                "MapieCalibrator: calibration file not found at %s — "
                "passthrough mode active (prediction_set will contain only argmax class)",
                target,
            )
            return
        try:
            with open(target, "rb") as fh:
                obj = pickle.load(fh)
            if isinstance(obj, dict) and "quantiles" in obj:
                self._mapie = obj
                ece_raw = obj.get("ece", {})
                # Store a scalar ECE for the QUIET class (class 0) as a representative
                self._last_ece = float(self._scalar_ece(ece_raw))
                logger.info(
                    "MapieCalibrator: loaded dict-format calibration from %s  ECE≈%.4f",
                    target, self._last_ece,
                )
            elif hasattr(obj, "predict"):
                self._mapie = obj
                logger.info("MapieCalibrator: loaded MAPIE object from %s", target)
            else:
                logger.error(
                    "MapieCalibrator: unrecognised format in %s (type=%s) — passthrough mode",
                    target, type(obj).__name__,
                )
        except Exception as exc:
            logger.error("MapieCalibrator: failed to load %s: %s — passthrough mode", target, exc)
            self._mapie = None

    @staticmethod
    def _scalar_ece(ece_raw) -> float:
        """Extract a single ECE float from the per-class ECE structure."""
        if isinstance(ece_raw, (int, float)):
            return float(ece_raw)
        if isinstance(ece_raw, dict):
            values = []
            for v in ece_raw.values():
                if isinstance(v, dict) and "accs" in v and "confs" in v:
                    accs   = np.array(v["accs"],   dtype=float)
                    confs  = np.array(v["confs"],  dtype=float)
                    counts = np.array(v.get("counts", np.ones(len(accs))), dtype=float)
                    n = counts.sum()
                    if n > 0:
                        values.append(float((counts / n * np.abs(accs - confs)).sum()))
            return float(np.mean(values)) if values else 0.0
        return 0.0

    @property
    def is_calibrated(self) -> bool:
        """True when a fitted MAPIE calibrator is loaded."""
        return self._mapie is not None

    def calibrate(
        self,
        probs: np.ndarray,             # (5,) ensemble probability vector
    ) -> Tuple[np.ndarray, List[str], float]:
        """
        Apply conformal calibration to an ensemble probability vector.

        Parameters
        ----------
        probs : (5,) float32, must sum ≈ 1.0

        Returns
        -------
        calibrated_probs : (5,) float32 — unchanged in passthrough mode
        prediction_set   : list of class label strings included in the 90% conformal set
        ece              : float — last recorded ECE (0.0 if uncalibrated)
        """
        probs = np.asarray(probs, dtype=np.float32)
        argmax_class = CLASS_LABELS[int(np.argmax(probs))]

        if not self.is_calibrated:
            # Passthrough: return probs, singleton set, ECE = 0
            return probs.copy(), [argmax_class], 0.0

        try:
            if isinstance(self._mapie, dict):
                # Dict-format: LAC conformal prediction using stored quantiles.
                # A class k is included when probs[k] >= 1 - q_alpha (LAC rule).
                alpha   = self._cfg.mapie_alpha          # e.g. 0.10
                q       = self._mapie["quantiles"].get(alpha)
                if q is None:
                    # Nearest available alpha
                    available = self._mapie["quantiles"]
                    q = available[min(available, key=lambda a: abs(a - alpha))]
                threshold = 1.0 - float(q)
                included  = probs >= threshold
                pred_set  = [CLASS_LABELS[i] for i in range(5) if included[i]]
                if not pred_set:
                    pred_set = [argmax_class]
            else:
                # Fitted MAPIE object (legacy path)
                X_input = probs.reshape(1, -1)
                _, prediction_sets = self._mapie.predict(X_input, alpha=self._cfg.mapie_alpha)
                included = prediction_sets[0, :, 0]
                pred_set = [CLASS_LABELS[i] for i in range(5) if included[i]]
                if not pred_set:
                    pred_set = [argmax_class]

            return probs.copy(), pred_set, self._last_ece

        except Exception as exc:
            logger.warning("MapieCalibrator.calibrate() error: %s — passthrough", exc)
            return probs.copy(), [argmax_class], 0.0

    def update_ece(self, ece: float) -> None:
        """Store the latest ECE from a validation run. Called externally during eval."""
        self._last_ece = float(ece)
