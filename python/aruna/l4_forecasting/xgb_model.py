"""
XGBoostForecaster — gradient-boosted ensemble member for L4.

Replaces Moirai as the second ensemble voice alongside DualStreamTCN.
Operates on 26 statistical features extracted from the 600-frame physics
window: mean/std/max/slope/last for each of the 5 key signals, plus
HOPE presence and DNI peak timing.

When unavailable (xgboost not installed or weights missing), falls back
to uniform [0.2 × 5] — same contract as MoiraiForecaster.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_XGB_AVAILABLE = False

try:
    import xgboost as xgb  # noqa: F401
    _XGB_AVAILABLE = True
except ImportError:
    logger.warning(
        "XGBoostForecaster: xgboost not installed — uniform fallback active. "
        "Install with: pip install xgboost"
    )

_UNIFORM_PROBS = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

# xgb_model.py lives at python/aruna/l4_forecasting/xgb_model.py → project root 4 levels up
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_XGB_PATH = str(_PROJECT_ROOT / "models" / "xgb_v1.json")

# Indices in the 19-column feature_bytes matrix (from build_dataset.py)
# Cols 0:8  thermal: T_MK, EM, DNI, tnpsv_mag, tnpsv_ang, dT, dEM, d2T
# Cols 8:16 nonthermal: SHEI, dSHEI, FeKa, hel_10-25, hel_25-50, hel_50-70, QPP_per, QPP_amp
# Cols 16:19 shared: HFD, solexs_3_6, solexs_6_10
_KEY_COLS = {
    "T_MK": 0,
    "EM": 1,
    "DNI": 2,
    "tnpsv_mag": 3,
    "dT": 5,
    "SHEI": 8,
    "HFD": 16,
    "solexs_3_6": 17,
}
_N_BASE_FEATURES = 19
_SEQUENCE_LENGTH = 600


def extract_statistical_features(
    features: np.ndarray,          # (T, 19) float32 from parquet features_bytes
    phases: np.ndarray,            # (T,) uint8  0=QUIET 1=HOPE 2=IMPULSIVE 3=GRADUAL 4=DECAY
) -> np.ndarray:
    """
    Extract 26 statistical features from one sliding window.

    Per signal (8 key cols): mean, std, max, slope, last = 40
    Plus: hope_present (bool), dni_peak_time_ratio (0-1)       =  2
    Bias: constant 1.0                                          =  1
    Total: 43 features
    """
    T = features.shape[0]
    if T == 0:
        return np.zeros(43, dtype=np.float32)

    t_axis = np.linspace(0.0, 1.0, T)

    out: List[float] = []

    for col_name, col_idx in _KEY_COLS.items():
        col = features[:, col_idx].astype(np.float64)
        col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)

        mean_v = float(col.mean())
        std_v  = float(col.std() + 1e-9)
        max_v  = float(col.max())

        # OLS slope via dot product (avoids scipy dependency)
        t_c = t_axis - t_axis.mean()
        slope_v = float(np.dot(t_c, col) / (np.dot(t_c, t_c) + 1e-9))

        last_v = float(col[-1])

        out.extend([mean_v, std_v, max_v, slope_v, last_v])

    # HOPE presence: fraction of timesteps in HOPE phase
    hope_frac = float((phases == 1).sum()) / max(T, 1)
    out.append(hope_frac)

    # DNI peak time ratio: how early/late does DNI peak (0=start, 1=end)
    dni_col = features[:, 2].astype(np.float64)
    dni_col = np.nan_to_num(dni_col, nan=0.0)
    peak_idx = int(np.argmax(dni_col))
    out.append(float(peak_idx) / max(T - 1, 1))

    # Bias term
    out.append(1.0)

    return np.array(out, dtype=np.float32)


class XGBoostForecaster:
    """
    Wraps a fitted XGBClassifier to produce 5-class probability vectors.

    Usage
    -----
    forecaster = XGBoostForecaster()
    forecaster.load("models/xgb_v1.json")   # no-op if file missing
    probs = forecaster.predict(frames)       # List[PhysicsFrame] → (5,) float32
    """

    def __init__(self, model_path: str | None = None) -> None:
        self._path = model_path or _DEFAULT_XGB_PATH
        self._model = None
        self._available = False
        self._present_classes = np.arange(5, dtype=np.int32)  # default: all 5

        if _XGB_AVAILABLE:
            self.load(self._path)

    def load(self, path: str | None = None) -> None:
        target = Path(path or self._path)
        if not target.exists():
            logger.warning(
                "XGBoostForecaster: weights not found at %s — uniform fallback active. "
                "Run: python python/scripts/train_xgb.py",
                target,
            )
            return
        if not _XGB_AVAILABLE:
            return
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(str(target))
            self._model = model

            # Load label mapping (present_classes array) saved by train_xgb.py
            classes_path = target.parent / "xgb_label_classes.npy"
            if classes_path.exists():
                self._present_classes = np.load(str(classes_path)).astype(np.int32)
            else:
                # Fallback: assume all 5 classes present
                self._present_classes = np.arange(5, dtype=np.int32)

            self._available = True
            logger.info("XGBoostForecaster: loaded from %s (classes=%s)",
                        target, self._present_classes.tolist())
        except Exception as exc:
            logger.error("XGBoostForecaster: failed to load %s — %s", target, exc)

    @property
    def available(self) -> bool:
        return self._available and self._model is not None

    def predict_from_array(
        self,
        features: np.ndarray,   # (T, 19)
        phases: np.ndarray,     # (T,) uint8
    ) -> np.ndarray:
        """Returns (5,) float32 probability vector."""
        if not self.available:
            return _UNIFORM_PROBS.copy()
        feat_vec = extract_statistical_features(features, phases)
        proba_enc = self._model.predict_proba(feat_vec.reshape(1, -1))[0]
        # Expand to 5-class using saved label mapping
        proba_5 = np.zeros(5, dtype=np.float32)
        for enc_idx, orig_label in enumerate(self._present_classes):
            proba_5[orig_label] = float(proba_enc[enc_idx])
        return proba_5

    def predict(self, frames: List) -> np.ndarray:
        """
        Predict from a list of PhysicsFrame objects (live inference path).
        Returns (5,) float32 probability vector.
        """
        if not self.available or not frames:
            return _UNIFORM_PROBS.copy()

        T = len(frames)
        feat_matrix = np.zeros((T, _N_BASE_FEATURES), dtype=np.float32)
        phase_arr   = np.zeros(T, dtype=np.uint8)

        _PHASE_MAP = {"QUIET": 0, "HOPE": 1, "IMPULSIVE": 2, "GRADUAL": 3, "DECAY": 4}

        for i, f in enumerate(frames):
            def _g(v):
                if v is None: return 0.0
                try:
                    x = float(v)
                    return 0.0 if (x != x) else x   # NaN check
                except Exception:
                    return 0.0

            # Match build_dataset.py training transform: EM_cm3 stored as log10,
            # dEM_dt as symmetric-log. Without this the model (trained on ~48)
            # sees ~1e48 at inference and collapses to its QUIET prior.
            em_log  = float(np.log10(max(_g(f.EM_cm3), 1e30)))
            dem     = _g(f.dEM_dt)
            dem_log = float(np.sign(dem) * np.log10(1.0 + abs(dem)))

            feat_matrix[i] = [
                _g(f.T_MK), em_log, _g(f.DNI),
                _g(f.tnpsv_magnitude), _g(f.tnpsv_angle_rad),
                _g(f.dT_dt), dem_log, _g(f.d2T_dt2),
                _g(getattr(f, 'tnci', 0.0)),
                _g(f.SHEI), _g(f.dSHEI_dt), _g(f.FeKa_flux),
                _g(f.hel1os_10_25kev), _g(f.hel1os_25_50kev), _g(f.hel1os_50_70kev),
                0.0, 0.0,   # QPP period/amp placeholder
                _g(f.HFD), _g(f.solexs_3_6kev), _g(f.solexs_6_10kev),
            ][:_N_BASE_FEATURES]
            phase_arr[i] = _PHASE_MAP.get(getattr(f, 'flare_phase', 'QUIET'), 0)

        return self.predict_from_array(feat_matrix, phase_arr)
