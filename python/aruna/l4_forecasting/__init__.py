"""
aruna.l4_forecasting — L4 Forecasting Engine public API.

Entry point: ForecastOrchestrator.process(frame, nowcast_event)
Output:      list[ForecastFrame] | None
"""
from .calibrator import MapieCalibrator
from .ensemble import EnsembleWeighter
from .features import FeatureExtractor
from .forecaster import ForecastOrchestrator
from .models import ForecastConfig, CLASS_LABELS, FORECAST_HORIZONS
from .moirai_model import MoiraiForecaster

# Conditional export — tcn_model requires torch
try:
    from .tcn_model import DualStreamTCN, TemporalBlock, load_tcn, asymmetric_focal_loss
    _TCN_AVAILABLE = True
except ImportError:
    _TCN_AVAILABLE = False

__all__ = [
    "ForecastConfig",
    "ForecastOrchestrator",
    "FeatureExtractor",
    "EnsembleWeighter",
    "MapieCalibrator",
    "MoiraiForecaster",
    "CLASS_LABELS",
    "FORECAST_HORIZONS",
]

if _TCN_AVAILABLE:
    __all__ += ["DualStreamTCN", "TemporalBlock", "load_tcn", "asymmetric_focal_loss"]
