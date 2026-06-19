"""
L4 Forecasting Engine — ForecastConfig.
All thresholds sourced from config.yaml forecasting: block.
Never hardcode values in module files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


FORECAST_HORIZONS: Tuple[int, ...] = (5, 10, 15, 30, 60)  # minutes
CLASS_LABELS: Tuple[str, ...] = ("QUIET", "B", "C", "M", "X")

# Feature split sizes — must sum to N_FEATURES
N_THERMAL = 9       # T_MK, EM_cm3, DNI, tnpsv_magnitude, tnpsv_angle_rad, dT_dt, dEM_dt, d2T_dt2, tnci
N_NONTHERMAL = 8    # SHEI, dSHEI_dt, FeKa_flux, hel1os_10_25, hel1os_25_50, hel1os_50_70, QPP_period, QPP_amp
N_SHARED = 10       # HFD, solexs_3_6, solexs_6_10, suit_uv_flux, suit_uv_brightening_rate, + 5 one-hot flare_phase
N_FEATURES = N_THERMAL + N_NONTHERMAL + N_SHARED  # 27


@dataclass
class ForecastConfig:
    """
    All L4 Forecasting Engine thresholds.
    Loaded from config.yaml forecasting: block.
    Never instantiate with hardcoded values in production code.
    """

    # ── Input window ─────────────────────────────────────────────────────────
    sequence_length: int = 600             # 10-minute window at 1-second cadence
    forecast_horizons: Tuple[int, ...] = field(
        default_factory=lambda: FORECAST_HORIZONS
    )
    n_features: int = N_FEATURES           # 27

    # ── Ensemble weights ─────────────────────────────────────────────────────
    tcn_weight: float = 0.5
    moirai_weight: float = 0.5
    moirai_context_length: int = 512       # clip if window exceeds this
    moirai_fallback_tcn_weight: float = 0.7  # TCN weight when Moirai context clipped

    # ── Asymmetric focal loss ─────────────────────────────────────────────────
    alpha: float = 0.2                     # over-prediction penalty (under-pred = 1-alpha = 0.8×)
    gamma: float = 2.0                     # focal loss focusing parameter

    # ── MAPIE conformal calibration ───────────────────────────────────────────
    mapie_coverage: float = 0.90           # 90% conformal coverage guarantee
    mapie_alpha: float = 0.10              # 1 - coverage

    # ── Model paths ───────────────────────────────────────────────────────────
    tcn_weights_path: str = "models/tcn_v1.pt"
    mapie_calibration_path: str = "models/mapie_calibration.pkl"
    feature_scaler_path: str = "models/feature_scaler.pkl"

    # ── Temporal split (STRICT — no random shuffling) ─────────────────────────
    train_start: str = "2023-09-02"
    train_end: str = "2024-06-30"
    val_start: str = "2024-07-01"
    val_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    test_end: str = "2025-06-30"
    buffer_hours: int = 24                 # minimum gap between splits

    # ── Kafka topics ──────────────────────────────────────────────────────────
    kafka_topic_input: str = "aruna.nowcast.events"
    kafka_topic_output: str = "aruna.forecast.frames"

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    pipeline_version: str = "2.0.0"
    model_version: str = "tcn_v1+moirai_base"

    def __post_init__(self) -> None:
        invalid_horizons = [h for h in self.forecast_horizons if h not in FORECAST_HORIZONS]
        if invalid_horizons:
            raise ValueError(
                f"Invalid forecast horizons: {invalid_horizons}. "
                f"Must be subset of {FORECAST_HORIZONS}."
            )
        if not (0.0 < self.alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if not (0.0 < self.mapie_coverage < 1.0):
            raise ValueError(f"mapie_coverage must be in (0, 1), got {self.mapie_coverage}")
        if not (0.0 < self.tcn_weight <= 1.0) or not (0.0 < self.moirai_weight <= 1.0):
            raise ValueError("tcn_weight and moirai_weight must be in (0, 1]")
        if self.n_features != N_FEATURES:
            raise ValueError(
                f"n_features must be {N_FEATURES} (N_THERMAL + N_NONTHERMAL + N_SHARED), "
                f"got {self.n_features}"
            )
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
        # Validate temporal split ordering
        if self.train_end >= self.val_start:
            raise ValueError(
                f"train_end ({self.train_end}) must be before val_start ({self.val_start})"
            )
        if self.val_end >= self.test_start:
            raise ValueError(
                f"val_end ({self.val_end}) must be before test_start ({self.test_start})"
            )
