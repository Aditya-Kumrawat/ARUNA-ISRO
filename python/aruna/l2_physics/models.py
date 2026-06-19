from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, Enum
from typing import List, Optional

import numpy as np

from aruna.models.observation import QualityFlag


class PhysicsQuality(IntEnum):
    FIT_OK           = 0
    FIT_DIVERGED     = 1
    LOW_CONFIDENCE   = 2
    BACKEND_FALLBACK = 3


class EngineState(Enum):
    WARMUP   = "WARMUP"
    ACTIVE   = "ACTIVE"
    DEGRADED = "DEGRADED"


@dataclass
class FitResult:
    T_MK:             float
    EM_cm3:           float
    T_uncertainty_MK: float
    EM_uncertainty:   float
    chi2_reduced:     float
    backend:          str


@dataclass
class HOPEResult:
    hope_flag:      bool
    reason:         str
    lead_time_sec:  int


@dataclass
class PhysicsConfig:
    fit_energy_min_kev:        float = 3.0
    fit_energy_max_kev:        float = 20.0
    fit_chi2_diverged:         float = 5.0
    fit_chi2_low_confidence:   float = 2.0
    hope_dni_threshold:        float = 1.5
    hope_tnpsv_speed_threshold:float = 0.05
    hope_tnpsv_angle_min:      float = np.deg2rad(60)
    hope_tnpsv_angle_max:      float = np.deg2rad(120)
    hope_em_rise_threshold:    float = 1e46
    hope_hel_quiet_multiplier: float = 2.0
    hfd_window:                int   = 600
    qpp_window:                int   = 200
    qpp_min_frames:            int   = 50
    tnpsv_history_len:         int   = 30
    shei_history_len:          int   = 10
    dni_reset_min_non_quiet:   int   = 30
    kafka_topic_physics:       str   = "aruna.physics.frames"
    degraded_window:           int   = 60
    degraded_threshold:        float = 0.5


@dataclass
class PhysicsFrame:
    frame_id:             str
    timestamp:            datetime
    source_frame_ids:     List[str]
    T_MK:                 float
    T_uncertainty_MK:     float
    EM_cm3:               float
    EM_uncertainty:       float
    fit_chi_squared:      float
    DNI:                  float
    tnpsv_magnitude:      float
    tnpsv_angle_rad:      float
    SHEI:                 float
    dSHEI_dt:             float
    HFD:                  float
    QPP_period_sec:       Optional[float]
    QPP_amplitude:        Optional[float]
    FeKa_flux:            float
    flare_phase:          str
    hope_flag:            bool
    hope_lead_time_sec:   Optional[int]
    hope_reason:          Optional[str]
    engine_state:         str
    data_quality_flag:    QualityFlag
    physics_quality_flag: PhysicsQuality
    fit_backend:          str
    pipeline_version:     str
    dT_dt:                float    = 0.0
    dEM_dt:               float    = 0.0
    d2T_dt2:              float    = 0.0
    solexs_3_6kev:        float    = 0.0
    solexs_6_10kev:       float    = 0.0
    hel1os_10_25kev:      float    = 0.0
    hel1os_25_50kev:      float    = 0.0
    hel1os_50_70kev:      float    = 0.0
    suit_uv_flux:         float    = 0.0   # integrated UV brightness [counts/s]; 0 if SUIT absent
    suit_uv_brightening_rate: float = 0.0  # dUV/dt [counts/s per second]
    tnci:                 float    = 0.0   # Thermal-Nonthermal Coupling Index (PRD §13.4)
    provenance_hash:      str      = ""
    quality_flag:         QualityFlag = QualityFlag.GOOD
    schema_version:       str      = "1.0"
    created_at:           datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
