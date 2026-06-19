"""
Global data contract: L2 output.
PhysicsFrame carries all plasma diagnostics and novel observables
computed from one 1-second window of SoLEXS + HEL1OS spectra.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from .observation import QualityFlag


FLARE_PHASES = frozenset({"QUIET", "HOPE", "IMPULSIVE", "GRADUAL", "DECAY"})


@dataclass
class PhysicsFrame:
    """
    Physics diagnostics for one 1-second observation window.
    Published to: aruna.physics.frames
    """

    timestamp: datetime

    # ── Source provenance ────────────────────────────────────────────────────
    source_frame_ids: List[str]        # L1 ObservationFrame.frame_id list used

    # ── Plasma diagnostics (isothermal fit, sunkit-spex f_vth) ───────────────
    T_MK: float                        # plasma temperature [megakelvin]
    T_uncertainty_MK: float            # 1-sigma
    EM_cm3: float                      # emission measure [cm^-3]
    EM_uncertainty: float              # 1-sigma
    fit_chi_squared: float             # goodness of isothermal fit (target < 2.0)

    # ── Novel physics observables ────────────────────────────────────────────
    DNI: float                         # Dynamic Neupert Integral (threshold 1.5)
    tnpsv_magnitude: float             # phase-space speed [normalised 0–1]
    tnpsv_angle_rad: float             # phase-space heading [radians]
    SHEI: float                        # Spectral Hardness Evolution Index
    dSHEI_dt: float                    # hardening rate [per second]
    HFD: float                         # Higuchi Fractal Dimension (10-min window)
    FeKa_flux: float                   # Fe Kα fluorescence residual [counts/s]

    # ── QPP (quasi-periodic pulsations) — None when not detected ─────────────
    QPP_period_sec: Optional[float]
    QPP_amplitude: Optional[float]

    # ── Derived rates ────────────────────────────────────────────────────────
    dT_dt: float                       # temperature gradient [MK/s]
    dEM_dt: float                      # emission measure gradient [cm^-3/s]
    d2T_dt2: float                     # temperature acceleration [MK/s²]

    # ── Raw flux bands (for L4 feature vector) ───────────────────────────────
    solexs_3_6kev: float
    solexs_6_10kev: float
    hel1os_10_25kev: float
    hel1os_25_50kev: float
    hel1os_50_70kev: float

    # ── Flare state ──────────────────────────────────────────────────────────
    flare_phase: str                   # one of FLARE_PHASES
    hope_flag: bool
    hope_lead_time_sec: Optional[int]  # estimated seconds before peak
    hope_reason: Optional[str]         # human-readable HOPE trigger explanation

    # ── Quality ──────────────────────────────────────────────────────────────
    quality_flag: QualityFlag

    # ── Multi-payload (SUIT UV, XL-2 MPCI inputs) — 0.0 if SUIT not available ─
    suit_uv_flux: float = 0.0               # integrated UV brightness [counts/s]
    suit_uv_brightening_rate: float = 0.0   # dUV/dt [counts/s per second]
    tnci: float = 0.0                       # Thermal-Nonthermal Coupling Index (PRD §13.4)

    # ── Auto-generated ───────────────────────────────────────────────────────
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provenance_hash: str = ""          # SHA-256 of input frame checksums

    # ── Message bus metadata ─────────────────────────────────────────────────
    schema_version: str = "1.0"
    pipeline_version: str = "2.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.flare_phase not in FLARE_PHASES:
            raise ValueError(f"flare_phase must be one of {FLARE_PHASES}, got {self.flare_phase!r}")
        if self.fit_chi_squared < 0:
            raise ValueError("fit_chi_squared cannot be negative")

    def is_hope_active(self) -> bool:
        return self.hope_flag and self.flare_phase == "HOPE"

    def is_impulsive(self) -> bool:
        return self.flare_phase == "IMPULSIVE"

    def is_quiet(self) -> bool:
        return self.flare_phase == "QUIET"
