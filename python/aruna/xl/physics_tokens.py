"""
XL-3 — Physics Observable Engine.

Formalises all physics-derived quantities into a named, typed, physically
interpretable feature vector published as "physics tokens" (PRD §13.4).

The 21-token vector is consumed by XL-4 M3 (logistic regression) and
published to Kafka topic `aruna.xl.features`.

New observable: TNCI (Thermal-Nonthermal Coupling Index) — see PRD §13.4.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import numpy as np

from aruna.models.physics import PhysicsFrame

logger = logging.getLogger(__name__)

# ── Token registry (21 tokens matching PRD §13.4) ────────────────────────────
TOKEN_NAMES: List[str] = [
    # Thermal (isothermal fit derivatives)
    "T_MK",              # 0  plasma temperature [MK]
    "EM_cm3",            # 1  emission measure [cm⁻³]
    "dT_dt",             # 2  T gradient [MK/s]
    "dEM_dt",            # 3  EM gradient [cm⁻³/s]
    "d2T_dt2",           # 4  T acceleration [MK/s²]
    "TNCI",              # 5  Thermal-Nonthermal Coupling Index
    # Nonthermal & coupling
    "DNI",               # 6  Dynamic Neupert Integral
    "SHEI",              # 7  Spectral Hardness Evolution Index
    "dSHEI_dt",          # 8  SHEI rate of change
    "FeKa_flux",         # 9  Fe Kα fluorescence [counts/s]
    # Phase space
    "tnpsv_magnitude",   # 10 TNPSV speed [0–1]
    "tnpsv_angle_rad",   # 11 TNPSV heading [rad]
    # Structure & oscillation
    "HFD",               # 12 Higuchi Fractal Dimension
    "QPP_period_sec",    # 13 QPP period [s]; 0 if absent
    "QPP_amplitude",     # 14 QPP amplitude; 0 if absent
    # Raw energy bands
    "solexs_3_6kev",     # 15
    "solexs_6_10kev",    # 16
    "hel1os_10_25kev",   # 17
    "hel1os_25_50kev",   # 18
    "hel1os_50_70kev",   # 19
    # Multi-payload
    "suit_uv_flux",      # 20 SUIT UV brightness [counts/s]; 0 if absent
]

N_TOKENS = len(TOKEN_NAMES)  # 21


@dataclass
class PhysicsTokens:
    """21-token physics feature vector for one 1-second window."""
    timestamp: datetime
    tokens: List[float]             # length N_TOKENS = 21, order = TOKEN_NAMES
    source_frame_id: str
    mpci: float = 0.0               # MPCI score from XL-2 (set by caller)
    flags: List[str] = field(default_factory=list)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        if len(self.tokens) != N_TOKENS:
            raise ValueError(
                f"tokens length must be {N_TOKENS} (got {len(self.tokens)})"
            )

    def as_array(self) -> np.ndarray:
        return np.array(self.tokens, dtype=np.float32)

    def __getitem__(self, name: str) -> float:
        return self.tokens[TOKEN_NAMES.index(name)]


def _safe(v, default: float = 0.0) -> float:
    """Return float v, replacing None and NaN with default."""
    if v is None:
        return default
    f = float(v)
    return default if (f != f) else f   # NaN check


def extract_tokens(frame: PhysicsFrame, mpci: float = 0.0) -> PhysicsTokens:
    """
    Convert one PhysicsFrame into a PhysicsTokens (21-token vector).

    Parameters
    ----------
    frame : L2 PhysicsFrame with all observables populated
    mpci  : MPCI score from XL-2; injected here so M3 can consume it

    Returns
    -------
    PhysicsTokens with tokens ordered by TOKEN_NAMES
    """
    tokens = [
        _safe(frame.T_MK),
        _safe(frame.EM_cm3),
        _safe(frame.dT_dt),
        _safe(frame.dEM_dt),
        _safe(frame.d2T_dt2),
        _safe(getattr(frame, 'tnci', 0.0)),
        _safe(frame.DNI),
        _safe(frame.SHEI),
        _safe(frame.dSHEI_dt),
        _safe(frame.FeKa_flux),
        _safe(frame.tnpsv_magnitude),
        _safe(frame.tnpsv_angle_rad),
        _safe(frame.HFD),
        _safe(frame.QPP_period_sec),
        _safe(frame.QPP_amplitude),
        _safe(frame.solexs_3_6kev),
        _safe(frame.solexs_6_10kev),
        _safe(frame.hel1os_10_25kev),
        _safe(frame.hel1os_25_50kev),
        _safe(frame.hel1os_50_70kev),
        _safe(getattr(frame, 'suit_uv_flux', 0.0)),
    ]

    return PhysicsTokens(
        timestamp=frame.timestamp,
        tokens=tokens,
        source_frame_id=frame.frame_id,
        mpci=mpci,
    )
