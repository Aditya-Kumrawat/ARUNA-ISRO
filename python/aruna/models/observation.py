"""
Global data contract: L1 output.
ObservationFrame is the canonical representation of one 1-second spectrum
from a single Aditya-L1 instrument, after dead-time correction and QA.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum

import numpy as np


class QualityFlag(IntEnum):
    GOOD        = 0  # all checks passed
    SAA_PASSAGE = 1  # South Atlantic Anomaly contamination
    DEAD_TIME   = 2  # dead-time correction applied, uncertainty elevated
    SATURATED   = 3  # SDD1 saturated, SDD2 used
    GAP         = 4  # data gap, interpolated
    NOISY       = 5  # background elevated
    MISSING     = 6  # frame absent, no interpolation possible
    CALIBRATION = 7  # in-flight calibration window


# Valid instrument identifiers
INSTRUMENT_IDS = frozenset({
    "SOLEXS_SDD1",
    "SOLEXS_SDD2",
    "HEL1OS_CDTE",
    "HEL1OS_CZT",
    "SUIT_FD",      # SUIT full-disk integrated UV photometry (60s cadence)
    "VELC_DP",      # VELC Doppler-proxy scalar (optional, may be NaN)
})


@dataclass
class ObservationFrame:
    """
    One dead-time-corrected 1-second spectrum from one Aditya-L1 instrument.
    Published to: aruna.observations.solexs | aruna.observations.hel1os
    """

    # ── Identity ────────────────────────────────────────────────────────────
    instrument: str            # one of INSTRUMENT_IDS
    timestamp: datetime        # UTC, nanosecond precision internally

    # ── Spectral data ────────────────────────────────────────────────────────
    energy_bins_kev: np.ndarray        # shape (N+1,) — bin edges in keV
    counts_per_s_kev: np.ndarray       # shape (N,)   — dead-time corrected

    # ── Quality ──────────────────────────────────────────────────────────────
    quality_flag: QualityFlag
    dead_time_fraction: float          # 0.0–1.0
    calibration_version: str           # e.g. 'sarwade_2025_v1'

    # ── Metadata ─────────────────────────────────────────────────────────────
    solar_distance_au: float
    spacecraft_mode: str
    saa_flag: bool
    source_file: str                   # original FITS filename
    checksum_sha256: str
    pipeline_version: str

    # ── GOES label (joined by L1 from SunPy HEK) ────────────────────────────
    goes_class: str | None = None      # 'B'|'C'|'M'|'X'|'QUIET_SUN'|None
    goes_subclass: float | None = None

    # ── Auto-generated ───────────────────────────────────────────────────────
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # ── Message bus metadata ─────────────────────────────────────────────────
    schema_version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.instrument not in INSTRUMENT_IDS:
            raise ValueError(f"Unknown instrument: {self.instrument!r}. Must be one of {INSTRUMENT_IDS}")
        if not (0.0 <= self.dead_time_fraction <= 1.0):
            raise ValueError(f"dead_time_fraction must be in [0, 1], got {self.dead_time_fraction}")
        if len(self.energy_bins_kev) != len(self.counts_per_s_kev) + 1:
            raise ValueError(
                f"energy_bins_kev must have length N+1 where N = len(counts_per_s_kev). "
                f"Got {len(self.energy_bins_kev)} bins for {len(self.counts_per_s_kev)} channels."
            )

    def is_good(self) -> bool:
        return self.quality_flag == QualityFlag.GOOD

    def is_solexs(self) -> bool:
        return self.instrument.startswith("SOLEXS")

    def is_hel1os(self) -> bool:
        return self.instrument.startswith("HEL1OS")

    def is_suit(self) -> bool:
        return self.instrument.startswith("SUIT")

    def is_velc(self) -> bool:
        return self.instrument.startswith("VELC")

    def band_flux(self, low_kev: float, high_kev: float) -> float:
        """Sum counts/s/keV within an energy band [low_kev, high_kev)."""
        centres = (self.energy_bins_kev[:-1] + self.energy_bins_kev[1:]) / 2
        mask = (centres >= low_kev) & (centres < high_kev)
        return float(np.sum(self.counts_per_s_kev[mask]))
