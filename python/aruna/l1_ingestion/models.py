from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class FitsData:
    """Raw parsed data from one FITS detector extension, before correction."""
    instrument: str
    timestamps: np.ndarray       # shape (N_times,), numpy.datetime64[ns] UTC
    counts_raw: np.ndarray       # shape (N_times, N_bins), counts/s
    energy_bins_kev: np.ndarray  # shape (N_bins+1,), bin edges in keV
    livetime: np.ndarray         # shape (N_times,), fraction 0–1
    deadtime_frac: np.ndarray    # shape (N_times,), fraction 0–1
    source_file: str
    checksum_sha256: str


@dataclass
class HKData:
    """Parsed housekeeping data from Aditya_HK FITS."""
    timestamps: np.ndarray    # shape (N_times,), datetime64[ns] UTC
    saa_flags: np.ndarray     # shape (N_times,), bool
    instrument_modes: list    # shape (N_times,), str per element
    solar_dist_au: np.ndarray # shape (N_times,), float


@dataclass
class IngestionConfig:
    """Tunable L1 thresholds — read from config.yaml ingestion section."""
    quarantine_ttl_sec: float = 300.0
    completeness_threshold: float = 0.80
    clock_drift_warn_sec: float = 0.5
    clock_drift_max_sec: float = 1.0
    goes_match_window_min: float = 3.0
    goes_query_window_min: float = 10.0
    background_window_sec: int = 300
    background_poly_order: int = 2
    chi2_dof_limit: float = 2.0
    kafka_topic_solexs: str = "aruna.observations.solexs"
    kafka_topic_hel1os: str = "aruna.observations.hel1os"
    kafka_topic_quality: str = "aruna.quality.summary"
    kafka_topic_errors: str = "aruna.ingestion.errors"


@dataclass
class QuarantineEntry:
    """A frame set held pending a retry after quarantine_ttl_sec."""
    fits_path: Path
    instrument: str
    quarantined_at: datetime
    retry_after: datetime
    reason: str
    attempt_count: int = 0


@dataclass
class ValidationResult:
    all_passed: bool
    checks: dict
    failed_checks: List[str] = field(default_factory=list)
