from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from ..models.observation import QualityFlag
from .models import FitsData

logger = logging.getLogger(__name__)

# Sarwade et al. 2025 Table 2 — hardcoded calibration constants, NOT in config
DEADTIME_CONSTANTS: dict = {
    'SOLEXS_SDD1': 4.5e-6,
    'SOLEXS_SDD2': 4.5e-6,
    'HEL1OS_CDTE': 2.1e-6,
    'HEL1OS_CZT':  1.8e-6,
}

SATURATION_LIMITS: dict = {
    'SOLEXS_SDD1': 5e4,
    'SOLEXS_SDD2': 5e6,
}


class DeadTimeCorrector:
    def correct(self, data: FitsData) -> Tuple[FitsData, np.ndarray]:
        """
        Apply non-paralysable dead-time correction: R_true = R_obs / (1 - R_obs * tau).
        Applied per bin per time step.
        Returns (corrected FitsData, quality_flags shape (N_times,)).
        """
        tau = DEADTIME_CONSTANTS.get(data.instrument)
        if tau is None:
            raise ValueError(f"Unknown instrument: {data.instrument!r}. Must be one of {set(DEADTIME_CONSTANTS)}")

        counts = data.counts_raw.copy()
        n_times = len(counts)
        quality_flags = np.full(n_times, int(QualityFlag.GOOD), dtype=np.int8)

        denom = 1.0 - counts * tau
        safe = denom > 0
        counts[safe] = counts[safe] / denom[safe]
        counts[~safe] = np.nan

        if data.instrument in SATURATION_LIMITS:
            sat_limit = SATURATION_LIMITS[data.instrument]
            total_rate = np.nansum(counts, axis=1)
            saturated = total_rate > sat_limit
            if saturated.any():
                quality_flags[saturated] = int(QualityFlag.SATURATED)
                logger.info(
                    "Saturation detected in %s at %d/%d frames",
                    data.instrument, saturated.sum(), n_times,
                )

        high_dt = data.deadtime_frac > 0.05
        not_sat = quality_flags != int(QualityFlag.SATURATED)
        quality_flags[high_dt & not_sat] = int(QualityFlag.DEAD_TIME)

        corrected = FitsData(
            instrument=data.instrument,
            timestamps=data.timestamps,
            counts_raw=counts,
            energy_bins_kev=data.energy_bins_kev,
            livetime=data.livetime,
            deadtime_frac=data.deadtime_frac,
            source_file=data.source_file,
            checksum_sha256=data.checksum_sha256,
        )
        return corrected, quality_flags

    def select_solexs_detector(
        self,
        sdd1_data: FitsData,
        sdd1_flags: np.ndarray,
        sdd2_data: FitsData,
        sdd2_flags: np.ndarray,
    ) -> Tuple[FitsData, np.ndarray]:
        """Per saturated SDD1 frame, substitute SDD2 counts. Returns merged (data, flags)."""
        sat = sdd1_flags == int(QualityFlag.SATURATED)
        if not sat.any():
            return sdd1_data, sdd1_flags

        merged_counts = sdd1_data.counts_raw.copy()
        merged_flags  = sdd1_flags.copy()
        merged_counts[sat] = sdd2_data.counts_raw[sat]

        merged = FitsData(
            instrument=sdd1_data.instrument,
            timestamps=sdd1_data.timestamps,
            counts_raw=merged_counts,
            energy_bins_kev=sdd1_data.energy_bins_kev,
            livetime=sdd1_data.livetime,
            deadtime_frac=sdd1_data.deadtime_frac,
            source_file=sdd1_data.source_file,
            checksum_sha256=sdd1_data.checksum_sha256,
        )
        return merged, merged_flags
