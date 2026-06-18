"""
SHEI (Spectral Hardness Energy Index) Computer.
Tracks hard (40-70 keV) and soft (10-25 keV) X-ray flux ratio over time.
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np


class SHEIComputer:
    """
    SHEI = log10[F_HEL1OS(40-70 keV) / F_HEL1OS(10-25 keV)].
    Transitions from thermal (< -1.5) to nonthermal (> -0.8) during impulsive phase.
    """

    def __init__(self, maxlen: int = 10) -> None:
        """
        Initialize with rolling SHEI history.

        Args:
            maxlen: Maximum number of SHEI values to retain (default 10)
        """
        self._history: deque = deque(maxlen=maxlen)

    def update_single(self, hel_frame) -> Tuple[float, float]:
        """
        Convenience wrapper: extract spectral data from ObservationFrame and update.

        Args:
            hel_frame: ObservationFrame from HEL1OS instrument

        Returns:
            (SHEI, dSHEI_dt)
        """
        return self.update(hel_frame.counts_per_s_kev, hel_frame.energy_bins_kev)

    def update(self, counts: np.ndarray, energy_bins: np.ndarray) -> Tuple[float, float]:
        """
        Compute SHEI from spectral counts and update history.

        Args:
            counts: Spectral counts array (shape N,)
            energy_bins: Energy bin edges (shape N+1,)

        Returns:
            (SHEI, dSHEI_dt): Current SHEI and its time derivative
        """
        centres = (energy_bins[:-1] + energy_bins[1:]) / 2.0

        # Hard (40-70 keV) and soft (10-25 keV) band fluxes
        F_hard = float(np.sum(counts[(centres >= 40) & (centres <= 70)]))
        F_soft = float(np.sum(counts[(centres >= 10) & (centres <= 25)]))

        # SHEI = log10(hard / soft)
        shei = float(np.log10(F_hard / (F_soft + 1e-30)))

        self._history.append(shei)

        # Time derivative: use numpy.gradient on history
        dshei_dt = float(np.gradient(np.array(self._history))[-1]) if len(self._history) >= 2 else 0.0

        return shei, dshei_dt
