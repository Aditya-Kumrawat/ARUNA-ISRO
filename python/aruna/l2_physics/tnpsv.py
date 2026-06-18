"""
TNPSV (T-EM Normalised Phase Space Velocity) Computer.
Tracks T and EM history, computes normalized velocity in T-EM phase space.
"""

from __future__ import annotations

from collections import deque
from typing import Tuple

import numpy as np


class TNPSVComputer:
    """
    TNPSV = |(dT/dt / T_max, dEM/dt / EM_max)| in T-EM phase space.
    Returns magnitude and angle (for precursor detection).
    HOPE precursor quadrant: 60°–120° (T rising, EM flat).
    """

    def __init__(self, maxlen: int = 30) -> None:
        """
        Initialize with rolling history buffers.

        Args:
            maxlen: Maximum number of frames to retain (default 30)
        """
        self._T = deque(maxlen=maxlen)
        self._EM = deque(maxlen=maxlen)

    def update(self, T_MK: float, EM_cm3: float) -> Tuple[float, float]:
        """
        Update history and return TNPSV magnitude and angle.

        Args:
            T_MK: Temperature in million Kelvin
            EM_cm3: Emission measure in cm^-3

        Returns:
            (magnitude, angle_rad) or (0.0, 0.0) if insufficient history (< 3 frames)
        """
        self._T.append(T_MK)
        self._EM.append(EM_cm3)

        if len(self._T) < 3:
            return 0.0, 0.0

        T = np.array(self._T)
        EM = np.array(self._EM)

        T_max = max(float(np.max(T)), 1e-30)
        EM_max = max(float(np.max(EM)), 1e-30)

        dT = np.gradient(T)
        dEM = np.gradient(EM)

        # Normalized velocity magnitude
        mag = float(np.sqrt((dT[-1] / T_max) ** 2 + (dEM[-1] / EM_max) ** 2))

        # Angle in T-EM phase space (atan2(dEM, dT))
        ang = float(np.arctan2(dEM[-1], dT[-1]))

        return mag, ang
