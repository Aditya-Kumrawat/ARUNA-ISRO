"""
DNI (Dynamic Neupert Integral) Computer.
Accumulates HEL1OS 25-50 keV flux over time, normalizes by SoLEXS 8-15 keV flux.
"""

from __future__ import annotations


class DNIComputer:
    """
    DNI = ∫F_HEL1OS(25-50 keV) dt / F_SoLEXS(8-15 keV).
    Used as an impulsive-onset indicator. HOPE threshold: DNI > 1.5.
    """

    def __init__(self) -> None:
        self._cumulative_hel: float = 0.0

    def update(self, hel_25_50_excess: float, sol_3_10_flux: float, dt: float = 1.0) -> float:
        """
        Update DNI accumulator and return current DNI value.

        Args:
            hel_25_50_excess: HEL1OS 25-50 keV flux above quiet baseline
            sol_3_10_flux:    SoLEXS 3-10 keV flux (always non-zero at quiet sun)
            dt:               integration timestep (default 1.0 s)

        Returns:
            DNI value: cumulative_hel_excess / sol_3_10_flux, capped at 1000
        """
        self._cumulative_hel += max(hel_25_50_excess, 0.0) * dt
        return min(self._cumulative_hel / max(sol_3_10_flux, 1.0), 1000.0)

    def reset(self) -> None:
        """Zero the cumulative HEL1OS integral."""
        self._cumulative_hel = 0.0
