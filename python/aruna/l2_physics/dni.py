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

    def update(self, hel_25_50_flux: float, sol_8_15_flux: float, dt: float = 1.0) -> float:
        """
        Update DNI accumulator and return current DNI value.

        Args:
            hel_25_50_flux: HEL1OS hard X-ray flux [25-50 keV]
            sol_8_15_flux: SoLEXS soft X-ray flux [8-15 keV]
            dt: integration timestep (default 1.0 s)

        Returns:
            DNI value: cumulative_hel / sol_8_15_flux
        """
        self._cumulative_hel += hel_25_50_flux * dt
        return self._cumulative_hel / (sol_8_15_flux + 1e-30)

    def reset(self) -> None:
        """Zero the cumulative HEL1OS integral."""
        self._cumulative_hel = 0.0
