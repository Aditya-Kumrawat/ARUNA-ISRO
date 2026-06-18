"""FeKα (Iron K-alpha 6.4 keV) fluorescence line extractor.

Subtracts CHIANTI continuum model from observed spectrum and integrates
the residual 6.2–6.6 keV band to quantify Iron fluorescence.
"""
from __future__ import annotations
import numpy as np
from .chianti_table import CHIANTILookupTable


class FeKaExtractor:
    """Extract Iron fluorescence flux from X-ray spectrum.

    The FeKα line (6.4 keV) is a signature of collisional plasma ionization
    in solar flares. This extractor removes the bremsstrahlung continuum
    using the CHIANTI isothermal model (including line emission contributions)
    and sums the residual in the 6.2–6.6 keV band.

    Continuum is computed via CHIANTILookupTable.lookup() as required by the
    spec, so that line emission contributions beyond pure bremsstrahlung are
    properly accounted for.
    """

    def __init__(self) -> None:
        self._table = CHIANTILookupTable()

    def extract(
        self,
        counts_per_s_kev: np.ndarray,
        energy_bins_kev: np.ndarray,
        T_MK: float,
        EM_cm3: float,
    ) -> float:
        """Extract Fe Kα fluorescence flux by subtracting CHIANTI continuum.

        Returns 0.0 if T_MK or EM_cm3 is NaN or non-positive.
        Fluorescence is physically non-negative; result is clipped to [0, ∞).
        """
        if np.isnan(T_MK) or np.isnan(EM_cm3) or T_MK <= 0 or EM_cm3 <= 0:
            return 0.0
        continuum = self._table.lookup(T_MK, EM_cm3, energy_bins_kev)
        residual = counts_per_s_kev - continuum
        centres = (energy_bins_kev[:-1] + energy_bins_kev[1:]) / 2.0
        mask = (centres >= 6.2) & (centres <= 6.6)
        if not mask.any():
            return 0.0
        return float(np.sum(np.clip(residual[mask], 0.0, None)))
