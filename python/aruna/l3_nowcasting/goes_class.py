from __future__ import annotations
from typing import List, Optional, Tuple

_THRESHOLDS = [
    ("X", 1.0e-4),
    ("M", 1.0e-5),
    ("C", 1.0e-6),
    ("B", 1.0e-7),
]

# B1: Default is a physics-informed placeholder. Must be calibrated against
# matched GOES events using calibrate_from_matched_events() before production use.
# Calibration writes the factor to config.yaml (nowcasting.solexs_to_wm2_factor).
_DEFAULT_FACTOR = 1.0e-8


class GOESClassifier:
    def __init__(self, solexs_to_wm2_factor: float = _DEFAULT_FACTOR) -> None:
        self._factor = solexs_to_wm2_factor

    def classify(self, peak_flux_counts: float) -> Tuple[str, float]:
        wm2 = peak_flux_counts * self._factor
        for letter, threshold in _THRESHOLDS:
            if wm2 >= threshold:
                return letter, round(wm2 / threshold, 1)
        subclass = round(wm2 / 1.0e-8, 1) if wm2 > 0 else 0.0
        return "A", subclass

    def is_major_xclass(self, goes_class: str, subclass: float) -> bool:
        return goes_class == "X" and subclass >= 3.0

    @classmethod
    def calibrate_from_matched_events(
        cls,
        peak_counts_list: List[float],
        goes_wm2_list: List[float],
    ) -> float:
        """B1 fix: derive solexs_to_wm2_factor by least-squares regression
        on matched events (L3 peak counts vs GOES peak flux W/m²).

        Args:
            peak_counts_list: SoLEXS peak counts/s for each matched event
            goes_wm2_list: Corresponding GOES peak flux in W/m²

        Returns:
            Calibrated conversion factor (multiply counts by this to get W/m²)
        """
        import numpy as np
        counts = np.array(peak_counts_list, dtype=np.float64)
        flux = np.array(goes_wm2_list, dtype=np.float64)
        # Log-space regression: log(flux) = log(factor) + log(counts)
        mask = (counts > 0) & (flux > 0)
        if mask.sum() < 2:
            return _DEFAULT_FACTOR
        log_counts = np.log10(counts[mask])
        log_flux = np.log10(flux[mask])
        # slope forced to 1 (linear relationship) — only fit intercept
        log_factor = float(np.mean(log_flux - log_counts))
        factor = 10.0 ** log_factor
        return factor

    @property
    def factor(self) -> float:
        return self._factor
