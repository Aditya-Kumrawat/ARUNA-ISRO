from __future__ import annotations
from typing import Tuple

_THRESHOLDS = [
    ("X", 1.0e-4),
    ("M", 1.0e-5),
    ("C", 1.0e-6),
    ("B", 1.0e-7),
]


class GOESClassifier:
    def __init__(self, solexs_to_wm2_factor: float = 1.0e-8) -> None:
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
