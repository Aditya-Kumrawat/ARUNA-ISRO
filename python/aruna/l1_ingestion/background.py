from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.optimize import curve_fit

from ..models.observation import QualityFlag
from .models import FitsData

logger = logging.getLogger(__name__)


def _poly2(x, a, b, c):
    return a * x ** 2 + b * x + c


class BackgroundSubtractor:
    def __init__(self, poly_order: int = 2, chi2_dof_limit: float = 2.0):
        self._poly_order = poly_order
        self._chi2_limit = chi2_dof_limit

    def subtract(
        self,
        data: FitsData,
        background_frames: List[FitsData],
        saa_flags: np.ndarray,
    ) -> Tuple[FitsData, np.ndarray, float]:
        """
        Fit polynomial background to quiet window, subtract per energy bin, validate χ²/dof.
        SAA bins are flagged QualityFlag.SAA_PASSAGE but NOT dropped.
        """
        n_cur = len(data.counts_raw)
        quality_flags = np.full(n_cur, int(QualityFlag.GOOD), dtype=np.int8)
        quality_flags[saa_flags] = int(QualityFlag.SAA_PASSAGE)

        if not background_frames:
            logger.warning("No background frames for %s — skipping subtraction", data.instrument)
            return data, quality_flags, 0.0

        bg_counts = np.vstack([f.counts_raw for f in background_frames])
        n_bg, n_bins = bg_counts.shape
        x_bg = np.arange(n_bg, dtype=np.float64)
        x_cur = np.arange(n_bg, n_bg + n_cur, dtype=np.float64)

        bg_model = np.zeros_like(bg_counts)
        bg_current = np.zeros((n_cur, n_bins))

        for b in range(n_bins):
            y = bg_counts[:, b]
            try:
                popt, _ = curve_fit(_poly2, x_bg, y, maxfev=2000,
                                    p0=[0.0, 0.0, float(np.median(y))])
                bg_model[:, b] = _poly2(x_bg, *popt)
                bg_current[:, b] = _poly2(x_cur, *popt)
            except RuntimeError:
                med = float(np.median(y))
                bg_model[:, b] = med
                bg_current[:, b] = med

        subtracted = np.maximum(data.counts_raw - bg_current, 0.0)

        poisson_var = np.maximum(bg_model, 1.0)
        residuals = bg_counts - bg_model
        chi2_total = float(np.sum(residuals ** 2 / poisson_var))
        dof = max(n_bg * n_bins - (self._poly_order + 1) * n_bins, 1)
        chi2_dof = chi2_total / dof

        if chi2_dof > self._chi2_limit:
            logger.warning(
                "Background χ²/dof=%.3f > %.1f for %s — flagging NOISY",
                chi2_dof, self._chi2_limit, data.instrument,
            )
            noisy = quality_flags == int(QualityFlag.GOOD)
            quality_flags[noisy] = int(QualityFlag.NOISY)

        result = FitsData(
            instrument=data.instrument,
            timestamps=data.timestamps,
            counts_raw=subtracted,
            energy_bins_kev=data.energy_bins_kev,
            livetime=data.livetime,
            deadtime_frac=data.deadtime_frac,
            source_file=data.source_file,
            checksum_sha256=data.checksum_sha256,
        )
        return result, quality_flags, chi2_dof
