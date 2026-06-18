from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from .models import FitsData

logger = logging.getLogger(__name__)

_XCORR_BAND_KEV = (10.0, 15.0)


def _band_sum(data: FitsData) -> np.ndarray:
    centres = (data.energy_bins_kev[:-1] + data.energy_bins_kev[1:]) / 2.0
    mask = (centres >= _XCORR_BAND_KEV[0]) & (centres < _XCORR_BAND_KEV[1])
    if not mask.any():
        return np.zeros(len(data.counts_raw))
    return data.counts_raw[:, mask].sum(axis=1)


def _trim(data: FitsData, n: int) -> FitsData:
    return FitsData(
        instrument=data.instrument,
        timestamps=data.timestamps[:n],
        counts_raw=data.counts_raw[:n],
        energy_bins_kev=data.energy_bins_kev,
        livetime=data.livetime[:n],
        deadtime_frac=data.deadtime_frac[:n],
        source_file=data.source_file,
        checksum_sha256=data.checksum_sha256,
    )


def _interpolate_to_grid(data: FitsData, target_times: np.ndarray) -> FitsData:
    src_t = data.timestamps.astype(np.float64)
    tgt_t = target_times.astype(np.float64)
    n_bins = data.counts_raw.shape[1]
    new_counts = np.zeros((len(tgt_t), n_bins))
    for b in range(n_bins):
        f = interp1d(src_t, data.counts_raw[:, b], kind='cubic',
                     bounds_error=False, fill_value=0.0)
        new_counts[:, b] = f(tgt_t)
    return FitsData(
        instrument=data.instrument,
        timestamps=target_times,
        counts_raw=new_counts,
        energy_bins_kev=data.energy_bins_kev,
        livetime=np.interp(tgt_t, src_t, data.livetime),
        deadtime_frac=np.interp(tgt_t, src_t, data.deadtime_frac),
        source_file=data.source_file,
        checksum_sha256=data.checksum_sha256,
    )


class TimeAligner:
    def __init__(self, drift_warn_sec: float = 0.5, drift_max_sec: float = 1.0):
        self._drift_warn = drift_warn_sec
        self._drift_max = drift_max_sec

    def compute_lag(self, solexs: FitsData, hel1os: FitsData) -> float:
        """Cross-correlate 10-15 keV band. Returns lag in seconds (1 sample = 1s)."""
        s = _band_sum(solexs)
        h = _band_sum(hel1os)
        n = min(len(s), len(h))
        s, h = s[:n], h[:n]
        corr = signal.correlate(s - s.mean(), h - h.mean(), mode='full')
        lags = signal.correlation_lags(n, n, mode='full')
        return float(lags[np.argmax(corr)])

    def align(
        self, solexs: FitsData, hel1os: FitsData,
    ) -> Tuple[FitsData, FitsData, float]:
        """
        Align SoLEXS and HEL1OS to common 1s UTC grid.
        Returns (aligned_solexs, aligned_hel1os, lag_sec).
        """
        lag_sec = self.compute_lag(solexs, hel1os)

        if abs(lag_sec) > self._drift_warn:
            logger.warning(
                "CLOCK_DRIFT detected: lag=%.3fs between SoLEXS and HEL1OS",
                lag_sec,
            )

        if abs(lag_sec) <= self._drift_max:
            n = min(len(solexs.timestamps), len(hel1os.timestamps))
            return _trim(solexs, n), _trim(hel1os, n), lag_sec

        logger.warning("CLOCK_DRIFT > %.1fs (%.3fs) — cubic interpolation applied", self._drift_max, lag_sec)
        hel1os_aligned = _interpolate_to_grid(hel1os, solexs.timestamps)
        n = min(len(solexs.timestamps), len(hel1os_aligned.timestamps))
        return _trim(solexs, n), _trim(hel1os_aligned, n), lag_sec
