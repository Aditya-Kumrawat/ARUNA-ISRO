"""
Higuchi Fractal Dimension (HFD) Computer for time-series analysis.

HFD is a measure of signal complexity. Lower values indicate smoother signals
(e.g., sine waves: HFD ~ 1.0), higher values indicate rougher/noisier signals
(e.g., white noise: HFD ~ 1.5-1.9).
"""
from __future__ import annotations
from collections import deque
import numpy as np


class HFDComputer:
    """
    Compute Higuchi Fractal Dimension on a sliding window of signal values.

    Returns NaN during warm-up until the buffer reaches the target window size.
    After that, returns a float in [1.0, 2.0] at each update.
    """

    def __init__(self, window: int = 600, k_max: int = 10) -> None:
        """
        Args:
            window: Maximum buffer length (returns NaN until filled).
            k_max: Maximum lag factor for HFD calculation.
        """
        self._window = window
        self._k_max  = k_max
        self._buf: deque = deque(maxlen=window)
        self._sample_count = 0

    def update(self, total_flux: float) -> float:
        """
        Add one sample and compute HFD if buffer is full.

        Args:
            total_flux: Next signal sample (e.g., total X-ray flux in counts/s).

        Returns:
            HFD value (float, range ~ [1.0, 2.0]) if buffer is full, else NaN.
        """
        self._buf.append(total_flux)
        self._sample_count += 1
        if self._sample_count <= self._window:
            return float("nan")
        return self._hfd(np.array(self._buf))

    def _hfd(self, x: np.ndarray) -> float:
        """
        Compute Higuchi Fractal Dimension.

        References:
            Higuchi, T. (1988). Physica D, 31(2), 277-283.
        """
        L = []
        for k in range(1, self._k_max + 1):
            Lk = []
            for m in range(1, k + 1):
                idx = np.arange(m - 1, len(x), k)
                if len(idx) < 2:
                    continue
                Lmk = np.sum(np.abs(np.diff(x[idx]))) * (len(x) - 1) / (len(idx) * k ** 2)
                Lk.append(Lmk)
            if Lk:
                L.append(float(np.mean(Lk)))

        if len(L) < 2:
            return float("nan")

        k_vals = np.arange(1, len(L) + 1, dtype=float)
        try:
            slope, _ = np.polyfit(np.log(k_vals), np.log(np.array(L) + 1e-30), 1)
            hfd = float(-slope)
            # Clamp to theoretical bounds [1.0, 2.0]
            return np.clip(hfd, 1.0, 2.0)
        except Exception:
            return float("nan")
