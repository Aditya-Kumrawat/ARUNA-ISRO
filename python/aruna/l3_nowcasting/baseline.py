from __future__ import annotations
from collections import deque
from typing import Optional, Tuple
import numpy as np


class RollingP85Baseline:
    def __init__(
        self,
        window_sec: int = 10800,
        percentile: float = 85.0,
        recompute_interval: int = 60,
        sigma_multiplier: float = 3.0,
    ) -> None:
        self._window = deque(maxlen=window_sec)
        self._window_sec = window_sec
        self._percentile = percentile
        self._recompute_interval = recompute_interval
        self._sigma_multiplier = sigma_multiplier
        self._baseline: Optional[float] = None
        self._threshold: Optional[float] = None
        self._last_recompute: int = 0
        self._frozen: bool = False
        # Scale warm-up requirement with window size so small test windows work.
        # Production (window=10800) → min_frames=300 (~5 min). Tests can be smaller.
        self._min_frames: int = min(300, max(10, window_sec // 4))

    def update(self, flux: float, t_sec: int) -> Tuple[Optional[float], Optional[float]]:
        self._window.append(flux)
        if not self._frozen and (t_sec - self._last_recompute) >= self._recompute_interval:
            self._recompute(t_sec)
        return self._baseline, self._threshold

    def _recompute(self, t_sec: int) -> None:
        data = list(self._window)
        if len(data) < self._min_frames:
            return
        arr = np.array(data, dtype=np.float64)
        # Robust outlier clipping: MAD-based (handles up to ~40% outliers vs std which breaks at 15%)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        # Convert MAD to Gaussian sigma; use 10% of median as floor to handle near-zero data
        sigma_mad = mad / 0.6745
        clip_hi = med + 5.0 * max(sigma_mad, med * 0.10)
        arr_r = arr[arr <= clip_hi]
        if len(arr_r) < len(arr) // 2:
            arr_r = arr  # fallback: too many clipped means the distribution IS elevated
        self._baseline = float(np.percentile(arr_r, self._percentile))
        # Robust sigma from IQR (not sensitive to spike outliers)
        q25, q75 = float(np.percentile(arr_r, 25)), float(np.percentile(arr_r, 75))
        sigma = (q75 - q25) / 1.349
        self._threshold = self._baseline + self._sigma_multiplier * sigma
        self._last_recompute = t_sec

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False
        self._last_recompute = -self._recompute_interval

    def is_frozen(self) -> bool:
        return self._frozen

    def clear(self) -> None:
        """Clear the rolling window and reset cached baseline/threshold.

        Call after a flare ends so the window is not contaminated by
        the elevated flare flux, which would suppress the threshold on
        the next quiet period.
        """
        self._window.clear()
        self._baseline = None
        self._threshold = None
        self._last_recompute = -self._recompute_interval

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold
