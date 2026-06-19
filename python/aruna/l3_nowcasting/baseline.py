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

    def update(self, flux: float, t_sec: int) -> Tuple[Optional[float], Optional[float]]:
        self._window.append(flux)
        if not self._frozen and (t_sec - self._last_recompute) >= self._recompute_interval:
            self._recompute(t_sec)
        return self._baseline, self._threshold

    def _recompute(self, t_sec: int) -> None:
        data = list(self._window)
        if not data:
            return
        self._baseline = float(np.percentile(data, self._percentile))
        quiet = data[:3600]
        sigma = float(np.std(quiet)) if len(quiet) >= 2 else 0.0
        self._threshold = self._baseline + self._sigma_multiplier * sigma
        self._last_recompute = t_sec

    def freeze(self) -> None:
        self._frozen = True

    def unfreeze(self) -> None:
        self._frozen = False
        self._last_recompute = -self._recompute_interval

    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def baseline(self) -> Optional[float]:
        return self._baseline

    @property
    def threshold(self) -> Optional[float]:
        return self._threshold
