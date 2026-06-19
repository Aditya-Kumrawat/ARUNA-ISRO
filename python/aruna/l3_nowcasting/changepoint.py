from __future__ import annotations
import logging
from collections import deque
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import ruptures as rpt
    _RPT_AVAILABLE = True
except ImportError:
    _RPT_AVAILABLE = False
    logger.warning("ruptures not installed — ChangepointDetector disabled")


class ChangepointDetector:
    def __init__(
        self,
        flux_window: int = 600,
        penalty: float = 1.5,
        min_size: int = 10,
        model: str = "rbf",
    ) -> None:
        self._window = flux_window
        self._penalty = penalty
        self._min_size = min_size
        self._model = model
        self._buf: deque = deque(maxlen=flux_window)

    def update(self, flux: float) -> Optional[int]:
        self._buf.append(flux)
        if len(self._buf) < self._window or not _RPT_AVAILABLE:
            return None
        return self._detect(np.array(self._buf, dtype=np.float64))

    def _detect(self, data: np.ndarray) -> Optional[int]:
        try:
            algo = rpt.Pelt(model=self._model, min_size=self._min_size).fit(data.reshape(-1, 1))
            bpts = algo.predict(pen=self._penalty)
            if len(bpts) > 1:
                return int(bpts[-2])
            return None
        except Exception as exc:
            logger.warning("PELT detection failed: %s", exc)
            return None
