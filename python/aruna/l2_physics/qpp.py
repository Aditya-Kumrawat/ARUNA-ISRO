"""
Quasi-Periodic Pulsation (QPP) Detector using Morlet Continuous Wavelet Transform.

Detects periodic oscillations in solar flare X-ray emission that may precede
or accompany major flares. Requires pywt and statsmodels for CWT and detrending.
"""
from __future__ import annotations
import logging
from collections import deque
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pywt
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    _QPP_AVAILABLE = True
except ImportError:
    _QPP_AVAILABLE = False
    logger.warning("QPPDetector: pywt/statsmodels unavailable — QPP disabled")


class QPPDetector:
    """
    Detect quasi-periodic pulsations in X-ray flux time series.

    Uses Morlet continuous wavelet transform on detrended flux to extract
    dominant oscillation periods and amplitudes. Returns (None, None) during
    warm-up or if dependencies are unavailable.
    """

    def __init__(self, window: int = 200, min_frames: int = 50) -> None:
        """
        Args:
            window: Maximum buffer length.
            min_frames: Minimum samples before analysis begins.
        """
        self._window     = window
        self._min_frames = min_frames
        self._buf: deque = deque(maxlen=window)

    def update(self, hel_25_50_flux: float) -> Tuple[Optional[float], Optional[float]]:
        """
        Add one flux sample and detect QPP if buffer has reached min_frames.

        Args:
            hel_25_50_flux: HEL1OS 25-50 keV flux in counts/s.

        Returns:
            (period_sec, amplitude) if detected, else (None, None).
            period_sec: Dominant oscillation period in seconds (samples).
            amplitude: Relative power amplitude (dimensionless).
        """
        self._buf.append(hel_25_50_flux)
        if len(self._buf) < self._min_frames or not _QPP_AVAILABLE:
            return None, None
        return self._detect(np.array(self._buf))

    def _detect(self, data: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """
        Detect QPP period and amplitude using Morlet CWT.

        Args:
            data: Flux time series.

        Returns:
            (period, amplitude) or (None, None) on failure.
        """
        try:
            # Detrend using LOWESS (locally weighted scatterplot smoother)
            trend     = _lowess(data, np.arange(len(data)), frac=0.3)[:, 1]
            detrended = data - trend

            # Continuous wavelet transform with Morlet wavelet
            scales    = np.arange(5, 200)
            coeffs, _ = pywt.cwt(detrended, scales, "morl", sampling_period=1.0)

            # Power spectrum: |coefficient|^2
            power      = np.abs(coeffs) ** 2
            mean_power = np.mean(power, axis=1)

            # Threshold: noise floor at 99th percentile
            noise_floor = np.percentile(mean_power, 99)
            mask        = mean_power > noise_floor

            if not mask.any():
                return None, None

            # Dominant scale index and corresponding period
            dominant_idx = np.argmax(mean_power[mask])
            dominant = scales[mask][dominant_idx]

            # Amplitude: ratio of peak power to mean
            A_QPP = float(np.max(mean_power) / (np.mean(mean_power) + 1e-30))

            return float(dominant), A_QPP

        except Exception as exc:
            logger.warning("QPP detection failed: %s", exc)
            return None, None
