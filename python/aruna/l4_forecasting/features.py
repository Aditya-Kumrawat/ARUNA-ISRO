"""
FeatureExtractor — maps a sliding window of PhysicsFrame objects into
three PyTorch tensors for the DualStreamTCN:
  tensor_thermal    (1, T, 8) — SoLEXS plasma diagnostics + rates
  tensor_nonthermal (1, T, 8) — HEL1OS spectral + QPP features
  tensor_shared     (1, T, 8) — HFD + raw band fluxes + one-hot flare phase
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Lazy import — torch not available in all test environments, but always required for L4
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from aruna.models.physics import PhysicsFrame
from .models import N_THERMAL, N_NONTHERMAL, N_SHARED, ForecastConfig

# ── One-hot encoding for flare_phase ─────────────────────────────────────────
_PHASE_INDEX = {"QUIET": 0, "HOPE": 1, "IMPULSIVE": 2, "GRADUAL": 3, "DECAY": 4}


def _nan_to_zero(v: Optional[float]) -> float:
    """Replace NaN and None with 0.0 — safe default for warm-up or missing values."""
    if v is None:
        return 0.0
    if isinstance(v, float) and math.isnan(v):
        return 0.0
    return float(v)


def _phase_onehot(phase: str) -> List[float]:
    """5-element one-hot for flare_phase. Unknown phase → all zeros."""
    vec = [0.0] * 5
    idx = _PHASE_INDEX.get(phase, -1)
    if idx >= 0:
        vec[idx] = 1.0
    return vec


class FeatureExtractor:
    """
    Converts a list[PhysicsFrame] (one sliding window) into
    (tensor_thermal, tensor_nonthermal, tensor_shared), each shape (1, T, F_i).

    Parameters
    ----------
    scaler : optional fitted sklearn StandardScaler or compatible object.
             If None, raw values are used (test mode).
             If provided, scaler.transform() is called on the concatenated
             (T, 24) feature matrix before splitting back into streams.
    """

    def __init__(self, scaler=None) -> None:
        self._scaler = scaler

    @classmethod
    def from_file(cls, scaler_path: str) -> "FeatureExtractor":
        """Load a pre-fitted scaler from pickle. Returns FeatureExtractor(scaler=None) on missing."""
        path = Path(scaler_path)
        if not path.exists():
            import logging
            logging.getLogger(__name__).warning(
                "Feature scaler not found at %s — using raw values", scaler_path
            )
            return cls(scaler=None)
        with open(path, "rb") as fh:
            scaler = pickle.load(fh)
        return cls(scaler=scaler)

    def extract(
        self, frames: List[PhysicsFrame]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Parameters
        ----------
        frames : list of PhysicsFrame, must be non-empty and ordered chronologically.

        Returns
        -------
        (tensor_thermal, tensor_nonthermal, tensor_shared)
        Each shape: (1, T, F_i)

        Raises
        ------
        ValueError  if frames is empty.
        RuntimeError if torch is not installed.
        """
        if not frames:
            raise ValueError("frames must be non-empty")
        if not _TORCH_AVAILABLE:
            raise RuntimeError("torch is required for FeatureExtractor.extract()")

        T = len(frames)
        thermal_rows: List[List[float]] = []
        nonthermal_rows: List[List[float]] = []
        shared_rows: List[List[float]] = []

        for f in frames:
            # ── Thermal stream (9) ────────────────────────────────────────────
            thermal_rows.append([
                _nan_to_zero(f.T_MK),
                _nan_to_zero(f.EM_cm3),
                _nan_to_zero(f.DNI),
                _nan_to_zero(f.tnpsv_magnitude),
                _nan_to_zero(f.tnpsv_angle_rad),
                _nan_to_zero(f.dT_dt),
                _nan_to_zero(f.dEM_dt),
                _nan_to_zero(f.d2T_dt2),
                _nan_to_zero(getattr(f, 'tnci', 0.0)),   # TNCI (PRD §13.4)
            ])

            # ── Non-thermal stream (8) ────────────────────────────────────────
            nonthermal_rows.append([
                _nan_to_zero(f.SHEI),
                _nan_to_zero(f.dSHEI_dt),
                _nan_to_zero(f.FeKa_flux),
                _nan_to_zero(f.hel1os_10_25kev),
                _nan_to_zero(f.hel1os_25_50kev),
                _nan_to_zero(f.hel1os_50_70kev),
                _nan_to_zero(f.QPP_period_sec),    # None → 0.0
                _nan_to_zero(f.QPP_amplitude),     # None → 0.0
            ])

            # ── Shared stream (10) ────────────────────────────────────────────
            phase_vec = _phase_onehot(f.flare_phase)  # 5 values
            shared_rows.append([
                _nan_to_zero(f.HFD),                  # NaN during warm-up → 0.0
                _nan_to_zero(f.solexs_3_6kev),
                _nan_to_zero(f.solexs_6_10kev),
                _nan_to_zero(getattr(f, 'suit_uv_flux', 0.0)),            # SUIT UV
                _nan_to_zero(getattr(f, 'suit_uv_brightening_rate', 0.0)),# dUV/dt
            ] + phase_vec)  # 5 + 5 = 10

        # Stack to numpy (T, F_i) — use float64 first to avoid overflow on large EM values
        # (EM_cm3 ~ 1e47–1e49 overflows float32 max ~3.4e38; a fitted scaler normalises these)
        arr_thermal = np.array(thermal_rows, dtype=np.float64)
        arr_nonthermal = np.array(nonthermal_rows, dtype=np.float64)
        arr_shared = np.array(shared_rows, dtype=np.float64)

        # Apply scaler on full concatenated matrix if available
        if self._scaler is not None:
            full = np.concatenate([arr_thermal, arr_nonthermal, arr_shared], axis=1)  # (T, 24)
            full = self._scaler.transform(full)
            arr_thermal = full[:, :N_THERMAL]
            arr_nonthermal = full[:, N_THERMAL:N_THERMAL + N_NONTHERMAL]
            arr_shared = full[:, N_THERMAL + N_NONTHERMAL:]

        # Convert to (1, T, F_i) float32 tensors — batch dimension first
        t_thermal = torch.tensor(arr_thermal, dtype=torch.float32).unsqueeze(0)
        t_nonthermal = torch.tensor(arr_nonthermal, dtype=torch.float32).unsqueeze(0)
        t_shared = torch.tensor(arr_shared, dtype=torch.float32).unsqueeze(0)

        return t_thermal, t_nonthermal, t_shared
