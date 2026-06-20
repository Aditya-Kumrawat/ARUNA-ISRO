"""
DualStreamTCN — dual-stream Temporal Convolutional Network for solar flare
probability forecasting. Exactly matches PRD §8.4 architecture.

Two parallel TCN streams capture SoLEXS thermal and HEL1OS non-thermal signals.
Cross-attention learns the thermal→non-thermal timing relationship (HOPE lead time).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TemporalBlock(nn.Module):
    """
    Standard TCN residual block: causal dilated Conv1d + BatchNorm + ReLU + Dropout.
    Residual projection applied when in_channels != out_channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # Causal padding: pad left only so future frames are not visible
        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Residual projection — only when dimensions change
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)
        residual = x if self.residual_proj is None else self.residual_proj(x)

        out = self.conv1(x)[:, :, :x.shape[2]]   # causal trim
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)[:, :, :x.shape[2]]  # causal trim
        out = self.bn2(out)
        out = self.relu(out + residual)
        return self.dropout(out)


def _build_tcn_stack(
    num_inputs: int,
    num_channels: List[int],
    kernel_size: int,
    dropout: float,
) -> nn.Sequential:
    """Build a stack of TemporalBlocks with exponentially increasing dilation."""
    layers = []
    in_ch = num_inputs
    for i, out_ch in enumerate(num_channels):
        dilation = 2 ** i
        layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        in_ch = out_ch
    return nn.Sequential(*layers)


class DualStreamTCN(nn.Module):
    """
    Dual-stream TCN for solar flare probability estimation.

    Architecture (PRD §8.4):
    - tcn_thermal:    SoLEXS features   → channels [64, 64, 128, 128], kernel=7
    - tcn_nonthermal: HEL1OS features   → channels [64, 64, 128, 128], kernel=7
    - cross_attention: nonthermal attends to thermal (learns thermal→nonthermal lag)
    - shared_encoder: shared features   → channels [32, 64], kernel=5
    - classifier: Linear(128+128+64, 256) → ReLU → Dropout(0.3) → Linear(256, 5)
    - No Softmax in forward — applied externally (asymmetric loss operates on logits)

    Input feature sizes:
      n_thermal    = 9   (T_MK, EM, DNI, TNPSV mag, TNPSV angle, dT/dt, dEM/dt, d²T/dt², TNCI)
      n_nonthermal = 8   (SHEI, dSHEI/dt, FeKa, HEL bands ×3, QPP period, QPP amp)
      n_shared     = 10  (HFD, SoLEXS bands ×2, SUIT UV flux, SUIT dUV/dt, flare_phase one-hot ×5)
    """

    def __init__(
        self,
        n_thermal: int = 9,
        n_nonthermal: int = 8,
        n_shared: int = 10,
        n_classes: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.tcn_thermal = _build_tcn_stack(
            num_inputs=n_thermal,
            num_channels=[64, 64, 128, 128],
            kernel_size=7,
            dropout=dropout,
        )
        self.tcn_nonthermal = _build_tcn_stack(
            num_inputs=n_nonthermal,
            num_channels=[64, 64, 128, 128],
            kernel_size=7,
            dropout=dropout,
        )
        # Cross-attention: nonthermal (query) attends to thermal (key/value)
        # embed_dim must match final TCN channel count = 128
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=8, batch_first=True, dropout=dropout
        )
        self.shared_encoder = _build_tcn_stack(
            num_inputs=n_shared,
            num_channels=[32, 64],
            kernel_size=5,
            dropout=dropout,
        )
        # Classifier head: 128 (thermal) + 128 (cross-attn) + 64 (shared) = 320
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
            # NOTE: no Softmax here — applied externally for asymmetric loss
        )

    def forward(
        self,
        x_thermal: torch.Tensor,     # (B, T, n_thermal)
        x_nonthermal: torch.Tensor,  # (B, T, n_nonthermal)
        x_shared: torch.Tensor,      # (B, T, n_shared)
    ) -> torch.Tensor:               # (B, n_classes)
        # TCN expects (B, C, T) — permute from (B, T, C)
        h_thermal = self.tcn_thermal(x_thermal.permute(0, 2, 1))       # (B, 128, T)
        h_nonthermal = self.tcn_nonthermal(x_nonthermal.permute(0, 2, 1))  # (B, 128, T)
        h_shared = self.shared_encoder(x_shared.permute(0, 2, 1))      # (B, 64, T)

        # Cross-attention on sequence dim: (B, T, 128) format (batch_first=True)
        h_th_seq = h_thermal.permute(0, 2, 1)        # (B, T, 128)
        h_nth_seq = h_nonthermal.permute(0, 2, 1)    # (B, T, 128)
        h_cross, _ = self.cross_attention(h_nth_seq, h_th_seq, h_th_seq)  # nonthermal attends to thermal
        h_cross_last = h_cross[:, -1, :]             # (B, 128) — last time step

        # Take last time step from each stream
        h_thermal_last = h_thermal[:, :, -1]         # (B, 128)
        h_shared_last = h_shared[:, :, -1]           # (B, 64)

        h_cat = torch.cat([h_thermal_last, h_cross_last, h_shared_last], dim=1)  # (B, 320)
        return self.classifier(h_cat)                # (B, 5) logits


def load_tcn(
    path: str = "models/tcn_v1.pt",
    device: str = "cpu",
    n_thermal: int = 9,
    n_nonthermal: int = 8,
    n_shared: int = 10,
) -> DualStreamTCN:
    """
    Load DualStreamTCN weights from a .pt file.

    If the file does not exist (test environment without trained weights):
    - Returns a randomly-initialised model in eval() mode
    - Logs a WARNING so the absence is always visible in logs

    Parameters
    ----------
    path   : path to .pt weights file
    device : torch device string ('cpu' | 'cuda')
    """
    model = DualStreamTCN(
        n_thermal=n_thermal,
        n_nonthermal=n_nonthermal,
        n_shared=n_shared,
    )
    weights_path = Path(path)
    if not weights_path.exists():
        logger.warning(
            "TCN weights not found at %s — using randomly-initialised model. "
            "TSS/AUC metrics will not be meaningful until tcn_v1.pt is trained.",
            path,
        )
        model.eval()
        return model

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("DualStreamTCN loaded from %s on %s", path, device)
    return model


def physics_heuristic_probs(
    hope_flag: bool,
    DNI: float,
    SHEI: float,
    T_MK: float,
) -> List[float]:
    """
    Physics-based class probability estimate when TCN weights are unavailable.

    Lookup table derived from Sarwade et al. 2025 X6.3 analysis and GOES statistics:
      HOPE + DNI>1.5 + SHEI>-0.8  → strong impulsive X-class precursor
      DNI>1.5 only                 → non-thermal dominant → likely M or X
      T_MK>20 only                 → hot thermal → C–M range
      Otherwise                    → quiet or B-class background

    Returns [p_quiet, p_B, p_C, p_M, p_X] summing to 1.0.
    """
    if hope_flag and DNI > 1.5 and SHEI > -0.8:
        probs = [0.00, 0.00, 0.05, 0.20, 0.75]
    elif hope_flag and DNI > 1.5:
        probs = [0.00, 0.02, 0.10, 0.33, 0.55]
    elif DNI > 1.5:
        probs = [0.01, 0.04, 0.15, 0.45, 0.35]
    elif T_MK > 20.0:
        probs = [0.05, 0.10, 0.35, 0.35, 0.15]
    elif T_MK > 10.0:
        probs = [0.15, 0.35, 0.35, 0.10, 0.05]
    else:
        probs = [0.60, 0.25, 0.10, 0.04, 0.01]

    total = sum(probs)
    return [p / total for p in probs]


def asymmetric_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Asymmetric focal loss for space weather flare classification.

    CRITICAL: Standard MSE penalises over- and under-prediction equally.
    For operational solar forecasting, missing an M5+ event is catastrophically
    worse than a false alarm. This loss penalises under-prediction 4× more.

    L = alpha * over_pred² + (1-alpha) * under_pred²
        ^-- 0.2 (over)           ^-- 0.8 (under) — 4× heavier

    Parameters
    ----------
    logits  : (B, 5) raw TCN output — Softmax applied internally
    targets : (B, 5) soft one-hot labels
    alpha   : over-prediction penalty weight (0.2 → under-pred = 0.8×)
    gamma   : focal loss focusing parameter (downweights easy negatives)
    """
    probs = torch.softmax(logits, dim=1)
    over_pred = torch.clamp(probs - targets, min=0.0)
    under_pred = torch.clamp(targets - probs, min=0.0)
    loss = alpha * over_pred ** 2 + (1.0 - alpha) * under_pred ** 2

    # Focal weighting — down-weight easy negatives
    pt = (probs * targets).sum(dim=1)
    focal_weight = (1.0 - pt) ** gamma

    return (focal_weight.unsqueeze(1) * loss).sum(dim=1).mean()
