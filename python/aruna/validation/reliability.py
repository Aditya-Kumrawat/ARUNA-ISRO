from __future__ import annotations
import numpy as np
from typing import Tuple


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Expected Calibration Error using equal-width bins.

    Parameters
    ----------
    probs  : (N,) predicted probabilities for the positive class
    labels : (N,) binary ground-truth labels (0 or 1)
    n_bins : number of equal-width bins in [0, 1]

    Returns
    -------
    (ece, bin_centres, bin_accuracies, bin_counts)
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs >= lo) & (probs < hi) if i < n_bins - 1 else (probs >= lo) & (probs <= hi)
        bin_counts[i] = int(mask.sum())
        if bin_counts[i] > 0:
            bin_accuracies[i] = float(labels[mask].mean())

    n = len(probs)
    ece = float(np.sum(bin_counts * np.abs(bin_centres - bin_accuracies)) / max(n, 1))
    return ece, bin_centres, bin_accuracies, bin_counts


def ece_scalar(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Convenience wrapper — returns just the ECE scalar."""
    ece, _, _, _ = compute_ece(probs, labels, n_bins=n_bins)
    return ece
