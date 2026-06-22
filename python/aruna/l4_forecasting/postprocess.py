"""
Post-processing for L4 forecast probabilities — rare-event handling without
retraining the underlying models.

Two transforms, both standard and citable:

1. prior_correction (logit adjustment, Menon et al. 2020, "Long-tail learning
   via logit adjustment"): divides each class probability by its training-set
   prior raised to a power tau, then renormalises. This undoes the base-rate
   bias a model learns on heavily imbalanced data (98.5% QUIET here), lifting
   probability mass toward rare M/X classes.

   IMPORTANT: prior correction amplifies whatever ranking signal a model has.
   It cannot manufacture discrimination — a model with TSS≈0 stays useless.
   It makes well-ranked-but-under-confident rare-class scores usable.

2. blend_with_physics: convex combination of a physics heuristic vector with
   the (corrected) ML vector. The physics signal is the reliable backbone for
   physically-obvious events; the ML adds data-driven nuance.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_EPS = 1e-9


def prior_correction(
    probs: np.ndarray,                 # (5,) model probability vector
    class_priors: Sequence[float],     # (5,) training-set class frequencies
    tau: float = 0.5,                  # 0 = no correction, 1 = full inverse-frequency
) -> np.ndarray:
    """
    Logit-adjust ``probs`` against ``class_priors``.

    p_adj[k] ∝ probs[k] / (prior[k] ** tau)

    Returns a (5,) float32 vector that sums to 1.0. With tau=0 returns the
    input unchanged (renormalised).
    """
    probs = np.asarray(probs, dtype=np.float64)
    priors = np.asarray(class_priors, dtype=np.float64)

    if probs.shape != priors.shape:
        # Defensive: shape mismatch → return input untouched
        s = probs.sum()
        return (probs / s if s > 0 else probs).astype(np.float32)

    if tau <= 0.0:
        s = probs.sum()
        return (probs / s if s > 0 else probs).astype(np.float32)

    adjusted = probs / np.power(priors + _EPS, tau)
    adjusted = np.clip(adjusted, 0.0, None)
    total = adjusted.sum()
    if total <= 0.0:
        s = probs.sum()
        return (probs / s if s > 0 else probs).astype(np.float32)
    return (adjusted / total).astype(np.float32)


def blend_with_physics(
    ml_probs: np.ndarray,              # (5,) corrected ML probabilities
    physics_probs: Sequence[float],    # (5,) physics heuristic probabilities
    physics_weight: float = 0.5,       # weight on the physics vector (0..1)
) -> np.ndarray:
    """
    Convex blend: ``w * physics + (1 - w) * ml``.

    Both inputs are renormalised first so the result is a valid distribution
    regardless of minor rounding in the inputs. Returns (5,) float32.
    """
    ml = np.asarray(ml_probs, dtype=np.float64)
    phys = np.asarray(physics_probs, dtype=np.float64)

    w = float(np.clip(physics_weight, 0.0, 1.0))

    ml_sum = ml.sum()
    phys_sum = phys.sum()
    if ml_sum > 0:
        ml = ml / ml_sum
    if phys_sum > 0:
        phys = phys / phys_sum

    blended = w * phys + (1.0 - w) * ml
    total = blended.sum()
    if total <= 0.0:
        return ml.astype(np.float32)
    return (blended / total).astype(np.float32)
