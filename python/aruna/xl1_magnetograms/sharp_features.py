"""
XL-1 SHARP Feature Extractor — derives flare probability and normalised
feature vector from raw SHARPRecord values.

Probability estimation uses the Bobra & Couvidat (2015) heuristic thresholds.
The full SVM/LightGBM model (XL-4 M1) replaces this once training data is
available; this heuristic operates from day 1 with zero labelled data.

Key physics:
  R_VALUE   — Wang-Sheffield energy proxy; best single 24h predictor
  TOTPOT    — free magnetic energy stored in the AR corona
  MEANSHR   — mean shear angle; >45° indicates pre-eruption configuration
  TOTUSJH   — unsigned current helicity; large values → complex topology
  dUSFLUX   — rapid flux emergence (computed from consecutive records) → 6h predictor
"""
from __future__ import annotations

import logging
import math
from typing import List, Optional

import numpy as np

from .models import MagneticContext, SHARPConfig, SHARPRecord

logger = logging.getLogger(__name__)

# ── Normalisation constants (approximate population medians for active ARs) ──
# Values chosen so that a moderate active region gives ~0.5 normalised score.
_NORM = {
    "usflux":   1e22,   # Mx
    "totusjh":  1e19,   # G²/m
    "totusjz":  5e11,   # A
    "totbsq":   1e4,    # dyn/cm²
    "totpot":   5e22,   # erg/cm
    "savncpp":  5e3,
    "absnjzh":  1e18,
    "meanshr":  40.0,   # deg
    "meangam":  30.0,   # deg
}


def _norm(value: float, key: str) -> float:
    """Sigmoid-normalise a SHARP parameter to (0, 1)."""
    scale = _NORM.get(key, 1.0)
    if scale == 0:
        return 0.0
    return float(1.0 / (1.0 + math.exp(-2.0 * value / scale + 1.0)))


def _r_value_norm(r: float) -> float:
    """R_VALUE is already logarithmic — normalise on a 0–20 scale."""
    return min(1.0, max(0.0, r / 20.0))


def _estimate_24h_probability(rec: SHARPRecord, cfg: SHARPConfig) -> float:
    """
    Heuristic P(M+X flare in next 24h) from SHARP parameters.

    Approximates the Bobra & Couvidat (2015) logistic regression using
    published threshold bands. Returns a value in [0, 1].

    Feature weights (relative importance from literature):
      TOTUSJH: 37%, TOTBSQ: 15%, TOTPOT: 14%, TOTUSJZ: 13%,
      ABSNJZH: 11%, SAVNCPP: 5%, USFLUX: 5%
    """
    score = 0.0

    # R_VALUE (primary discriminator)
    if rec.r_value >= cfg.r_value_high:
        score += 0.40
    elif rec.r_value >= cfg.r_value_medium:
        score += 0.25
    elif rec.r_value >= cfg.r_value_low:
        score += 0.12

    # TOTPOT (free magnetic energy)
    if rec.totpot >= cfg.totpot_high:
        score += 0.20
    elif rec.totpot >= cfg.totpot_medium:
        score += 0.10

    # TOTUSJH (current helicity — most important individual feature)
    if rec.totusjh >= cfg.totusjh_high:
        score += 0.22
    elif rec.totusjh >= cfg.totusjh_medium:
        score += 0.11

    # MEANSHR (shear angle)
    if rec.meanshr >= cfg.meanshr_high:
        score += 0.18
    elif rec.meanshr >= cfg.meanshr_medium:
        score += 0.09

    return min(1.0, score)


def _estimate_6h_probability(
    records: List[SHARPRecord], cfg: SHARPConfig
) -> float:
    """
    P(M+X flare in next 6h) estimated from rapid flux emergence rate.

    Rapid changes in USFLUX (dUSFLUX/dt) and TOTUSJH over the past 1–2 hours
    are strong short-range predictors (Liu et al. 2017, Kontogiannis et al. 2019).

    Requires ≥ 2 consecutive records (12-min cadence).
    Returns 0.0 if insufficient history.
    """
    if len(records) < 2:
        return 0.0

    # Use last 5 records (≈ 1 hour at 12-min cadence) for rate computation
    window = records[-5:]
    if len(window) < 2:
        return 0.0

    dt_hours = (window[-1].t_rec - window[0].t_rec).total_seconds() / 3600.0
    if dt_hours < 0.1:
        return 0.0

    # Flux emergence rate [Mx/hr]
    d_usflux = (window[-1].usflux - window[0].usflux) / dt_hours
    # Helicity injection rate [G²/m/hr]
    d_totusjh = (window[-1].totusjh - window[0].totusjh) / dt_hours

    score = 0.0
    # Rapid flux emergence threshold: 2e20 Mx/hr → elevated 6h risk
    if d_usflux > 4e20:
        score += 0.35
    elif d_usflux > 2e20:
        score += 0.20
    elif d_usflux > 1e20:
        score += 0.10

    # Helicity injection
    if d_totusjh > 1e18:
        score += 0.25
    elif d_totusjh > 5e17:
        score += 0.12

    # Boost if latest MEANSHR already elevated
    latest = records[-1]
    if latest.meanshr >= cfg.meanshr_high:
        score *= 1.4
    elif latest.meanshr >= cfg.meanshr_medium:
        score *= 1.2

    return min(1.0, score)


def build_magnetic_context(
    records: List[SHARPRecord],
    config: Optional[SHARPConfig] = None,
) -> Optional[MagneticContext]:
    """
    Build a MagneticContext from a list of SHARPRecords (sorted chronologically).

    Uses the latest record for snapshot values and the full list for rate
    features. Returns None if records is empty — caller must handle gracefully.
    """
    if not records:
        return None

    cfg = config or SHARPConfig()
    latest = records[-1]

    ar_label = (
        f"NOAA{latest.noaa_ar}" if latest.noaa_ar
        else f"HARP{latest.harpnum}"
    )

    p24 = _estimate_24h_probability(latest, cfg)
    p6  = _estimate_6h_probability(records, cfg)

    # Normalise key parameters to [0, 1] for downstream use
    return MagneticContext(
        active_region_id=ar_label,
        timestamp=latest.t_rec,
        usflux=_norm(latest.usflux, "usflux"),
        r_value=latest.r_value,                    # raw — more interpretable
        shear=latest.meanshr,                      # raw [deg] — dashboard-friendly
        current_helicity=_norm(latest.totusjh, "totusjh"),
        magnetic_energy=_norm(latest.totpot, "totpot"),
        flare_probability_24h=p24,
        flare_probability_6h=p6,
        source=latest.source,
        raw_harpnum=latest.harpnum,
    )


def as_feature_vector(ctx: MagneticContext) -> np.ndarray:
    """
    Convert MagneticContext to a (8,) float32 feature vector for XL-4 M1/M3.

    Features: [usflux_norm, r_value_norm, shear_norm, helicity_norm,
               energy_norm, p24h, p6h, r_value_raw/20]
    """
    return np.array([
        ctx.usflux,
        _r_value_norm(ctx.r_value),
        min(1.0, ctx.shear / 90.0),
        ctx.current_helicity,
        ctx.magnetic_energy,
        ctx.flare_probability_24h or 0.0,
        ctx.flare_probability_6h or 0.0,
        min(1.0, ctx.r_value / 20.0),
    ], dtype=np.float32)
