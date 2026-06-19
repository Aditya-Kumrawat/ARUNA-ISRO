"""
XL-1 Magnetogram Integration — data schemas.

MagneticContext is the canonical output of one SHARP observation window.
It is optional everywhere downstream: None means "SHARP unavailable" and
every consumer must continue operating without it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SHARPRecord:
    """
    Raw SHARP parameter values for one HARP at one timestamp.

    All values are as returned by JSOC hmi.sharp_cea_720s before any
    normalisation. Units follow the JSOC keyword definitions.
    """
    harpnum: int                    # HARPNum (HARP = HMI Active Region Patch)
    noaa_ar: Optional[int]          # NOAA AR number (may be 0 if unassigned)
    t_rec: datetime                 # observation time (TAI, stored as UTC equivalent)

    # ── SHARP parameters (Bobra & Couvidat 2015 feature set) ────────────────
    usflux:    float                # Total unsigned flux [Mx]
    totusjh:   float                # Total unsigned current helicity [G²/m]
    totusjz:   float                # Total unsigned vertical current [A]
    totbsq:    float                # Total magnitude of Lorentz force [dyn/cm²]
    totpot:    float                # Proxy for free magnetic energy [erg/cm]
    savncpp:   float                # Schrijver R-value based quantity
    absnjzh:   float                # Absolute value of net current helicity
    r_value:   float                # Wang-Sheffield R (log-scale energy proxy)
    meanshr:   float                # Mean shear angle [deg]
    meangam:   float                # Mean inclination of field to normal [deg]

    # ── AR geometry ──────────────────────────────────────────────────────────
    lat_fwt:   float = 0.0          # Flux-weighted centroid latitude [deg]
    lon_fwt:   float = 0.0          # Flux-weighted centroid longitude [deg]
    area_acr:  float = 0.0          # Area of core (strong-field) pixels [μhem]

    # ── Source provenance ─────────────────────────────────────────────────────
    source: str = "jsoc_sharp_cea_720s"


@dataclass
class MagneticContext:
    """
    Processed magnetic context for one active region at one snapshot.

    flare_probability_24h is a heuristic estimate from SHARP parameters
    (Bobra & Couvidat 2015 logistic regression approximation). When SHARP
    data is absent the entire MagneticContext is None, not this field.
    """
    active_region_id: str           # "HARP{harpnum}" or "NOAA{ar_number}"
    timestamp: datetime

    # ── Core SHARP values (normalised 0–1 for downstream use) ────────────────
    usflux: float                   # total unsigned flux (normalised)
    r_value: float                  # Wang-Sheffield R value (raw, not normalised)
    shear: float                    # mean shear angle [deg]
    current_helicity: float         # total unsigned current helicity (normalised)
    magnetic_energy: float          # free energy proxy TOTPOT (normalised)

    # ── Derived risk estimate ─────────────────────────────────────────────────
    flare_probability_24h: Optional[float]  # P(M+X flare in next 24h), 0–1
    flare_probability_6h: Optional[float]   # P(M+X flare in next 6h), 0–1

    # ── Fusion modifier (applied by SHARPFusionLayer) ────────────────────────
    confidence_boost: float = 0.0   # +/− applied to ForecastFrame prob_M + prob_X

    # ── Provenance ────────────────────────────────────────────────────────────
    source: str = "jsoc_sharp_cea_720s"
    raw_harpnum: Optional[int] = None
    schema_version: str = "1.0"


@dataclass
class SHARPConfig:
    """Tunable parameters for the magnetogram integration layer."""

    # ── JSOC data source ─────────────────────────────────────────────────────
    jsoc_series: str = "hmi.sharp_cea_720s"
    jsoc_base_url: str = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info"
    cadence_minutes: int = 12           # SHARP 720s = 12-min cadence
    lookback_hours: int = 6             # how far back to fetch for a current snapshot

    # ── SHARP parameters to fetch ────────────────────────────────────────────
    sharp_keywords: tuple = field(default_factory=lambda: (
        "USFLUX", "TOTUSJH", "TOTUSJZ", "TOTBSQ", "TOTPOT",
        "SAVNCPP", "ABSNJZH", "R_VALUE", "MEANSHR", "MEANGAM",
        "LAT_FWT", "LON_FWT", "AREA_ACR", "NOAA_AR",
    ))

    # ── Probability thresholds (Bobra & Couvidat 2015 approximation) ─────────
    r_value_high:   float = 15.0        # R > 15 → high 24h risk
    r_value_medium: float = 10.0        # R > 10 → moderate risk
    r_value_low:    float = 6.0         # R > 6  → elevated risk
    totpot_high:    float = 4e23        # [erg/cm]
    totpot_medium:  float = 1e23
    totusjh_high:   float = 1e20        # [G²/m]
    totusjh_medium: float = 1e19
    meanshr_high:   float = 60.0        # [deg]
    meanshr_medium: float = 45.0

    # ── Fusion weights ────────────────────────────────────────────────────────
    boost_high: float   =  0.20         # P(M+X) boost when 24h prob > 0.75
    boost_medium: float =  0.10         # boost when 24h prob > 0.50
    dampen: float       = -0.08         # dampening when 24h prob < 0.10

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_dir: str = "data/sharp_cache"
    cache_ttl_hours: int = 2            # re-fetch after 2 hours

    # ── Kafka ─────────────────────────────────────────────────────────────────
    kafka_topic: str = "aruna.xl.ar_state"
