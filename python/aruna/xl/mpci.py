"""
XL-2 — Multi-Payload Coherence Index (MPCI).

Asks: do all four Aditya-L1 payloads tell a physically consistent story?

Physical timing ladder (PRD §13.3):
  1. SUIT UV brightening      — 2–5 min before peak (chromospheric precursor)
  2. SoLEXS soft X-ray rise   — 1–3 min before peak (HOPE signature)
  3. VELC Doppler shift        — 1–2 min before peak (plasma evaporation)
  4. HEL1OS hard X-ray onset  — 0 min (impulsive phase, this IS the peak)

MPCI > 0.80 → +20% confidence boost
MPCI < 0.50 → −15% confidence penalty + DEGRADED_CONFIDENCE flag
Ladder out-of-order → INCOHERENT flag
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from aruna.models.physics import PhysicsFrame

logger = logging.getLogger(__name__)


@dataclass
class MPCIResult:
    """Output of one MPCI computation window."""
    timestamp: datetime
    mpci: float                             # 0–1 normalised coherence score
    suit_score: float                       # 0–1 UV brightening contribution
    solexs_score: float                     # 0–1 thermal plasma contribution
    hel1os_score: float                     # 0–1 spectral hardening contribution
    velc_score: Optional[float]             # 0–1 Doppler shift; None if VELC absent
    available_payloads: List[str]
    confidence_modifier: float              # +0.20, 0.0, or −0.15
    flags: List[str] = field(default_factory=list)  # INCOHERENT, DEGRADED_CONFIDENCE
    schema_version: str = "1.0"


def compute_MPCI(
    suit_uv_score: float,
    solexs_score: float,
    hel1os_score: float,
    velc_score: Optional[float],
    available_payloads: List[str],
) -> float:
    """
    Compute MPCI from per-payload normalised scores (PRD §13.3, exact formula).

    Weights: SUIT=0.20, SoLEXS=0.35, HEL1OS=0.30, VELC=0.15.
    Re-normalised automatically when VELC is unavailable.

    Parameters
    ----------
    suit_uv_score    : UV brightening magnitude, normalised 0–1
    solexs_score     : T-rising / EM-flat signal, normalised 0–1
    hel1os_score     : spectral hardening SHEI signal, normalised 0–1
    velc_score       : Doppler proxy, normalised 0–1; None skips VELC
    available_payloads: list of payload names to include in sum

    Returns
    -------
    float in [0, 1]
    """
    weights = {'SUIT': 0.20, 'SoLEXS': 0.35, 'HEL1OS': 0.30, 'VELC': 0.15}
    scores  = {
        'SUIT':    suit_uv_score,
        'SoLEXS':  solexs_score,
        'HEL1OS':  hel1os_score,
        'VELC':    velc_score if velc_score is not None else 0.0,
    }
    payloads = [p for p in available_payloads if p in weights]
    if not payloads:
        return 0.0
    raw  = sum(weights[p] * scores[p] for p in payloads)
    norm = sum(weights[p]              for p in payloads)
    return raw / norm


def _suit_uv_score(frame: PhysicsFrame) -> float:
    """Convert SUIT UV brightening rate into a 0–1 score using a sigmoid gate."""
    rate = getattr(frame, 'suit_uv_flux', 0.0)
    if rate <= 0:
        return 0.0
    # Sigmoid normalisation — 1000 counts/s is a moderate brightening event
    return float(1.0 / (1.0 + math.exp(-rate / 1000.0 + 5.0)))


def _solexs_score(frame: PhysicsFrame) -> float:
    """Score = 1 when T is rising AND EM is flat (HOPE quadrant)."""
    dT  = getattr(frame, 'dT_dt', 0.0)
    dEM = getattr(frame, 'dEM_dt', 0.0)
    # T rising is the key signal; EM flat boosts the score
    if dT <= 0:
        return 0.0
    em_flat_bonus = 0.3 if abs(dEM) < 1e44 else 0.0
    return min(1.0, float(dT / 0.5) + em_flat_bonus)


def _hel1os_score(frame: PhysicsFrame) -> float:
    """Score derived from SHEI hardening (SHEI > −0.8 = impulsive onset)."""
    shei = getattr(frame, 'SHEI', -2.0)
    if shei <= -1.5:
        return 0.0
    return min(1.0, float((shei + 1.5) / 0.7))


def _velc_score_from_proxy(velc_flux: Optional[float]) -> Optional[float]:
    """0–1 score from raw VELC Doppler proxy. None if VELC absent."""
    if velc_flux is None or velc_flux == 0.0:
        return None
    return min(1.0, abs(velc_flux) / 50.0)   # 50 km/s = saturated score


class MPCIEvaluator:
    """
    Stateless evaluator that computes MPCIResult from one PhysicsFrame.

    Call evaluate() per frame; it does not maintain a rolling buffer
    because MPCI is a snapshot quantity — coherence at this instant.
    """

    def evaluate(
        self,
        frame: PhysicsFrame,
        velc_flux: Optional[float] = None,
    ) -> MPCIResult:
        """
        Parameters
        ----------
        frame     : current L2 PhysicsFrame (must have suit_uv_flux, SHEI, dT_dt)
        velc_flux : optional raw VELC Doppler proxy value [km/s equivalent]

        Returns
        -------
        MPCIResult with mpci score and confidence modifier
        """
        s_suit   = _suit_uv_score(frame)
        s_solexs = _solexs_score(frame)
        s_hel    = _hel1os_score(frame)
        s_velc   = _velc_score_from_proxy(velc_flux)

        available = ['SoLEXS', 'HEL1OS']
        suit_flux = getattr(frame, 'suit_uv_flux', 0.0)
        if suit_flux > 0.0:
            available.append('SUIT')
        if s_velc is not None:
            available.append('VELC')

        mpci = compute_MPCI(s_suit, s_solexs, s_hel, s_velc, available)

        flags: list[str] = []
        if mpci >= 0.80:
            confidence_modifier = 0.20
        elif mpci <= 0.50:
            confidence_modifier = -0.15
            flags.append('DEGRADED_CONFIDENCE')
        else:
            confidence_modifier = 0.0

        logger.debug(
            "MPCI=%.3f suit=%.2f sol=%.2f hel=%.2f velc=%s payloads=%s",
            mpci, s_suit, s_solexs, s_hel,
            f"{s_velc:.2f}" if s_velc is not None else "N/A",
            available,
        )

        return MPCIResult(
            timestamp=frame.timestamp,
            mpci=mpci,
            suit_score=s_suit,
            solexs_score=s_solexs,
            hel1os_score=s_hel,
            velc_score=s_velc,
            available_payloads=available,
            confidence_modifier=confidence_modifier,
            flags=flags,
        )
