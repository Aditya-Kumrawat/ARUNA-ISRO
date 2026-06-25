from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from aruna.models.frame_header import FrameHeader

# Kintner model: S4 = k * (flux / flux_ref)^alpha per region
_REGIONS = ["North India", "Central India", "South India", "East India", "West India", "Island Chains"]

_REGION_K = {
    "North India":   0.45,
    "Central India": 0.55,
    "South India":   0.85,  # EIA crest
    "East India":    0.50,
    "West India":    0.48,
    "Island Chains": 0.80,  # EIA crest
}

_FLUX_REF = 1.0e-5   # M1-class flux reference
_S4_ALPHA = 0.38

_RISK_THRESHOLDS = [
    (0.80, "EXTREME"),
    (0.60, "VERY_HIGH"),
    (0.40, "HIGH"),
    (0.20, "MODERATE"),
    (0.05, "LOW"),
    (0.00, "NEGLIGIBLE"),
]

_SAT_RISK = [
    (2.0e-4, "CRITICAL"), (5.0e-5, "HIGH"), (1.0e-5, "MEDIUM"), (0.0, "LOW")
]
_EVA = [
    (2.0e-4, "ABORT"), (5.0e-5, "DELAY"), (1.0e-5, "CAUTION"), (0.0, "PROCEED")
]
_LAUNCH = [
    (5.0e-5, "SCRUB"), (1.0e-5, "HOLD"), (0.0, "PROCEED")
]


def _flux_to_s4(flux_wm2: float, k: float) -> float:
    ratio = max(flux_wm2, 1e-9) / _FLUX_REF
    return min(1.0, k * math.pow(ratio, _S4_ALPHA))


def _s4_to_risk(s4: float) -> str:
    for threshold, label in _RISK_THRESHOLDS:
        if s4 >= threshold:
            return label
    return "NEGLIGIBLE"


def _lookup(table: list, flux: float) -> str:
    for threshold, label in table:
        if flux >= threshold:
            return label
    return table[-1][1]


@dataclass
class MissionImpactFrame:
    header: FrameHeader
    navic_s4_per_region: Dict[str, float]
    navic_risk_per_region: Dict[str, str]
    satellite_risk: str
    navigation_risk: str
    communications_risk: str
    astronaut_eva_recommendation: str
    launch_recommendation: str
    hf_blackout_duration_min: int
    impact_summary: str
    schema_version: str = "1.0"
    pipeline_version: str = "2.0.0"


class MissionImpactLayer:
    def compute(
        self,
        goes_class: str,
        peak_flux_wm2: float,
        source: str = "LIVE",
    ) -> MissionImpactFrame:
        s4 = {r: _flux_to_s4(peak_flux_wm2, k) for r, k in _REGION_K.items()}
        risk = {r: _s4_to_risk(v) for r, v in s4.items()}

        sat_risk = _lookup(_SAT_RISK, peak_flux_wm2)
        eva = _lookup(_EVA, peak_flux_wm2)
        launch = _lookup(_LAUNCH, peak_flux_wm2)

        # Navigation risk = worst NavIC region
        nav_levels = ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "VERY_HIGH", "EXTREME"]
        nav_risk = max(risk.values(), key=lambda r: nav_levels.index(r) if r in nav_levels else 0)

        # HF blackout duration (minutes) — scaled with flux
        hf_min = int(min(480, max(0, 90 * math.log10(max(peak_flux_wm2, 1e-9) / 1e-7 + 1))))

        # Communications risk based on HF
        comms = "BLACKOUT" if hf_min > 120 else ("DEGRADED" if hf_min > 30 else "NOMINAL")

        summary = (
            f"{goes_class} event — NavIC risk {nav_risk} (South India worst), "
            f"EVA: {eva}, Launch: {launch}, HF blackout ~{hf_min} min."
        )

        return MissionImpactFrame(
            header=FrameHeader(timestamp=datetime.utcnow(), source=source),
            navic_s4_per_region=s4,
            navic_risk_per_region=risk,
            satellite_risk=sat_risk,
            navigation_risk=nav_risk,
            communications_risk=comms,
            astronaut_eva_recommendation=eva,
            launch_recommendation=launch,
            hf_blackout_duration_min=hf_min,
            impact_summary=summary,
        )
