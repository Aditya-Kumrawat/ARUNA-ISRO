from aruna.models.nowcast import NowcastEvent
from .rules import (
    NAVIC_BASE_RISK,
    EIA_AMPLIFIED_REGIONS,
    RISK_LEVEL_ORDER,
)

# Map from level string to probability
PROBABILITY_MAP = {
    "NEGLIGIBLE": 0.02,
    "LOW": 0.15,
    "MODERATE": 0.30,
    "HIGH": 0.55,
    "VERY_HIGH": 0.67,
    "EXTREME": 0.90,
}

def parse_goes_band(goes_class: str, goes_subclass: float) -> str:
    """Explicit banding of GOES class and subclass."""
    c = goes_class.upper()
    if c == "B":
        return "B"
    elif c == "C":
        return "C"
    elif c == "M":
        return "M1" if goes_subclass < 5.0 else "M5"
    elif c == "X":
        return "X1" if goes_subclass < 5.0 else "X5"
    else:
        return "B"  # Default fallback for A, SUB_A, etc.

def compute_navic_risk(event: NowcastEvent) -> tuple[dict[str, str], dict[str, float]]:
    """
    Computes NavIC risk level and probability for different regions.
    Returns: (navic_risk_dict, navic_prob_dict)
    """
    band = parse_goes_band(event.goes_class, event.goes_subclass)
    base_level, _ = NAVIC_BASE_RISK.get(band, ("NEGLIGIBLE", 0.02))
    
    regions = ["south_india", "north_india", "island_chains"]
    navic_risk = {}
    navic_prob = {}

    for region in regions:
        # Find base index
        idx = RISK_LEVEL_ORDER.index(base_level)
        
        # 1. confidence > 0.90 -> +1 level
        if event.confidence > 0.90:
            idx += 1
            
        # 2. confidence < 0.50 -> -1 level
        if event.confidence < 0.50:
            idx -= 1
            
        # 3. flare_phase == "IMPULSIVE" -> +1 for EIA regions
        # We map event.alert_type in ("FLARE_ONSET", "FLARE_PEAK") as IMPULSIVE phase
        is_impulsive = event.alert_type in ("FLARE_ONSET", "FLARE_PEAK")
        if is_impulsive and region in EIA_AMPLIFIED_REGIONS:
            idx += 1
            
        # 4. peak_flux > class_midpoint * 3 -> +1 level
        # subclass > 3.0 represents flux > 3.0 * class_base
        if event.goes_subclass > 3.0:
            idx += 1
            
        # Cap index between NEGLIGIBLE and EXTREME
        idx = max(0, min(idx, len(RISK_LEVEL_ORDER) - 1))
        
        # 5. Safety clamp: X-class eventsfloor at VERY_HIGH, never downgrade below
        if event.goes_class.upper() == "X":
            very_high_idx = RISK_LEVEL_ORDER.index("VERY_HIGH")
            idx = max(idx, very_high_idx)
            
        final_level = RISK_LEVEL_ORDER[idx]
        navic_risk[region] = final_level
        navic_prob[region] = PROBABILITY_MAP[final_level]

    return navic_risk, navic_prob
