from .models import SafingEntry

# GOES band maps to (level, base_probability)
NAVIC_BASE_RISK = {
    "B": ("NEGLIGIBLE", 0.02),
    "C": ("LOW", 0.15),
    "M1": ("MODERATE", 0.30),
    "M5": ("HIGH", 0.55),
    "X1": ("VERY_HIGH", 0.67),
    "X5": ("EXTREME", 0.90),
}

# GOES band maps to (blackout_label, duration_min)
HF_BLACKOUT = {
    "B": ("NONE", 0),
    "C": ("MINOR", 15),
    "M1": ("MODERATE", 45),
    "M5": ("STRONG", 120),
    "X1": ("SEVERE", 240),
    "X5": ("COMPLETE", 480),
}

# GOES band maps to Gaganyaan astronaut advisories
GAGANYAAN = {
    "B": "NOMINAL",
    "C": "NOMINAL",
    "M1": "ELEVATED",
    "M5": "HIGH - SHELTER ADVISORY",
    "X1": "CRITICAL - SHELTER REQUIRED",
    "X5": "EMERGENCY - EVA ABORT",
}

# GOES band maps to radiation risk level
RADIATION_RISK = {
    "B": "NOMINAL",
    "C": "NOMINAL",
    "M1": "ELEVATED",
    "M5": "HIGH",
    "X1": "VERY_HIGH",
    "X5": "EXTREME",
}

# Satellite database
SAFING_REGISTRY = [
    SafingEntry(
        name="Chandrayaan-3",
        orbit="LUNAR_POLAR",
        vulnerability={"B": 0.05, "C": 0.10, "M1": 0.30, "M5": 0.60, "X1": 0.95, "X5": 1.00}
    ),
    SafingEntry(
        name="Aditya-L1",
        orbit="L1_HALO",
        vulnerability={"B": 0.04, "C": 0.08, "M1": 0.25, "M5": 0.50, "X1": 0.90, "X5": 0.98}
    ),
    SafingEntry(
        name="RISAT-2BR1",
        orbit="LEO_SAR",
        vulnerability={"B": 0.01, "C": 0.05, "M1": 0.15, "M5": 0.40, "X1": 0.70, "X5": 0.85}
    ),
    SafingEntry(
        name="GSAT-30",
        orbit="GEO",
        vulnerability={"B": 0.02, "C": 0.06, "M1": 0.20, "M5": 0.45, "X1": 0.80, "X5": 0.90}
    ),
    SafingEntry(
        name="IRNSS-1I",
        orbit="MEO_NAVIC",
        vulnerability={"B": 0.03, "C": 0.07, "M1": 0.22, "M5": 0.48, "X1": 0.85, "X5": 0.92}
    ),
    SafingEntry(
        name="EOS-06",
        orbit="LEO_OCEAN",
        vulnerability={"B": 0.01, "C": 0.04, "M1": 0.12, "M5": 0.35, "X1": 0.65, "X5": 0.80}
    ),
]

EIA_AMPLIFIED_REGIONS = frozenset({"south_india", "island_chains"})

RISK_LEVEL_ORDER = ["NEGLIGIBLE", "LOW", "MODERATE", "HIGH", "VERY_HIGH", "EXTREME"]
