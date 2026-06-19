from dataclasses import dataclass, field

@dataclass
class SafingEntry:
    name: str
    orbit: str
    vulnerability: dict[str, float]   # e.g., {"X": 1.0, "M5": 0.85}

@dataclass
class ThreatOutput:
    navic_risk: dict[str, str]        # region → level string (e.g. 'HIGH')
    navic_prob: dict[str, float]      # region → 0.0–1.0
    hf_risk: str
    hf_duration_min: int
    radiation_risk: str
    gaganyaan_advisory: str
    safing_queue: list[SafingEntry]   # ranked by vulnerability for this goes_class
    dos: list[str]
    donts: list[str]
    bulletin_text: str
    source_event_id: str
