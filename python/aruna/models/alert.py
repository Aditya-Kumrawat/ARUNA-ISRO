"""
Global data contract: unified alert published after L6 threat assessment.
AlertEvent is the top-level operational object consumed by L7 (LLM briefing)
and L8 (dashboard). It combines nowcast + forecast + threat into one record.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class AlertEvent:
    """
    Unified operational alert — issued after threat intelligence enrichment.
    Published to: (consumed by L7 and dashboard directly from L6 output)
    """

    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)

    # ── Flare characterisation ────────────────────────────────────────────────
    alert_type: str = ""               # 'HOPE_PRECURSOR'|'FLARE_ONSET'|'FLARE_PEAK'|'FLARE_END'
    goes_class: str = ""               # 'B'|'C'|'M'|'X'
    goes_subclass: float = 0.0
    confidence: float = 0.0            # 0.0–1.0
    lead_time_sec: int = 0

    # ── Physics state ─────────────────────────────────────────────────────────
    physics_state: str = "QUIET"       # 'HOPE'|'IMPULSIVE'|'GRADUAL'|'DECAY'
    peak_temperature_MK: float = 0.0
    peak_emission_measure: float = 0.0
    dni_at_alert: float = 0.0

    # ── Threat flags (from L6) ────────────────────────────────────────────────
    # Keys: 'navic_south_india', 'navic_north_india', 'hf_comms', 'gaganyaan', 'radiation'
    # Values: risk level string e.g. 'HIGH', 'MODERATE', 'NOMINAL'
    threat_flags: Dict[str, str] = field(default_factory=dict)

    # ── Instrument provenance ─────────────────────────────────────────────────
    instruments: List[str] = field(default_factory=list)

    # ── Provenance chain ─────────────────────────────────────────────────────
    nowcast_event_id: Optional[str] = None
    physics_frame_id: Optional[str] = None
    provenance_hash: str = ""          # SHA-256 of all input frames

    # ── Message bus metadata ──────────────────────────────────────────────────
    schema_version: str = "1.0"
    pipeline_version: str = "2.0.0"

    @property
    def goes_label(self) -> str:
        return f"{self.goes_class}{self.goes_subclass:.1f}"

    def is_critical(self) -> bool:
        """True for X3+ events — always escalated regardless of confidence."""
        return self.goes_class == "X" and self.goes_subclass >= 3.0

    def navic_risk(self, region: str = "south_india") -> str:
        return self.threat_flags.get(f"navic_{region}", "UNKNOWN")
