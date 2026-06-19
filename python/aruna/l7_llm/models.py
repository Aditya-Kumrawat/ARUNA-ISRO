from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from aruna.models.nowcast import NowcastEvent
from aruna.models.forecast import ForecastFrame
from aruna.l6_threat.models import ThreatOutput
from aruna.l5_analogs.models import AnalogMatch


@dataclass
class BriefingContext:
    nowcast: NowcastEvent
    forecast: ForecastFrame
    threat: ThreatOutput
    analogs: List[AnalogMatch]   # top-3, or [NO_ANALOG]


@dataclass
class BriefingJSON:
    """Structured output from the LLM — validated before rendering."""
    goes_class: str              # must match ctx.nowcast.goes_label exactly
    confidence_pct: float        # within ±2 of ctx.nowcast.confidence * 100
    lead_time_min: float         # within ±2 of ctx.forecast.lead_time_estimate_sec / 60
    navic_risk_summary: str      # must be a value in ctx.threat.navic_risk
    hf_risk: str                 # must match ctx.threat.hf_risk
    hf_duration_min: int         # must match ctx.threat.hf_duration_min
    radiation_risk: str          # must match ctx.threat.radiation_risk
    gaganyaan_advisory: str      # must match ctx.threat.gaganyaan_advisory
    summary: str                 # free narrative ≤ 3 sentences
    recommended_action: str      # free narrative ≤ 2 sentences
    operator_note: str           # free narrative ≤ 1 sentence
    analog_class: str            # goes_class of top analog, or "NONE"
    analog_similarity_pct: float # similarity * 100, or 0.0


@dataclass
class AlertBriefing:
    briefing_text: str
    briefing_source: str                 # "azure_gpt-4.1-mini" | "deterministic"
    hallucination_violations: List[str]  # empty if guard passed
    context_used: BriefingContext
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
