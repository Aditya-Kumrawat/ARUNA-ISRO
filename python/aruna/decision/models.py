from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from aruna.models.frame_header import FrameHeader

RECOMMENDATIONS = frozenset({
    "PROCEED", "INCREASE_MONITORING", "NOTIFY_GROUND",
    "DELAY", "SAFE_MODE", "ABORT_EVA",
})


@dataclass
class ExplainabilityTrace:
    physics_evidence: List[str]
    trigger_chain: List[str]
    confidence_contributors: Dict[str, float]
    analog_match: str
    physics_consistency: str
    mission_impact_summary: str


@dataclass
class DecisionFrame:
    header: FrameHeader
    recommendation: str
    confidence: float
    actions: List[str]
    affected_assets: List[str]
    risk_timeline: List[Dict[str, Any]]  # [{minutes: int, risk: str}]
    mission_brief: str
    reasoning: ExplainabilityTrace
    valid_until: str                   # ISO 8601 UTC string
    schema_version: str = "1.0"
    pipeline_version: str = "2.0.0"
