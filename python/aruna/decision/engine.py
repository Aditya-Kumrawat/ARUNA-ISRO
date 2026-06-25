from __future__ import annotations

from datetime import datetime, timedelta

from aruna.decision.models import DecisionFrame, ExplainabilityTrace
from aruna.models.frame_header import FrameHeader
from aruna.xl.confidence_engine import StackedForecastFrame
from aruna.xl.mission_impact import MissionImpactFrame


# Decision rules: (confidence_threshold, eva_triggers, output_recommendation)
# Evaluated top-to-bottom, first match wins.
_RULES = [
    (0.80, {"ABORT"},          "ABORT_EVA"),
    (0.60, {"ABORT", "DELAY"}, "SAFE_MODE"),
    (0.30, {"DELAY", "CAUTION"}, "NOTIFY_GROUND"),
    (0.10, set(),              "INCREASE_MONITORING"),
    (0.00, set(),              "PROCEED"),
]

_VALID_DURATION_SEC = 300  # decision valid for 5 minutes


class DecisionEngine:
    def decide(
        self,
        forecast: StackedForecastFrame,
        impact: MissionImpactFrame,
    ) -> DecisionFrame:
        confidence = forecast.decision_confidence
        eva = impact.astronaut_eva_recommendation

        recommendation = "PROCEED"
        for threshold, eva_triggers, rec in _RULES:
            if confidence >= threshold or eva in eva_triggers:
                recommendation = rec
                break

        actions = self._build_actions(recommendation, impact)
        affected = self._build_assets(impact)
        risk_timeline = self._build_risk_timeline(confidence)

        brief = (
            f"{recommendation.replace('_', ' ')} — "
            f"forecast confidence {confidence:.0%}, "
            f"{impact.impact_summary}"
        )

        reasoning = ExplainabilityTrace(
            physics_evidence=[
                f"P(M+X) adjusted to {forecast.adjusted_probabilities[3] + forecast.adjusted_probabilities[4]:.1%}",
                f"Decision confidence: {confidence:.1%}",
            ],
            trigger_chain=[recommendation],
            confidence_contributors=forecast.contributor_breakdown,
            analog_match="See Forecast Center for top analog match.",
            physics_consistency=f"Satellite risk: {impact.satellite_risk}",
            mission_impact_summary=impact.impact_summary,
        )

        valid_until = (datetime.utcnow() + timedelta(seconds=_VALID_DURATION_SEC)).isoformat() + "Z"

        return DecisionFrame(
            header=FrameHeader(
                timestamp=datetime.utcnow(),
                source=forecast.header.source,
            ),
            recommendation=recommendation,
            confidence=confidence,
            actions=actions,
            affected_assets=affected,
            risk_timeline=risk_timeline,
            mission_brief=brief,
            reasoning=reasoning,
            valid_until=valid_until,
        )

    def _build_actions(self, rec: str, impact: MissionImpactFrame) -> list[str]:
        base = {
            "PROCEED": ["Continue nominal operations", "Monitor SoLEXS/HEL1OS"],
            "INCREASE_MONITORING": ["Increase check-in frequency to 5 min", "Brief flight director"],
            "NOTIFY_GROUND": ["Notify ISTRAC", "Alert NavIC operations", "Standby satellite safing"],
            "DELAY": ["Delay EVA by 30 min minimum", "Monitor flare decay phase"],
            "SAFE_MODE": ["Initiate satellite safing sequence", "Suspend NavIC-critical ops", "Notify ISTRAC"],
            "ABORT_EVA": ["ABORT EVA IMMEDIATELY", "Crew return to station", "Close all EVA hatches"],
        }
        return base.get(rec, ["Monitor situation"])

    def _build_assets(self, impact: MissionImpactFrame) -> list[str]:
        assets = []
        if impact.satellite_risk in ("HIGH", "CRITICAL"):
            assets.append(f"Satellites: {impact.satellite_risk} risk")
        if impact.navigation_risk in ("HIGH", "VERY_HIGH", "EXTREME"):
            assets.append(f"NavIC: {impact.navigation_risk} — worst in South India")
        if impact.communications_risk != "NOMINAL":
            assets.append(f"HF Comms: {impact.communications_risk} for ~{impact.hf_blackout_duration_min} min")
        if impact.astronaut_eva_recommendation in ("DELAY", "ABORT"):
            assets.append(f"Gaganyaan EVA: {impact.astronaut_eva_recommendation}")
        return assets

    def _build_risk_timeline(self, current_confidence: float) -> list[dict]:
        return [
            {"minutes": 0,  "risk": f"{current_confidence:.0%}"},
            {"minutes": 15, "risk": f"{max(0, current_confidence - 0.10):.0%}"},
            {"minutes": 30, "risk": f"{max(0, current_confidence - 0.20):.0%}"},
            {"minutes": 60, "risk": f"{max(0, current_confidence - 0.35):.0%}"},
        ]
