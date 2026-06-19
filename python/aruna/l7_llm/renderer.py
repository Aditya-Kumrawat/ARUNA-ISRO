from __future__ import annotations
from aruna.l7_llm.models import BriefingJSON, BriefingContext


def render(briefing: BriefingJSON, ctx: BriefingContext) -> str:
    """Converts a validated BriefingJSON into a human-readable operations bulletin."""
    regions = ", ".join(
        f"{r.replace('_', ' ').title()}: {v}"
        for r, v in ctx.threat.navic_risk.items()
    )
    analog_line = (
        f"Closest historical analog: {briefing.analog_class} "
        f"({briefing.analog_similarity_pct:.0f}% similarity)"
        if briefing.analog_class != "NONE"
        else "No historical analog found."
    )

    return (
        f"ARUNA SOLAR FLARE BRIEFING\n"
        f"{'='*60}\n"
        f"Class: {briefing.goes_class} | Confidence: {briefing.confidence_pct:.0f}% | "
        f"Lead Time: {briefing.lead_time_min:.0f} min\n\n"
        f"SUMMARY\n{briefing.summary}\n\n"
        f"IMPACT ASSESSMENT\n"
        f"NavIC: {regions}\n"
        f"HF Comms: {briefing.hf_risk} (~{briefing.hf_duration_min} min)\n"
        f"Radiation: {briefing.radiation_risk}\n"
        f"Gaganyaan: {briefing.gaganyaan_advisory}\n\n"
        f"RECOMMENDED ACTION\n{briefing.recommended_action}\n\n"
        f"OPERATOR NOTE\n{briefing.operator_note}\n\n"
        f"ANALOG\n{analog_line}\n"
        f"{'='*60}"
    )
