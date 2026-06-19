from __future__ import annotations
from aruna.l7_llm.models import BriefingContext


def format_deterministic(ctx: BriefingContext) -> str:
    """
    Deterministic fallback — no LLM required, always produces valid output.
    Used when LLM is unavailable or guard detects a violation.
    """
    nowcast = ctx.nowcast
    forecast = ctx.forecast
    threat = ctx.threat
    analogs = ctx.analogs

    top = analogs[0] if analogs else None
    analog_line = (
        f"Closest historical analog: {top.goes_class} ({top.similarity*100:.0f}% similarity)"
        if (top and top.event_id != "NO_ANALOG")
        else "No historical analog found."
    )

    regions = "\n".join(
        f"  {r.replace('_', ' ').title()}: {v} (p={ctx.threat.navic_prob.get(r, 0.0):.2f})"
        for r, v in threat.navic_risk.items()
    )

    dos = "\n".join(f"  - {d}" for d in threat.dos) or "  - Continue routine monitoring."
    donts = "\n".join(f"  - {d}" for d in threat.donts) or "  - No immediate restrictions."

    lead_min = round(forecast.lead_time_estimate_sec / 60, 1)

    return (
        f"ARUNA SOLAR FLARE ALERT [DETERMINISTIC]\n"
        f"{'='*60}\n"
        f"Class: {nowcast.goes_label} | Confidence: {nowcast.confidence*100:.0f}% | "
        f"Lead Time: {lead_min:.0f} min\n\n"
        f"NAVIC REGIONAL IMPACT\n{regions}\n\n"
        f"HF COMMS: {threat.hf_risk} (~{threat.hf_duration_min} min)\n"
        f"RADIATION: {threat.radiation_risk}\n"
        f"GAGANYAAN: {threat.gaganyaan_advisory}\n\n"
        f"DO:\n{dos}\n\n"
        f"AVOID:\n{donts}\n\n"
        f"ANALOG\n{analog_line}\n"
        f"{'='*60}"
    )
