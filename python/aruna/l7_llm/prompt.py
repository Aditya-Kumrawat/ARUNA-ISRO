from __future__ import annotations
import json
from aruna.l7_llm.models import BriefingContext, BriefingJSON

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are ARUNA's operational briefing module for ISRO mission operations.

RULES (non-negotiable):
- Use ONLY the values present in the CONTEXT JSON below.
- Do NOT estimate, infer, calculate, or invent any scientific values.
- If information is unavailable, write "unavailable".
- Output ONLY valid JSON matching the OUTPUT SCHEMA exactly.
- summary: max 3 sentences. recommended_action: max 2 sentences. operator_note: max 1 sentence.
"""

OUTPUT_SCHEMA = {k: "..." for k in BriefingJSON.__dataclass_fields__}


class PromptBuilder:
    version: str = PROMPT_VERSION

    def build(self, ctx: BriefingContext) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt) assembled from BriefingContext."""
        analogs = ctx.analogs
        top = analogs[0] if analogs else None
        analog_class = top.goes_class if (top and top.event_id != "NO_ANALOG") else "NONE"
        analog_sim = round(top.similarity * 100, 1) if (top and top.event_id != "NO_ANALOG") else 0.0

        context_dict = {
            "goes_class": ctx.nowcast.goes_label,
            "confidence_pct": round(ctx.nowcast.confidence * 100, 1),
            "lead_time_min": round(ctx.forecast.lead_time_estimate_sec / 60, 1),
            "navic_risk": ctx.threat.navic_risk,
            "hf_risk": ctx.threat.hf_risk,
            "hf_duration_min": ctx.threat.hf_duration_min,
            "radiation_risk": ctx.threat.radiation_risk,
            "gaganyaan_advisory": ctx.threat.gaganyaan_advisory,
            "dos": ctx.threat.dos,
            "donts": ctx.threat.donts,
            "analog_class": analog_class,
            "analog_similarity_pct": analog_sim,
        }

        user_prompt = (
            f"CONTEXT:\n{json.dumps(context_dict, indent=2)}\n\n"
            f"OUTPUT SCHEMA:\n{json.dumps(OUTPUT_SCHEMA, indent=2)}"
        )
        return SYSTEM_PROMPT, user_prompt
