from __future__ import annotations
from aruna.l7_llm.models import BriefingJSON, BriefingContext


def validate(briefing: BriefingJSON, ctx: BriefingContext) -> tuple[bool, list[str]]:
    """
    Validates structured LLM output field-by-field against BriefingContext.
    Returns (is_valid, violations).
    """
    violations: list[str] = []

    # goes_class must match ctx.nowcast.goes_label exactly
    if briefing.goes_class != ctx.nowcast.goes_label:
        violations.append(
            f"goes_class mismatch: got '{briefing.goes_class}', expected '{ctx.nowcast.goes_label}'"
        )

    # confidence_pct within ±2 of ctx.nowcast.confidence * 100
    expected_conf = round(ctx.nowcast.confidence * 100, 1)
    if abs(briefing.confidence_pct - expected_conf) > 2.0:
        violations.append(
            f"confidence_pct drift: got {briefing.confidence_pct}, expected {expected_conf} (±2)"
        )

    # lead_time_min within ±2 min of ctx.forecast.lead_time_estimate_sec / 60
    expected_lt = ctx.forecast.lead_time_estimate_sec / 60
    if abs(briefing.lead_time_min - expected_lt) > 2.0:
        violations.append(
            f"lead_time_min drift: got {briefing.lead_time_min}, expected {expected_lt:.1f} (±2)"
        )

    # hf_duration_min must match exactly
    if briefing.hf_duration_min != ctx.threat.hf_duration_min:
        violations.append(
            f"hf_duration_min mismatch: got {briefing.hf_duration_min}, expected {ctx.threat.hf_duration_min}"
        )

    # navic_risk_summary must be a value present in ctx.threat.navic_risk
    allowed_risks = set(ctx.threat.navic_risk.values())
    if briefing.navic_risk_summary not in allowed_risks:
        violations.append(
            f"navic_risk_summary '{briefing.navic_risk_summary}' not in allowed values {allowed_risks}"
        )

    return len(violations) == 0, violations
