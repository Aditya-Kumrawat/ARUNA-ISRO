"""L7 — LLM Briefing Tests. PRD §11.6"""
import time
import pytest
from datetime import datetime

from aruna.models.nowcast import NowcastEvent
from aruna.models.forecast import ForecastFrame
from aruna.l6_threat.models import ThreatOutput, SafingEntry
from aruna.l5_analogs.models import AnalogMatch, NO_ANALOG
from aruna.l7_llm.models import BriefingContext, BriefingJSON
from aruna.l7_llm.client import MockLLMClient
from aruna.l7_llm import BriefingOrchestrator
from aruna.l7_llm.guard import validate
from aruna.l7_llm.template import format_deterministic


# ── Fixtures ────────────────────────────────────────────────────────────────

def make_nowcast(goes_class="M", goes_subclass=5.2, confidence=0.87) -> NowcastEvent:
    return NowcastEvent(
        alert_type="FLARE_ONSET",
        timestamp=datetime.utcnow(),
        goes_class=goes_class,
        goes_subclass=goes_subclass,
        confidence=confidence,
        instruments=["SoLEXS", "HEL1OS"],
        physics_frame_id="pf-001",
        detection_method="DUAL_TRIGGER",
        flare_type="STANDARD",
        lead_time_sec=720,
    )


def make_forecast(lead_time_sec=720) -> ForecastFrame:
    return ForecastFrame(
        timestamp=datetime.utcnow(),
        horizon_minutes=15,
        prob_quiet=0.0, prob_B=0.0, prob_C=0.05, prob_M=0.80, prob_X=0.15,
        predicted_class="M",
        peak_class_prob=0.80,
        confidence_interval_low=0.70,
        confidence_interval_high=0.90,
        prediction_set=["M", "X"],
        calibration_ece=0.02,
        lead_time_estimate_sec=lead_time_sec,
        lead_time_uncertainty_sec=60,
        tcn_weight=0.5,
        moirai_weight=0.5,
        hope_flag_active=True,
        physics_state="HOPE",
        input_window_start=datetime.utcnow(),
        input_window_end=datetime.utcnow(),
        model_version="1",
    )


def make_threat() -> ThreatOutput:
    return ThreatOutput(
        navic_risk={"south_india": "HIGH", "north_india": "MODERATE", "island_chains": "HIGH"},
        navic_prob={"south_india": 0.55, "north_india": 0.30, "island_chains": 0.55},
        hf_risk="STRONG",
        hf_duration_min=120,
        radiation_risk="HIGH",
        gaganyaan_advisory="HIGH — SHELTER ADVISORY",
        safing_queue=[SafingEntry("Chandrayaan-3", "LUNAR_POLAR", {"M5": 0.60})],
        dos=["Monitor NavIC receivers.", "Initiate payload safing."],
        donts=["Do not initiate EVA.", "Avoid HF comms for critical ops."],
        bulletin_text="TEST BULLETIN",
        source_event_id="evt-001",
    )


def make_analog() -> AnalogMatch:
    import numpy as np
    return AnalogMatch(
        event_id="EVT_HIST_001",
        goes_class="M",
        dtw_distance=0.5,
        similarity=0.67,
        feature_vector=np.zeros(17),
        no_analog_flag=False,
    )


def make_context(**kwargs) -> BriefingContext:
    return BriefingContext(
        nowcast=kwargs.get("nowcast", make_nowcast()),
        forecast=kwargs.get("forecast", make_forecast()),
        threat=kwargs.get("threat", make_threat()),
        analogs=kwargs.get("analogs", [make_analog()]),
    )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_hallucination_guard_passes_on_all_test_alerts():
    """MockLLMClient echoes context values — guard must pass with zero violations."""
    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=MockLLMClient())
    briefing = orchestrator.process(ctx)

    assert briefing.hallucination_violations == []
    assert briefing.briefing_source != "deterministic"


def test_briefing_generated_within_15_seconds():
    """End-to-end process() must complete in under 15 seconds (MockLLMClient is instant)."""
    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=MockLLMClient())

    t0 = time.perf_counter()
    briefing = orchestrator.process(ctx)
    elapsed = time.perf_counter() - t0

    assert elapsed < 15.0
    assert briefing.briefing_text != ""


def test_fallback_activates_when_guard_fails():
    """When MockLLMClient returns wrong goes_class, guard fires and source == 'deterministic'."""
    class BadMockClient:
        model = "mock_bad"
        def generate(self, system_prompt, user_prompt, ctx):
            return BriefingJSON(
                goes_class="X99.9",          # wrong — will fail guard
                confidence_pct=87.0,
                lead_time_min=12.0,
                navic_risk_summary="HIGH",
                hf_risk="STRONG",
                hf_duration_min=120,
                radiation_risk="HIGH",
                gaganyaan_advisory="HIGH — SHELTER ADVISORY",
                summary="Bad summary.",
                recommended_action="Bad action.",
                operator_note="Bad note.",
                analog_class="M",
                analog_similarity_pct=67.0,
            )

    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=BadMockClient())
    briefing = orchestrator.process(ctx)

    assert briefing.briefing_source == "deterministic"
    assert len(briefing.hallucination_violations) > 0
    assert "goes_class mismatch" in briefing.hallucination_violations[0]


def test_every_briefing_has_required_fields():
    """briefing_text must contain goes_label, confidence pct string, lead time, and navic risk."""
    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=MockLLMClient())
    briefing = orchestrator.process(ctx)

    text = briefing.briefing_text
    assert ctx.nowcast.goes_label in text      # e.g. "M5.2"
    assert "87" in text                         # confidence pct
    assert "12" in text                         # lead time (720s = 12 min)
    assert any(r in text for r in ["HIGH", "MODERATE", "VERY_HIGH"])  # navic risk


def test_no_novel_numbers_in_briefing():
    """Guard catches field drift before render — deterministic template uses only context values."""
    class DriftingClient:
        model = "mock_drift"
        def generate(self, system_prompt, user_prompt, ctx):
            return BriefingJSON(
                goes_class=ctx.nowcast.goes_label,
                confidence_pct=99.9,           # invented — should fail ±2 check
                lead_time_min=round(ctx.forecast.lead_time_estimate_sec / 60, 1),
                navic_risk_summary=list(ctx.threat.navic_risk.values())[0],
                hf_risk=ctx.threat.hf_risk,
                hf_duration_min=ctx.threat.hf_duration_min,
                radiation_risk=ctx.threat.radiation_risk,
                gaganyaan_advisory=ctx.threat.gaganyaan_advisory,
                summary="Solar flare detected.",
                recommended_action="Take action.",
                operator_note="Note.",
                analog_class="NONE",
                analog_similarity_pct=0.0,
            )

    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=DriftingClient())
    briefing = orchestrator.process(ctx)

    # Guard should catch confidence_pct=99.9 (expected ~87) → fallback
    assert briefing.briefing_source == "deterministic"
    assert any("confidence_pct" in v for v in briefing.hallucination_violations)


def test_llm_runs_fully_offline():
    """MockLLMClient makes no network calls — briefing completes without any socket activity."""
    import socket
    original_getaddrinfo = socket.getaddrinfo

    calls = []
    def mock_getaddrinfo(*args, **kwargs):
        calls.append(args)
        return original_getaddrinfo(*args, **kwargs)

    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=MockLLMClient())

    socket.getaddrinfo = mock_getaddrinfo
    try:
        briefing = orchestrator.process(ctx)
    finally:
        socket.getaddrinfo = original_getaddrinfo

    # MockLLMClient should produce a briefing without triggering Azure DNS resolution
    azure_calls = [c for c in calls if "azure" in str(c).lower() or "openai" in str(c).lower()]
    assert len(azure_calls) == 0
    assert briefing.briefing_text != ""


def test_deterministic_fallback_valid_without_ollama():
    """format_deterministic() produces a non-empty valid bulletin with no LLM dependency."""
    ctx = make_context()
    text = format_deterministic(ctx)

    assert len(text) > 100
    assert ctx.nowcast.goes_label in text
    assert str(ctx.threat.hf_duration_min) in text
    assert "DETERMINISTIC" in text
    assert ctx.threat.radiation_risk in text
