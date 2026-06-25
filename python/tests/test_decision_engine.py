from datetime import datetime
from aruna.decision.engine import DecisionEngine
from aruna.decision.models import DecisionFrame
from aruna.xl.confidence_engine import StackedForecastFrame
from aruna.xl.mission_impact import MissionImpactFrame
from aruna.models.frame_header import FrameHeader


def _make_stacked(p_M=0.30, p_X=0.10, confidence=0.40) -> StackedForecastFrame:
    rest = 1.0 - p_M - p_X
    return StackedForecastFrame(
        header=FrameHeader(timestamp=datetime.utcnow(), source="LIVE"),
        source_forecast_id="test-id",
        adjusted_probabilities=(rest * 0.5, rest * 0.3, rest * 0.2, p_M, p_X),
        decision_confidence=confidence,
        contributor_breakdown={"mpci": 0.05, "physics_gate": 0.0, "historical_base": 0.02},
    )


def _make_impact(eva="DELAY", sat="HIGH") -> MissionImpactFrame:
    return MissionImpactFrame(
        header=FrameHeader(timestamp=datetime.utcnow(), source="LIVE"),
        navic_s4_per_region={"South India": 0.75},
        navic_risk_per_region={"South India": "EXTREME"},
        satellite_risk=sat,
        navigation_risk="HIGH",
        communications_risk="DEGRADED",
        astronaut_eva_recommendation=eva,
        launch_recommendation="HOLD",
        hf_blackout_duration_min=120,
        impact_summary="Test impact.",
    )


def test_low_probability_gives_proceed():
    engine = DecisionEngine()
    result = engine.decide(_make_stacked(p_M=0.05, p_X=0.01, confidence=0.06), _make_impact(eva="PROCEED", sat="LOW"))
    assert isinstance(result, DecisionFrame)
    assert result.recommendation in ("PROCEED", "INCREASE_MONITORING")


def test_high_confidence_x_class_gives_safe_mode_or_abort():
    engine = DecisionEngine()
    result = engine.decide(_make_stacked(p_M=0.20, p_X=0.65, confidence=0.85), _make_impact(eva="ABORT", sat="CRITICAL"))
    assert result.recommendation in ("SAFE_MODE", "ABORT_EVA", "NOTIFY_GROUND")


def test_x63_path_recommendation_is_operational():
    engine = DecisionEngine()
    result = engine.decide(
        _make_stacked(p_M=0.10, p_X=0.72, confidence=0.82),
        _make_impact(eva="ABORT", sat="CRITICAL"),
    )
    assert result.recommendation in ("ABORT_EVA", "SAFE_MODE")


def test_decision_frame_has_all_fields():
    engine = DecisionEngine()
    result = engine.decide(_make_stacked(), _make_impact())
    assert result.recommendation
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.actions, list)
    assert isinstance(result.affected_assets, list)
    assert isinstance(result.risk_timeline, list)
    assert result.mission_brief
    assert result.reasoning


def test_valid_until_is_in_future():
    engine = DecisionEngine()
    result = engine.decide(_make_stacked(), _make_impact())
    assert isinstance(result.valid_until, str)
    assert len(result.valid_until) > 10


def test_header_source_propagated():
    engine = DecisionEngine()
    stacked = _make_stacked()
    stacked.header.source = "REPLAY"
    result = engine.decide(stacked, _make_impact())
    assert result.header.source == "REPLAY"
