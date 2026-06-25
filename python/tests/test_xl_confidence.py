from datetime import datetime
from aruna.models.forecast import ForecastFrame
from aruna.xl.mpci import MPCIResult
from aruna.xl.confidence_engine import ConfidenceEngine, StackedForecastFrame


def _make_forecast(prob_M=0.30, prob_X=0.10) -> ForecastFrame:
    total_other = 1.0 - prob_M - prob_X
    return ForecastFrame(
        timestamp=datetime.utcnow(),
        horizon_minutes=10,
        prob_quiet=total_other * 0.5,
        prob_B=total_other * 0.3,
        prob_C=total_other * 0.2,
        prob_M=prob_M,
        prob_X=prob_X,
        predicted_class="M",
        peak_class_prob=prob_M,
        confidence_interval_low=0.20,
        confidence_interval_high=0.50,
        prediction_set=["M"],
        calibration_ece=0.03,
        lead_time_estimate_sec=1200,
        lead_time_uncertainty_sec=120,
        tcn_weight=1.0,
        moirai_weight=0.0,
        hope_flag_active=True,
        physics_state="HOPE",
        input_window_start=datetime.utcnow(),
        input_window_end=datetime.utcnow(),
        model_version="tcn_v1",
    )


def _make_mpci(mpci_score=0.91) -> MPCIResult:
    modifier = 0.20 if mpci_score > 0.80 else (-0.15 if mpci_score < 0.50 else 0.0)
    return MPCIResult(
        timestamp=datetime.utcnow(),
        mpci=mpci_score,
        suit_score=0.85,
        solexs_score=0.92,
        hel1os_score=0.95,
        velc_score=None,
        available_payloads=["SUIT", "SOLEXS", "HEL1OS"],
        confidence_modifier=modifier,
    )


def test_mpci_boost_increases_confidence():
    engine = ConfidenceEngine()
    forecast = _make_forecast(prob_M=0.30, prob_X=0.10)
    mpci = _make_mpci(mpci_score=0.91)   # → +0.20 modifier
    result = engine.compute(forecast, mpci)
    assert isinstance(result, StackedForecastFrame)
    assert result.decision_confidence > forecast.prob_M + forecast.prob_X
    assert "mpci" in result.contributor_breakdown
    assert result.contributor_breakdown["mpci"] > 0


def test_mpci_penalty_decreases_confidence():
    engine = ConfidenceEngine()
    forecast = _make_forecast(prob_M=0.30, prob_X=0.10)
    mpci = _make_mpci(mpci_score=0.40)   # → -0.15 modifier
    result = engine.compute(forecast, mpci)
    assert result.decision_confidence < forecast.prob_M + forecast.prob_X
    assert result.contributor_breakdown["mpci"] < 0


def test_adjusted_probabilities_sum_to_one():
    engine = ConfidenceEngine()
    result = engine.compute(_make_forecast(), _make_mpci())
    total = sum(result.adjusted_probabilities)
    assert abs(total - 1.0) < 1e-6


def test_contributor_breakdown_has_all_contributors():
    engine = ConfidenceEngine()
    result = engine.compute(_make_forecast(), _make_mpci())
    assert "mpci" in result.contributor_breakdown
    assert "physics_gate" in result.contributor_breakdown
    assert "historical_base" in result.contributor_breakdown


def test_header_source_preserved():
    engine = ConfidenceEngine()
    result = engine.compute(_make_forecast(), _make_mpci(), source="REPLAY")
    assert result.header.source == "REPLAY"
