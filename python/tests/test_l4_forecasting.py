"""L4 — Forecasting Engine Tests. PRD §8.10 + brainstorm."""
import pytest


@pytest.mark.xfail(reason="Phase 5 not yet implemented", strict=False)
@pytest.mark.slow
def test_tss_over_0_65_on_m_x_validation_2024_h2():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 5 not yet implemented", strict=False)
@pytest.mark.slow
def test_roc_auc_over_0_92_on_binary_mx_held_out():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 5 not yet implemented", strict=False)
def test_ece_under_0_05_after_mapie_calibration():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 5 not yet implemented", strict=False)
def test_lead_time_mae_under_90s_on_hope_events():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 5 not yet implemented", strict=False)
@pytest.mark.slow
def test_ensemble_beats_best_single_model_by_tss_0_03():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 5 not yet implemented", strict=False)
def test_no_temporal_leakage():
    """No training event post-dates any validation event by < 24 hours."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 5 not yet implemented", strict=False)
def test_forecast_output_within_10_seconds():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 5 not yet implemented", strict=False)
def test_under_prediction_under_5_pct_for_m5_plus():
    """Asymmetric loss validation: under-prediction rate < 5% for M5+ events."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 5 not yet implemented", strict=False)
def test_all_forecasts_include_confidence_interval():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 5 not yet implemented", strict=False)
def test_moirai_context_clip_fallback_bumps_tcn_weight():
    """When Moirai context is clipped to 512, TCN weight auto-adjusts to 0.7."""
    pytest.fail("not implemented")
