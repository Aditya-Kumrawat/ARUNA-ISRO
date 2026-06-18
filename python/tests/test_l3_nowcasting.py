"""L3 — Nowcasting Engine Tests. PRD §7.11 + brainstorm."""
import pytest


@pytest.mark.xfail(reason="Phase 4 not yet implemented", strict=False)
@pytest.mark.physics
def test_x63_detected_within_30_seconds_of_goes_onset():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 4 not yet implemented", strict=False)
@pytest.mark.physics
def test_hope_fires_90s_before_x63_peak():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 4 not yet implemented", strict=False)
def test_tpr_over_0_80_on_m_x_validation_set():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 4 not yet implemented", strict=False)
def test_far_under_0_05_on_100h_quiet_sun():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 4 not yet implemented", strict=False)
def test_tss_over_0_65_on_m_x_events():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 4 not yet implemented", strict=False)
def test_solexs_only_trigger_does_not_fire_confirmed_onset():
    """Single-instrument trigger must NOT generate a confirmed FLARE_ONSET alert."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 4 not yet implemented", strict=False)
def test_overlapping_flares_have_distinct_event_ids():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 4 not yet implemented", strict=False)
def test_catalog_write_under_100ms():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 4 not yet implemented", strict=False)
def test_all_detections_have_provenance_hash():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 4 not yet implemented", strict=False)
def test_x3_plus_escalated_to_critical():
    """X3+ events → CRITICAL alert regardless of confidence score."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 4 not yet implemented", strict=False)
def test_sep_event_classified_correctly_during_high_kp():
    """Kp > 6 + no SoLEXS precursor → SEP_EVENT, not FLARE."""
    pytest.fail("not implemented")
