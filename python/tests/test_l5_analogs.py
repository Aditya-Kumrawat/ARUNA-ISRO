"""L5 — Historical Analog Retrieval Tests. PRD §9.3 + brainstorm."""
import pytest


@pytest.mark.xfail(reason="Phase 6 not yet implemented", strict=False)
def test_retrieval_precision_over_0_75():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 6 not yet implemented", strict=False)
def test_faiss_search_under_100ms_for_10k_events():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 6 not yet implemented", strict=False)
def test_no_analog_fires_when_dtw_over_3_sigma():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 6 not yet implemented", strict=False)
@pytest.mark.physics
def test_x63_retrieves_at_least_one_x_class_analog():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 6 not yet implemented", strict=False)
def test_faiss_index_auto_rebuilds_after_new_catalog_entry():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 6 not yet implemented", strict=False)
def test_cold_start_empty_catalog_no_crash():
    """Cold-start with 0 events in catalog activates NO_ANALOG mode without crashing."""
    pytest.fail("not implemented")
