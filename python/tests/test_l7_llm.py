"""L7 — LLM Briefing Tests. PRD §11.6 + brainstorm."""
import pytest


@pytest.mark.xfail(reason="Phase 8 not yet implemented", strict=False)
def test_hallucination_guard_passes_on_all_test_alerts():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 8 not yet implemented", strict=False)
def test_briefing_generated_within_15_seconds():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 8 not yet implemented", strict=False)
def test_fallback_activates_when_guard_fails():
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 8 not yet implemented", strict=False)
def test_every_briefing_has_required_fields():
    """Every briefing includes: flare class, confidence, lead time, NavIC risk."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 8 not yet implemented", strict=False)
def test_no_novel_numbers_in_briefing():
    """Briefing introduces no numbers not present in the context object."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 8 not yet implemented", strict=False)
def test_llm_runs_fully_offline():
    """No network requests made during LLM briefing generation (air-gapped test)."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 8 not yet implemented", strict=False)
def test_deterministic_fallback_valid_without_ollama():
    """Deterministic template produces valid formatted output even if Ollama is offline."""
    pytest.fail("not implemented")
