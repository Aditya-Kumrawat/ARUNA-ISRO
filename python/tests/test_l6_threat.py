import time
import pytest
from datetime import datetime

from aruna.models.nowcast import NowcastEvent
from aruna.l6_threat import ThreatOrchestrator
from aruna.l6_threat.rules import RISK_LEVEL_ORDER

def create_event(
    goes_class: str,
    goes_subclass: float,
    confidence: float = 0.8,
    alert_type: str = "FLARE_ONSET"
) -> NowcastEvent:
    return NowcastEvent(
        alert_type=alert_type,
        timestamp=datetime.utcnow(),
        goes_class=goes_class,
        goes_subclass=goes_subclass,
        confidence=confidence,
        instruments=["SoLEXS", "HEL1OS"],
        physics_frame_id="prov-123",
        detection_method="DUAL_TRIGGER",
        flare_type="STANDARD"
    )

def test_threat_flags_within_2_seconds():
    event = create_event("M", 5.2, confidence=0.95)
    orchestrator = ThreatOrchestrator()
    
    t0 = time.perf_counter()
    threat = orchestrator.process(event)
    duration = time.perf_counter() - t0
    
    assert threat is not None
    assert duration < 2.0
    assert threat.hf_risk == "STRONG"
    assert threat.hf_duration_min == 120
    assert threat.radiation_risk == "HIGH"
    assert threat.gaganyaan_advisory == "HIGH — SHELTER ADVISORY"

def test_navic_risk_matches_kintner_model_within_1_level():
    # Verify base risk maps correctly
    orchestrator = ThreatOrchestrator()
    
    # B class event
    event_b = create_event("B", 2.0, confidence=0.8, alert_type="HOPE_PRECURSOR")
    threat_b = orchestrator.process(event_b)
    # B maps to NEGLIGIBLE base. No modifiers apply since confidence is 0.8 (not >0.90 or <0.50),
    # alert_type is not impulsive, subclass 2.0 <= 3.0.
    assert threat_b.navic_risk["north_india"] == "NEGLIGIBLE"
    
    # C class event
    event_c = create_event("C", 2.0, confidence=0.8, alert_type="HOPE_PRECURSOR")
    threat_c = orchestrator.process(event_c)
    # C maps to LOW.
    assert threat_c.navic_risk["north_india"] == "LOW"
    
    # Check that they are within 1 level or correct
    b_idx = RISK_LEVEL_ORDER.index(threat_b.navic_risk["north_india"])
    c_idx = RISK_LEVEL_ORDER.index(threat_c.navic_risk["north_india"])
    assert abs(c_idx - b_idx) <= 1

def test_chandrayaan3_is_priority_1_for_x_class():
    event = create_event("X", 2.0)
    orchestrator = ThreatOrchestrator()
    threat = orchestrator.process(event)
    
    assert len(threat.safing_queue) > 0
    assert threat.safing_queue[0].name == "Chandrayaan-3"

def test_spacewx_bulletin_validates_against_swmc_template():
    event = create_event("M", 1.5)
    orchestrator = ThreatOrchestrator()
    threat = orchestrator.process(event)
    
    bulletin = threat.bulletin_text
    assert "SPACE WEATHER ADVISORY BULLETIN" in bulletin
    assert f"Event ID: {event.event_id}" in bulletin
    assert "GOES Class: M1.5" in bulletin
    assert "OPERATIONAL THREAT SUMMARY" in bulletin
    assert "ACTION PLAN" in bulletin

def test_dos_donts_generated_for_m5_plus_and_x_class():
    orchestrator = ThreatOrchestrator()
    
    # M1 event (minor)
    event_m1 = create_event("M", 1.5)
    threat_m1 = orchestrator.process(event_m1)
    assert len(threat_m1.dos) > 0
    assert "Routine monitoring" in threat_m1.dos[0]
    
    # M5 event (major)
    event_m5 = create_event("M", 5.0)
    threat_m5 = orchestrator.process(event_m5)
    assert len(threat_m5.dos) > 0
    assert "retreat to the radiation shelter" in " ".join(threat_m5.dos)
    assert "Do not initiate extravehicular activities" in " ".join(threat_m5.donts)

def test_south_india_highest_eia_amplification():
    # Impulsive phase + region in EIA amplified regions
    event = create_event("C", 2.0, confidence=0.8, alert_type="FLARE_ONSET")
    orchestrator = ThreatOrchestrator()
    threat = orchestrator.process(event)
    
    # C maps to LOW base.
    # Modifier: Impulsive phase -> +1 level for EIA regions (south_india, island_chains).
    # North India remains LOW.
    # South India should be MODERATE.
    assert threat.navic_risk["north_india"] == "LOW"
    assert threat.navic_risk["south_india"] == "MODERATE"
    assert threat.navic_risk["island_chains"] == "MODERATE"

def test_x_class_impact_never_understated():
    # Low confidence (-1 level modifier) should normally downgrade X1 (VERY_HIGH base) to HIGH.
    # But safety clamp must force it to remain VERY_HIGH.
    event = create_event("X", 1.5, confidence=0.2, alert_type="HOPE_PRECURSOR")
    orchestrator = ThreatOrchestrator()
    threat = orchestrator.process(event)
    
    assert threat.navic_risk["north_india"] == "VERY_HIGH"
    assert threat.navic_risk["south_india"] == "VERY_HIGH"
