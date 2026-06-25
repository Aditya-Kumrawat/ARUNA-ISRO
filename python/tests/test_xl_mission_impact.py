from aruna.xl.mission_impact import MissionImpactLayer, MissionImpactFrame


def test_x63_gives_extreme_south_india():
    layer = MissionImpactLayer()
    result = layer.compute(goes_class="X6.3", peak_flux_wm2=6.3e-4)
    assert isinstance(result, MissionImpactFrame)
    assert result.navic_risk_per_region["South India"] == "EXTREME"
    assert result.navic_s4_per_region["South India"] > 0.7


def test_x_class_eva_recommendation_is_abort_or_delay():
    layer = MissionImpactLayer()
    result = layer.compute(goes_class="X2.0", peak_flux_wm2=2.0e-4)
    assert result.astronaut_eva_recommendation in ("ABORT", "DELAY")


def test_m_class_eva_recommendation_is_delay_or_caution():
    layer = MissionImpactLayer()
    result = layer.compute(goes_class="M5.0", peak_flux_wm2=5.0e-5)
    assert result.astronaut_eva_recommendation in ("DELAY", "CAUTION")


def test_b_class_proceed_is_safe():
    layer = MissionImpactLayer()
    result = layer.compute(goes_class="B2.0", peak_flux_wm2=2.0e-7)
    assert result.astronaut_eva_recommendation == "PROCEED"
    assert result.launch_recommendation == "PROCEED"


def test_satellite_risk_scales_with_class():
    layer = MissionImpactLayer()
    b = layer.compute("B1.0", 1.0e-7)
    x = layer.compute("X5.0", 5.0e-4)
    risk_order = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert risk_order.index(x.satellite_risk) > risk_order.index(b.satellite_risk)


def test_impact_summary_is_nonempty_string():
    layer = MissionImpactLayer()
    result = layer.compute("M3.0", 3.0e-5)
    assert isinstance(result.impact_summary, str)
    assert len(result.impact_summary) > 10


def test_header_source_set():
    layer = MissionImpactLayer()
    result = layer.compute("M1.0", 1.0e-5, source="REPLAY")
    assert result.header.source == "REPLAY"
