from datetime import datetime
from aruna.models.physics import PhysicsFrame
from aruna.models.observation import QualityFlag
from aruna.xl.integrity_monitor import IntegrityMonitor, IntegrityFlag


def _make_frame(**kwargs) -> PhysicsFrame:
    defaults = dict(
        timestamp=datetime.utcnow(),
        source_frame_ids=["f1"],
        T_MK=12.0, T_uncertainty_MK=0.5,
        EM_cm3=3.5e49, EM_uncertainty=0.1e49,
        fit_chi_squared=1.2,
        DNI=1.1, tnpsv_magnitude=0.04, tnpsv_angle_rad=1.5,
        SHEI=-0.9, dSHEI_dt=0.0, HFD=1.3, FeKa_flux=6.5,
        QPP_period_sec=None, QPP_amplitude=None,
        dT_dt=0.5, dEM_dt=0.01, d2T_dt2=0.0,
        solexs_3_6kev=2.1e-6, solexs_6_10kev=1.5e-6,
        hel1os_10_25kev=4.2e-7, hel1os_25_50kev=2.1e-7, hel1os_50_70kev=1.0e-7,
        flare_phase="HOPE", hope_flag=True, hope_lead_time_sec=1200,
        hope_reason="T rising",
        quality_flag=QualityFlag.GOOD,
    )
    defaults.update(kwargs)
    return PhysicsFrame(**defaults)


def test_impossible_temperature_triggers_critical_flag():
    monitor = IntegrityMonitor()
    frame = _make_frame(T_MK=500.0)   # physically impossible
    flags = monitor.observe(frame)
    assert any(f.severity == "CRITICAL" for f in flags)
    critical = next(f for f in flags if f.severity == "CRITICAL")
    assert "temperature" in critical.explanation.lower()


def test_normal_frame_has_no_critical_flags():
    monitor = IntegrityMonitor()
    frame = _make_frame()
    flags = monitor.observe(frame)
    assert not any(f.severity == "CRITICAL" for f in flags)


def test_chi_squared_zero_does_not_crash():
    monitor = IntegrityMonitor()
    # chi²=0 is not valid (post_init raises), but 0.05 (below 0.1) should give INFO
    frame = _make_frame(fit_chi_squared=0.05)
    flags = monitor.observe(frame)
    # Just confirm no crash
    assert isinstance(flags, list)


def test_hope_in_quiet_phase_triggers_warning():
    monitor = IntegrityMonitor()
    frame = _make_frame(hope_flag=True, flare_phase="QUIET")
    flags = monitor.observe(frame)
    assert any(f.severity == "WARNING" for f in flags)
    w = next(f for f in flags if f.severity == "WARNING")
    assert "hope" in w.explanation.lower() or "phase" in w.explanation.lower()


def test_flag_has_all_required_fields():
    monitor = IntegrityMonitor()
    frame = _make_frame(T_MK=500.0)
    flags = monitor.observe(frame)
    f = flags[0]
    assert f.anomaly_type
    assert f.severity in ("INFO", "WARNING", "CRITICAL")
    assert f.affected_stage
    assert f.explanation
    assert f.recommended_action


def test_monitor_never_modifies_input_frame():
    monitor = IntegrityMonitor()
    frame = _make_frame(T_MK=500.0)
    original_T = frame.T_MK
    monitor.observe(frame)
    assert frame.T_MK == original_T   # immutability guarantee
