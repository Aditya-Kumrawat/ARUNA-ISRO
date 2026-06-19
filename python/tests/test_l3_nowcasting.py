import pytest
import numpy as np
from datetime import datetime, timezone, timedelta


def _make_physics_frame(
    timestamp=None,
    hope_flag=False,
    hope_reason=None,
    hope_lead_time_sec=None,
    T_MK=12.0,
    EM_cm3=1e48,
    DNI=0.1,
    SHEI=-1.5,
    dSHEI_dt=0.0,
    provenance_hash="testhash",
    frame_id=None,
    solexs_flux=500.0,
    hel_flux=100.0,
):
    import uuid
    from aruna.l2_physics.models import PhysicsFrame, PhysicsQuality
    from aruna.models.observation import QualityFlag
    sol = solexs_flux
    hel = hel_flux
    return PhysicsFrame(
        frame_id=frame_id or str(uuid.uuid4()),
        timestamp=timestamp or datetime.now(timezone.utc),
        source_frame_ids=["sol1", "hel1"],
        T_MK=T_MK, T_uncertainty_MK=1.0,
        EM_cm3=EM_cm3, EM_uncertainty=1e46,
        fit_chi_squared=1.2,
        DNI=DNI,
        tnpsv_magnitude=0.1, tnpsv_angle_rad=1.57,
        SHEI=SHEI, dSHEI_dt=dSHEI_dt,
        HFD=float("nan"),
        QPP_period_sec=None, QPP_amplitude=None,
        FeKa_flux=0.0,
        flare_phase="HOPE" if hope_flag else "QUIET",
        hope_flag=hope_flag,
        hope_lead_time_sec=hope_lead_time_sec,
        hope_reason=hope_reason,
        engine_state="ACTIVE",
        data_quality_flag=QualityFlag.GOOD,
        physics_quality_flag=PhysicsQuality.FIT_OK,
        fit_backend="chianti_lookup",
        pipeline_version="2.0.0",
        provenance_hash=provenance_hash,
        solexs_3_6kev=sol * 0.6,
        solexs_6_10kev=sol * 0.4,
        hel1os_10_25kev=hel * 0.5,
        hel1os_25_50kev=hel * 0.3,
        hel1os_50_70kev=hel * 0.2,
    )


class TestNowcastConfig:
    def test_defaults_match_prd(self):
        from aruna.l3_nowcasting.models import NowcastConfig
        cfg = NowcastConfig()
        assert cfg.baseline_window_sec == 10800
        assert cfg.baseline_percentile == 85.0
        assert cfg.baseline_recompute_interval == 60
        assert cfg.trigger_sigma_multiplier == 3.0
        assert cfg.dual_trigger_window_sec == 180
        assert cfg.pelt_penalty == pytest.approx(1.5)
        assert cfg.pelt_min_size == 10
        assert cfg.solexs_to_wm2_factor == pytest.approx(1.0e-8)
        assert cfg.x3_auto_critical is True
        assert cfg.peak_decay_threshold == pytest.approx(0.5)

    def test_overrides_work(self):
        from aruna.l3_nowcasting.models import NowcastConfig
        cfg = NowcastConfig(baseline_window_sec=3600, trigger_sigma_multiplier=4.0)
        assert cfg.baseline_window_sec == 3600
        assert cfg.trigger_sigma_multiplier == pytest.approx(4.0)
