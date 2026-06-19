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


class TestRollingP85Baseline:
    def test_none_before_first_recompute(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline(window_sec=100, recompute_interval=60)
        val, thr = b.update(1.0, t_sec=0)
        assert val is None and thr is None

    def test_p85_computed_after_interval(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline(window_sec=200, recompute_interval=60, sigma_multiplier=0.0)
        for i in range(200):
            b.update(float(i), t_sec=i)
        val, thr = b.update(200.0, t_sec=200)
        assert val is not None
        # Last recompute was at t=180 on window [0..180], so p85 of [0..180]
        expected = np.percentile(list(range(181)), 85)
        assert val == pytest.approx(expected, rel=1e-3)
        assert thr == pytest.approx(expected, rel=1e-3)  # sigma=0

    def test_threshold_above_baseline(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline(window_sec=200, recompute_interval=1, sigma_multiplier=3.0)
        rng = np.random.default_rng(0)
        for i in range(200):
            b.update(float(rng.normal(10.0, 1.0)), t_sec=i)
        assert b.threshold > b.baseline

    def test_freeze_holds_baseline_fixed(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline(window_sec=200, recompute_interval=1)
        for i in range(200):
            b.update(1.0, t_sec=i)
        b.freeze()
        baseline_before = b.baseline
        for i in range(200, 400):
            b.update(9999.0, t_sec=i)
        assert b.baseline == baseline_before

    def test_unfreeze_resumes_updates(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline(window_sec=200, recompute_interval=1)
        for i in range(200):
            b.update(1.0, t_sec=i)
        b.freeze()
        b.unfreeze()
        for i in range(200, 400):
            b.update(9999.0, t_sec=i)
        assert b.baseline > 100.0

    def test_is_frozen_flag(self):
        from aruna.l3_nowcasting.baseline import RollingP85Baseline
        b = RollingP85Baseline()
        assert not b.is_frozen()
        b.freeze()
        assert b.is_frozen()
        b.unfreeze()
        assert not b.is_frozen()


class TestChangepointDetector:
    def test_none_during_warmup(self):
        from aruna.l3_nowcasting.changepoint import ChangepointDetector
        d = ChangepointDetector(flux_window=100)
        result = None
        for _ in range(99):
            result = d.update(1.0)
        assert result is None

    def test_detects_step_change(self):
        from aruna.l3_nowcasting.changepoint import ChangepointDetector
        d = ChangepointDetector(flux_window=100, penalty=1.5, min_size=5)
        result = None
        for i in range(100):
            result = d.update(1.0 if i < 50 else 100.0)
        # step at index 50 — PELT should detect it
        assert result is not None
        assert 30 <= result <= 70

    def test_flat_signal_returns_none_or_int(self):
        from aruna.l3_nowcasting.changepoint import ChangepointDetector
        d = ChangepointDetector(flux_window=100, penalty=5.0, min_size=10)
        result = None
        rng = np.random.default_rng(0)
        for _ in range(100):
            result = d.update(float(rng.normal(1.0, 0.001)))
        assert result is None or isinstance(result, int)

    def test_returns_int_not_list(self):
        from aruna.l3_nowcasting.changepoint import ChangepointDetector
        d = ChangepointDetector(flux_window=100, penalty=1.5, min_size=5)
        result = None
        for i in range(100):
            result = d.update(1.0 if i < 50 else 100.0)
        if result is not None:
            assert isinstance(result, int)


class TestGOESClassifier:
    def _gc(self):
        from aruna.l3_nowcasting.goes_class import GOESClassifier
        return GOESClassifier(solexs_to_wm2_factor=1.0e-8)

    def test_x_class(self):
        counts = 1.25e-4 / 1.0e-8   # → 1.25e-4 W/m² → X1.2
        letter, sub = self._gc().classify(counts)
        assert letter == "X"
        assert sub == pytest.approx(1.2, rel=0.05)

    def test_m38_class(self):
        counts = 3.8e-5 / 1.0e-8    # M3.8
        letter, sub = self._gc().classify(counts)
        assert letter == "M"
        assert sub == pytest.approx(3.8, rel=0.01)

    def test_c_class(self):
        counts = 5.0e-6 / 1.0e-8    # C5.0
        letter, sub = self._gc().classify(counts)
        assert letter == "C"

    def test_b_class(self):
        counts = 5.0e-7 / 1.0e-8    # B5.0
        letter, sub = self._gc().classify(counts)
        assert letter == "B"

    def test_sub_a(self):
        letter, sub = self._gc().classify(0.001)
        assert letter == "A"

    def test_zero_flux(self):
        letter, sub = self._gc().classify(0.0)
        assert letter == "A"
        assert sub == 0.0

    def test_x3_is_major(self):
        gc = self._gc()
        assert gc.is_major_xclass("X", 3.0) is True
        assert gc.is_major_xclass("X", 2.9) is False
        assert gc.is_major_xclass("M", 9.0) is False
