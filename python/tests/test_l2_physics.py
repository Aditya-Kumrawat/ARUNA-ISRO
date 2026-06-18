"""
L2 — Physics Engine Tests
PRD §6.12 acceptance criteria + brainstorm additions.
The X6.3 T/EM test is the single most important test in the entire suite.
"""

import math
from datetime import datetime, timezone

import pytest

from aruna.l2_physics.models import (
    PhysicsFrame, PhysicsQuality, EngineState, FitResult, HOPEResult, PhysicsConfig
)
from aruna.models.observation import QualityFlag


def _make_frame(**overrides):
    defaults = dict(
        frame_id="f1", timestamp=datetime.now(timezone.utc),
        source_frame_ids=["a", "b"],
        T_MK=15.0, T_uncertainty_MK=1.0, EM_cm3=1e48, EM_uncertainty=1e46,
        fit_chi_squared=1.2,
        DNI=0.5, tnpsv_magnitude=0.1, tnpsv_angle_rad=1.2,
        SHEI=-1.0, dSHEI_dt=0.01, HFD=float('nan'),
        QPP_period_sec=None, QPP_amplitude=None, FeKa_flux=0.0,
        flare_phase="QUIET", hope_flag=False,
        hope_lead_time_sec=None, hope_reason=None,
        engine_state="WARMUP",
        data_quality_flag=QualityFlag.GOOD,
        physics_quality_flag=PhysicsQuality.FIT_OK,
        fit_backend="chianti_lookup",
        pipeline_version="2.0.0",
    )
    defaults.update(overrides)
    return PhysicsFrame(**defaults)


class TestModels:
    def test_physics_quality_enum(self):
        assert PhysicsQuality.FIT_OK == 0
        assert PhysicsQuality.FIT_DIVERGED == 1
        assert PhysicsQuality.LOW_CONFIDENCE == 2
        assert PhysicsQuality.BACKEND_FALLBACK == 3

    def test_engine_state_enum(self):
        assert EngineState.WARMUP.value == "WARMUP"
        assert EngineState.ACTIVE.value == "ACTIVE"
        assert EngineState.DEGRADED.value == "DEGRADED"

    def test_fit_result(self):
        r = FitResult(T_MK=15.0, EM_cm3=1e48, T_uncertainty_MK=1.0,
                      EM_uncertainty=1e46, chi2_reduced=1.2, backend="chianti_lookup")
        assert r.T_MK == 15.0
        assert r.backend == "chianti_lookup"

    def test_hope_result(self):
        r = HOPEResult(hope_flag=True, reason="test", lead_time_sec=120)
        assert r.hope_flag is True

    def test_physics_frame_construction(self):
        f = _make_frame()
        assert f.flare_phase == "QUIET"
        assert f.schema_version == "1.0"
        assert f.pipeline_version == "2.0.0"
        assert math.isnan(f.HFD)

    def test_physics_config_defaults(self):
        cfg = PhysicsConfig()
        assert cfg.fit_energy_min_kev == 3.0
        assert cfg.hope_dni_threshold == 1.5
        assert cfg.hfd_window == 600


@pytest.mark.xfail(reason="Phase 3 not yet implemented", strict=False)
@pytest.mark.physics
def test_x63_temperature_matches_sarwade_within_1_sigma():
    """
    T_peak on X6.3 event (2024-02-22) ≈ 23.5 MK within 1σ instrument uncertainty.
    Reference: Sarwade et al. arXiv:2502.19795, Figure 3.
    THIS IS THE MOST CRITICAL TEST IN THE ENTIRE SUITE.
    """
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 3 not yet implemented", strict=False)
@pytest.mark.physics
def test_isothermal_fit_chi_squared_under_2_on_m_plus_spectra():
    """Isothermal fit χ² < 2.0 on > 90% of M+ class spectra."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 3 not yet implemented", strict=False)
@pytest.mark.physics
def test_dni_hope_threshold_fires_within_2_min_of_impulsive_onset():
    """DNI > 1.5 fires HOPE flag on X6.3 within 2 minutes of impulsive onset."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 3 not yet implemented", strict=False)
@pytest.mark.physics
def test_tnpsv_in_precursor_quadrant_90s_before_x63_peak():
    """TNPSV in precursor quadrant (60°–120°) at least 90 seconds before X6.3 peak."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 3 not yet implemented", strict=False)
@pytest.mark.physics
def test_shei_transitions_from_thermal_to_nonthermal_during_x63():
    """SHEI transitions from < -1.5 (thermal) to > -0.8 (nonthermal) during X6.3 impulsive phase."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 3 not yet implemented", strict=False)
def test_feka_fluorescence_detected_in_80_pct_of_x_class_events():
    """Fe Kα fluorescence detected (residual > 0) in ≥ 80% of X-class events."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 3 not yet implemented", strict=False)
def test_engine_runtime_under_1_second_per_window():
    """Physics engine computes one PhysicsFrame in < 1 second."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 3 not yet implemented", strict=False)
def test_all_physics_frames_have_provenance_hash():
    """Every PhysicsFrame.provenance_hash is non-empty and links to source ObservationFrame checksums."""
    pytest.fail("not implemented")


@pytest.mark.xfail(reason="Phase 3 not yet implemented", strict=False)
def test_no_div_zero_or_nan_propagation():
    """No NaN or infinity in any PhysicsFrame field under normal operating conditions."""
    pytest.fail("not implemented")


import numpy as np
from aruna.l2_physics.chianti_table import CHIANTILookupTable


class TestCHIANTITable:
    def _bins(self):
        return np.linspace(1.0, 25.0, 25)  # 24 bins

    def test_generate_creates_table(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "chianti_table.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        table = CHIANTILookupTable()
        table.generate(self._bins())
        assert (tmp_path / "chianti_table.npz").exists()

    def test_lookup_returns_nonneg_spectrum(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "chianti_table.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        table = CHIANTILookupTable()
        table.generate(self._bins())
        spec = table.lookup(15.0, 1e48, self._bins())
        assert spec.shape == (24,)
        assert np.all(spec >= 0)

    def test_lookup_higher_T_peaks_at_higher_energy(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "chianti_table.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        bins = np.linspace(1.0, 25.0, 25)
        table = CHIANTILookupTable()
        table.generate(bins)
        spec_cold = table.lookup(5.0, 1e48, bins)
        spec_hot  = table.lookup(30.0, 1e48, bins)
        centres = (bins[:-1] + bins[1:]) / 2.0
        mean_e_cold = np.average(centres, weights=spec_cold + 1e-30)
        mean_e_hot  = np.average(centres, weights=spec_hot  + 1e-30)
        assert mean_e_hot > mean_e_cold
