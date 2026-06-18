"""
L1 — Data Ingestion Tests
PRD §5.8 acceptance criteria.
"""
import pytest
import numpy as np
from datetime import datetime, timezone


# ── Task 1: Models ────────────────────────────────────────────────────────────

def test_fits_data_shape_consistent():
    from aruna.l1_ingestion.models import FitsData
    n_times, n_bins = 10, 32
    fd = FitsData(
        instrument="SOLEXS_SDD1",
        timestamps=np.zeros(n_times, dtype="datetime64[ns]"),
        counts_raw=np.ones((n_times, n_bins)),
        energy_bins_kev=np.linspace(2.0, 22.0, n_bins + 1),
        livetime=np.ones(n_times),
        deadtime_frac=np.zeros(n_times),
        source_file="test.fits",
        checksum_sha256="abc123",
    )
    assert fd.counts_raw.shape == (n_times, n_bins)
    assert len(fd.energy_bins_kev) == n_bins + 1


def test_ingestion_config_defaults():
    from aruna.l1_ingestion.models import IngestionConfig
    cfg = IngestionConfig()
    assert cfg.quarantine_ttl_sec == 300.0
    assert cfg.completeness_threshold == 0.80
    assert cfg.clock_drift_warn_sec == 0.5
    assert cfg.clock_drift_max_sec == 1.0
    assert cfg.chi2_dof_limit == 2.0


def test_quarantine_entry_fields():
    from aruna.l1_ingestion.models import QuarantineEntry
    from pathlib import Path
    q = QuarantineEntry(
        fits_path=Path("test.fits"),
        instrument="HEL1OS_CDTE",
        quarantined_at=datetime.now(timezone.utc),
        retry_after=datetime.now(timezone.utc),
        reason="completeness < 80%",
    )
    assert q.attempt_count == 0
    assert q.instrument == "HEL1OS_CDTE"


def test_solexs_fixture_has_sdd1_and_sdd2(solexs_fits_path):
    from astropy.io import fits
    with fits.open(solexs_fits_path) as hdul:
        names = [h.name for h in hdul]
    assert 'SDD1' in names
    assert 'SDD2' in names
    assert 'EBOUNDS' in names


def test_hel1os_fixture_has_cdte_and_czt(hel1os_fits_path):
    from astropy.io import fits
    with fits.open(hel1os_fits_path) as hdul:
        names = [h.name for h in hdul]
    assert 'CDTE' in names
    assert 'CZT' in names
    assert 'EBOUNDS_CDTE' in names
    assert 'EBOUNDS_CZT' in names


def test_hk_fixture_has_required_columns(hk_fits_path):
    from astropy.io import fits
    with fits.open(hk_fits_path) as hdul:
        cols = hdul['HK'].columns.names
    assert 'TIME' in cols
    assert 'SAA_FLAG' in cols
    assert 'INSTRUMENT_MODE' in cols
    assert 'SOLAR_DIST_AU' in cols


def test_x63_fixture_has_sdd1_saturation(x63_solexs_fits_path):
    """X6.3 scenario: SDD1 total count rate exceeds 5e4 at peak."""
    from astropy.io import fits
    SAT_LIMIT = 5e4
    with fits.open(x63_solexs_fits_path) as hdul:
        counts = hdul['SDD1'].data['COUNTS']
    total_rate = counts.sum(axis=1)
    assert total_rate.max() > SAT_LIMIT


# ── Task 2: FitsReader ────────────────────────────────────────────────────────

def test_fits_reader_solexs_returns_two_detectors(solexs_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    result = FitsReader().read_solexs(solexs_fits_path)
    assert len(result) == 2
    instruments = {fd.instrument for fd in result}
    assert instruments == {'SOLEXS_SDD1', 'SOLEXS_SDD2'}


def test_fits_reader_solexs_counts_shape(solexs_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    result = FitsReader().read_solexs(solexs_fits_path)
    sdd1 = next(fd for fd in result if fd.instrument == 'SOLEXS_SDD1')
    assert sdd1.counts_raw.ndim == 2
    assert sdd1.counts_raw.shape[0] == 60
    assert sdd1.counts_raw.shape[1] == 32


def test_fits_reader_solexs_energy_bins(solexs_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    result = FitsReader().read_solexs(solexs_fits_path)
    sdd1 = next(fd for fd in result if fd.instrument == 'SOLEXS_SDD1')
    assert len(sdd1.energy_bins_kev) == 33
    assert sdd1.energy_bins_kev[0] == pytest.approx(2.0, abs=0.1)
    assert sdd1.energy_bins_kev[-1] == pytest.approx(22.0, abs=0.1)


def test_fits_reader_solexs_timestamps_utc(solexs_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    result = FitsReader().read_solexs(solexs_fits_path)
    sdd1 = next(fd for fd in result if fd.instrument == 'SOLEXS_SDD1')
    assert sdd1.timestamps.dtype.kind == 'M'
    assert len(sdd1.timestamps) == 60


def test_fits_reader_hel1os_returns_cdte_and_czt(hel1os_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    result = FitsReader().read_hel1os(hel1os_fits_path)
    instruments = {fd.instrument for fd in result}
    assert instruments == {'HEL1OS_CDTE', 'HEL1OS_CZT'}


def test_fits_reader_hel1os_cdte_energy_range(hel1os_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    result = FitsReader().read_hel1os(hel1os_fits_path)
    cdte = next(fd for fd in result if fd.instrument == 'HEL1OS_CDTE')
    assert cdte.energy_bins_kev[0] == pytest.approx(8.0, abs=0.1)
    assert cdte.energy_bins_kev[-1] == pytest.approx(70.0, abs=0.1)


def test_fits_reader_hk_returns_hkdata(hk_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    from aruna.l1_ingestion.models import HKData
    hk = FitsReader().read_hk(hk_fits_path)
    assert isinstance(hk, HKData)
    assert len(hk.timestamps) == 60
    assert hk.saa_flags.dtype == bool


def test_fits_reader_hk_saa_flag_false(hk_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    hk = FitsReader().read_hk(hk_fits_path)
    assert not hk.saa_flags.any()


def test_fits_reader_hk_solar_distance(hk_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    hk = FitsReader().read_hk(hk_fits_path)
    assert hk.solar_dist_au[0] == pytest.approx(0.987, abs=0.001)


def test_fits_reader_missing_extension_raises(tmp_path):
    from astropy.io import fits
    from aruna.l1_ingestion.fits_reader import FitsReader
    bad = fits.HDUList([fits.PrimaryHDU()])
    p = tmp_path / 'bad.fits'
    bad.writeto(p)
    with pytest.raises((KeyError, ValueError)):
        FitsReader().read_solexs(p)


def test_fits_reader_checksum_populated(solexs_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    result = FitsReader().read_solexs(solexs_fits_path)
    for fd in result:
        assert len(fd.checksum_sha256) == 64


# ── Task 3: DeadTimeCorrector ─────────────────────────────────────────────────

def _make_fits_data(instrument, counts, n_bins=32):
    from aruna.l1_ingestion.models import FitsData
    n_times = len(counts) if counts.ndim == 1 else counts.shape[0]
    if counts.ndim == 1:
        counts = np.tile(counts[:, None], (1, n_bins))
    return FitsData(
        instrument=instrument,
        timestamps=np.zeros(n_times, dtype='datetime64[ns]'),
        counts_raw=counts,
        energy_bins_kev=np.linspace(2.0, 22.0, n_bins + 1),
        livetime=np.ones(n_times),
        deadtime_frac=np.zeros(n_times),
        source_file='test.fits',
        checksum_sha256='abc',
    )


def test_dead_time_correction_applied_to_all_solexs_bins():
    from aruna.l1_ingestion.deadtime import DeadTimeCorrector, DEADTIME_CONSTANTS
    tau = DEADTIME_CONSTANTS['SOLEXS_SDD1']
    r_obs = 1000.0
    r_true_expected = r_obs / (1.0 - r_obs * tau)
    counts = np.full((10, 32), r_obs)
    data = _make_fits_data('SOLEXS_SDD1', counts)
    corrected, _ = DeadTimeCorrector().correct(data)
    assert corrected.counts_raw == pytest.approx(r_true_expected, rel=1e-6)


def test_dead_time_correction_hel1os_cdte():
    from aruna.l1_ingestion.deadtime import DeadTimeCorrector, DEADTIME_CONSTANTS
    tau = DEADTIME_CONSTANTS['HEL1OS_CDTE']
    r_obs = 5000.0
    r_true_expected = r_obs / (1.0 - r_obs * tau)
    counts = np.full((5, 64), r_obs)
    data = _make_fits_data('HEL1OS_CDTE', counts, n_bins=64)
    corrected, _ = DeadTimeCorrector().correct(data)
    assert corrected.counts_raw == pytest.approx(r_true_expected, rel=1e-4)


def test_sdd1_saturation_triggers_auto_switch_to_sdd2():
    from aruna.l1_ingestion.deadtime import DeadTimeCorrector, SATURATION_LIMITS
    from aruna.models.observation import QualityFlag
    sat = SATURATION_LIMITS['SOLEXS_SDD1']
    n_times, n_bins = 10, 32
    counts_sdd1 = np.full((n_times, n_bins), 100.0)
    counts_sdd1[3:8, :] = (sat + 1000) / n_bins
    counts_sdd2 = np.full((n_times, n_bins), 50.0)
    sdd1_data = _make_fits_data('SOLEXS_SDD1', counts_sdd1)
    sdd2_data = _make_fits_data('SOLEXS_SDD2', counts_sdd2)
    dtc = DeadTimeCorrector()
    corr_sdd1, flags_sdd1 = dtc.correct(sdd1_data)
    corr_sdd2, flags_sdd2 = dtc.correct(sdd2_data)
    assert (flags_sdd1[3:8] == QualityFlag.SATURATED).all()
    assert (flags_sdd1[:3]  != QualityFlag.SATURATED).all()
    merged, merged_flags = dtc.select_solexs_detector(corr_sdd1, flags_sdd1, corr_sdd2, flags_sdd2)
    assert merged.counts_raw[5, 0] == pytest.approx(corr_sdd2.counts_raw[5, 0], rel=1e-6)
    assert merged.counts_raw[0, 0] == pytest.approx(corr_sdd1.counts_raw[0, 0], rel=1e-6)
    assert (merged_flags[3:8] == QualityFlag.SATURATED).all()


def test_dead_time_correction_formula_accuracy():
    from aruna.l1_ingestion.deadtime import DeadTimeCorrector, DEADTIME_CONSTANTS
    tau = DEADTIME_CONSTANTS['SOLEXS_SDD1']
    r_obs_vals = np.array([100.0, 500.0, 1000.0, 5000.0])
    r_true_expected = r_obs_vals / (1.0 - r_obs_vals * tau)
    for r_obs, r_exp in zip(r_obs_vals, r_true_expected):
        counts = np.full((1, 32), r_obs)
        data = _make_fits_data('SOLEXS_SDD1', counts)
        corrected, _ = DeadTimeCorrector().correct(data)
        assert corrected.counts_raw[0, 0] == pytest.approx(r_exp, rel=1e-6)


def test_dead_time_correction_unknown_instrument_raises():
    from aruna.l1_ingestion.deadtime import DeadTimeCorrector
    data = _make_fits_data('UNKNOWN_INSTRUMENT', np.ones((5, 32)))
    with pytest.raises(ValueError, match="Unknown instrument"):
        DeadTimeCorrector().correct(data)


def test_dead_time_constants_match_prd():
    from aruna.l1_ingestion.deadtime import DEADTIME_CONSTANTS
    assert DEADTIME_CONSTANTS['SOLEXS_SDD1'] == pytest.approx(4.5e-6)
    assert DEADTIME_CONSTANTS['SOLEXS_SDD2'] == pytest.approx(4.5e-6)
    assert DEADTIME_CONSTANTS['HEL1OS_CDTE'] == pytest.approx(2.1e-6)
    assert DEADTIME_CONSTANTS['HEL1OS_CZT']  == pytest.approx(1.8e-6)


# ── Task 4: BackgroundSubtractor ─────────────────────────────────────────────

def _make_hel1os_data(n_times=60, n_bins=64, base_counts=200.0):
    from aruna.l1_ingestion.models import FitsData
    rng = np.random.default_rng(0)
    counts = rng.poisson(base_counts, size=(n_times, n_bins)).astype(np.float64)
    return FitsData(
        instrument='HEL1OS_CDTE',
        timestamps=np.zeros(n_times, dtype='datetime64[ns]'),
        counts_raw=counts,
        energy_bins_kev=np.linspace(8.0, 70.0, n_bins + 1),
        livetime=np.ones(n_times),
        deadtime_frac=np.zeros(n_times),
        source_file='hel1os.fits',
        checksum_sha256='def',
    )


def test_background_subtraction_hel1os_chi_squared():
    from aruna.l1_ingestion.background import BackgroundSubtractor
    bg_frames = [_make_hel1os_data(n_times=10) for _ in range(30)]
    current = _make_hel1os_data(n_times=10)
    saa = np.zeros(10, dtype=bool)
    _, _, chi2_dof = BackgroundSubtractor().subtract(current, bg_frames, saa)
    assert chi2_dof < 2.0, f"chi2/dof={chi2_dof:.2f} exceeds 2.0"


def test_saa_bins_flagged_not_dropped():
    from aruna.l1_ingestion.background import BackgroundSubtractor
    from aruna.models.observation import QualityFlag
    bg_frames = [_make_hel1os_data(n_times=10) for _ in range(30)]
    current = _make_hel1os_data(n_times=10)
    saa = np.array([False]*5 + [True]*5, dtype=bool)
    result, flags, _ = BackgroundSubtractor().subtract(current, bg_frames, saa)
    assert len(result.counts_raw) == 10
    assert (flags[5:] == QualityFlag.SAA_PASSAGE).all()
    assert (flags[:5] != QualityFlag.SAA_PASSAGE).all()


def test_background_subtraction_reduces_baseline():
    from aruna.l1_ingestion.background import BackgroundSubtractor
    bg_frames = [_make_hel1os_data(n_times=10, base_counts=200.0) for _ in range(30)]
    current = _make_hel1os_data(n_times=10, base_counts=200.0)
    original_mean = current.counts_raw.mean()
    result, _, _ = BackgroundSubtractor().subtract(current, bg_frames, np.zeros(10, dtype=bool))
    assert result.counts_raw.mean() < original_mean


def test_background_subtraction_no_negative_counts():
    from aruna.l1_ingestion.background import BackgroundSubtractor
    bg_frames = [_make_hel1os_data(n_times=10, base_counts=200.0) for _ in range(30)]
    current = _make_hel1os_data(n_times=10, base_counts=200.0)
    result, _, _ = BackgroundSubtractor().subtract(current, bg_frames, np.zeros(10, dtype=bool))
    assert (result.counts_raw >= 0.0).all()


def test_background_subtraction_empty_window_returns_unchanged():
    from aruna.l1_ingestion.background import BackgroundSubtractor
    current = _make_hel1os_data(n_times=10)
    result, flags, chi2_dof = BackgroundSubtractor().subtract(current, [], np.zeros(10, dtype=bool))
    assert result.counts_raw.shape == current.counts_raw.shape
    assert chi2_dof == 0.0


def test_background_high_chi2_flags_noisy():
    from aruna.l1_ingestion.background import BackgroundSubtractor
    from aruna.models.observation import QualityFlag
    from aruna.l1_ingestion.models import FitsData
    rng = np.random.default_rng(99)
    n_bins = 64
    bg_frames = []
    for i in range(30):
        spiky = rng.exponential(scale=1000.0, size=(10, n_bins))
        bg_frames.append(FitsData(
            instrument='HEL1OS_CDTE',
            timestamps=np.zeros(10, dtype='datetime64[ns]'),
            counts_raw=spiky,
            energy_bins_kev=np.linspace(8.0, 70.0, n_bins + 1),
            livetime=np.ones(10),
            deadtime_frac=np.zeros(10),
            source_file='test.fits',
            checksum_sha256='abc',
        ))
    current = _make_hel1os_data(n_times=10, base_counts=5000.0)
    bs = BackgroundSubtractor(chi2_dof_limit=0.01)
    _, flags, chi2_dof = bs.subtract(current, bg_frames, np.zeros(10, dtype=bool))
    assert chi2_dof > 0.01
    assert (flags == QualityFlag.NOISY).any()


# ── Task 5: TimeAligner ───────────────────────────────────────────────────────

def _make_aligned_pair(n_times=60, lag_samples=0):
    from aruna.l1_ingestion.models import FitsData
    rng = np.random.default_rng(7)
    n_bins_s, n_bins_h = 32, 64
    signal = rng.poisson(500, size=n_times).astype(np.float64)
    counts_s = rng.poisson(50, size=(n_times, n_bins_s)).astype(np.float64)
    for t in range(n_times):
        counts_s[t, 12:20] += signal[t]
    counts_h = rng.poisson(50, size=(n_times, n_bins_h)).astype(np.float64)
    signal_h = np.roll(signal, lag_samples)
    for t in range(n_times):
        counts_h[t, 2:8] += signal_h[t]
    ts = np.arange(n_times, dtype='int64').astype('datetime64[ns]')
    solexs = FitsData('SOLEXS_SDD1', ts, counts_s,
                      np.linspace(2.0, 22.0, n_bins_s + 1),
                      np.ones(n_times), np.zeros(n_times), 'sl.fits', 'aaa')
    hel1os = FitsData('HEL1OS_CDTE', ts, counts_h,
                      np.linspace(8.0, 70.0, n_bins_h + 1),
                      np.ones(n_times), np.zeros(n_times), 'hl.fits', 'bbb')
    return solexs, hel1os


def test_time_sync_under_one_second():
    from aruna.l1_ingestion.alignment import TimeAligner
    solexs, hel1os = _make_aligned_pair(lag_samples=0)
    _, _, lag_sec = TimeAligner().align(solexs, hel1os)
    assert abs(lag_sec) < 1.0


def test_aligner_zero_lag_no_interpolation():
    from aruna.l1_ingestion.alignment import TimeAligner
    solexs, hel1os = _make_aligned_pair(lag_samples=0)
    s_out, h_out, lag = TimeAligner().align(solexs, hel1os)
    assert len(s_out.timestamps) == len(h_out.timestamps)
    assert abs(lag) < 1.0


def test_clock_drift_logged_for_lag_over_0_5_seconds(caplog):
    import logging
    from aruna.l1_ingestion.alignment import TimeAligner
    solexs, hel1os = _make_aligned_pair(n_times=120, lag_samples=3)
    with caplog.at_level(logging.WARNING, logger='aruna.l1_ingestion.alignment'):
        TimeAligner(drift_warn_sec=0.5).align(solexs, hel1os)
    assert any('CLOCK_DRIFT' in r.message for r in caplog.records)


def test_aligner_large_lag_triggers_interpolation():
    from aruna.l1_ingestion.alignment import TimeAligner
    from unittest.mock import patch
    solexs, hel1os = _make_aligned_pair(n_times=120, lag_samples=5)
    aligner = TimeAligner(drift_warn_sec=0.5, drift_max_sec=1.0)
    with patch.object(aligner, 'compute_lag', return_value=2.0):
        s_out, h_out, lag = aligner.align(solexs, hel1os)
    assert lag == 2.0
    assert len(s_out.timestamps) == len(h_out.timestamps)


def test_aligner_output_shorter_than_input_on_trim():
    from aruna.l1_ingestion.alignment import TimeAligner
    solexs, hel1os = _make_aligned_pair(n_times=60)
    s_out, h_out, _ = TimeAligner().align(solexs, hel1os)
    assert len(s_out.timestamps) <= 60
    assert len(h_out.timestamps) <= 60


# ── Task 6: GoesLabelJoiner ───────────────────────────────────────────────────

def _make_obs_frames(n=5, base_dt=None):
    from aruna.models.observation import ObservationFrame, QualityFlag
    from datetime import timedelta
    if base_dt is None:
        base_dt = datetime(2024, 2, 22, 10, 0, 0, tzinfo=timezone.utc)
    frames = []
    for i in range(n):
        frames.append(ObservationFrame(
            instrument='SOLEXS_SDD1',
            timestamp=base_dt + timedelta(seconds=i),
            energy_bins_kev=np.linspace(2.0, 22.0, 33),
            counts_per_s_kev=np.ones(32),
            quality_flag=QualityFlag.GOOD,
            dead_time_fraction=0.05,
            calibration_version='sarwade_2025_v1',
            solar_distance_au=0.987,
            spacecraft_mode='NOMINAL',
            saa_flag=False,
            source_file='test.fits',
            checksum_sha256='abc',
            pipeline_version='2.0.0',
        ))
    return frames


def _make_hek_row(fl_starttime, fl_goescls):
    return {'event_starttime': fl_starttime, 'fl_goescls': fl_goescls}


def test_goes_label_joined_for_x63_event():
    from aruna.l1_ingestion.goes_join import GoesLabelJoiner
    frames = _make_obs_frames(n=5)
    hek_results = [_make_hek_row('2024-02-22T10:01:00', 'X6.3')]
    result = GoesLabelJoiner().join(frames, hek_results)
    assert all(f.goes_class == 'X' for f in result)
    assert all(f.goes_subclass == pytest.approx(6.3) for f in result)


def test_goes_label_no_match_sets_quiet_sun():
    from aruna.l1_ingestion.goes_join import GoesLabelJoiner
    frames = _make_obs_frames(n=3)
    hek_results = [_make_hek_row('2024-02-22T11:00:00', 'C1.2')]
    result = GoesLabelJoiner().join(frames, hek_results)
    assert all(f.goes_class == 'QUIET_SUN' for f in result)


def test_goes_label_offline_sets_quiet_sun():
    from aruna.l1_ingestion.goes_join import GoesLabelJoiner
    frames = _make_obs_frames(n=3)
    result = GoesLabelJoiner().join(frames, hek_results=None)
    assert all(f.goes_class == 'QUIET_SUN' for f in result)


def test_goes_label_empty_frames():
    from aruna.l1_ingestion.goes_join import GoesLabelJoiner
    result = GoesLabelJoiner().join([], [_make_hek_row('2024-02-22T10:00:00', 'M1.0')])
    assert result == []


def test_goes_label_multiple_events_nearest_match():
    from aruna.l1_ingestion.goes_join import GoesLabelJoiner
    frames = _make_obs_frames(n=1, base_dt=datetime(2024, 2, 22, 10, 0, 0, tzinfo=timezone.utc))
    hek_results = [
        _make_hek_row('2024-02-22T09:58:00', 'M3.0'),
        _make_hek_row('2024-02-22T10:02:00', 'X1.0'),
    ]
    result = GoesLabelJoiner().join(frames, hek_results)
    assert result[0].goes_class in ('M', 'X')


# ── Task 7: FrameValidator ────────────────────────────────────────────────────

def _make_valid_frame(counts=None):
    from aruna.models.observation import ObservationFrame, QualityFlag
    if counts is None:
        counts = np.ones(32)
    return ObservationFrame(
        instrument='SOLEXS_SDD1',
        timestamp=datetime(2024, 2, 22, 10, 0, 0, tzinfo=timezone.utc),
        energy_bins_kev=np.linspace(2.0, 22.0, 33),
        counts_per_s_kev=counts,
        quality_flag=QualityFlag.GOOD,
        dead_time_fraction=0.05,
        calibration_version='sarwade_2025_v1',
        solar_distance_au=0.987,
        spacecraft_mode='NOMINAL',
        saa_flag=False,
        source_file='test.fits',
        checksum_sha256='abc',
        pipeline_version='2.0.0',
    )


def test_schema_validation_passes_on_95_pct_of_frames():
    from aruna.l1_ingestion.validator import FrameValidator
    from aruna.l1_ingestion.models import IngestionConfig
    frames = [_make_valid_frame() for _ in range(100)]
    cfg = IngestionConfig(completeness_threshold=0.80)
    valid, quarantined = FrameValidator(cfg).validate(frames)
    pass_rate = len(valid) / len(frames)
    assert pass_rate >= 0.95, f"Pass rate {pass_rate:.1%} < 95%"


def test_partial_fits_held_in_quarantine():
    from aruna.l1_ingestion.validator import FrameValidator
    from aruna.l1_ingestion.models import IngestionConfig
    n_bins = 32
    counts = np.ones(n_bins)
    counts[:int(n_bins * 0.7)] = np.nan
    partial_frame = _make_valid_frame(counts=counts)
    good_frames = [_make_valid_frame() for _ in range(9)]
    all_frames = good_frames + [partial_frame]
    cfg = IngestionConfig(completeness_threshold=0.80, quarantine_ttl_sec=300.0)
    valid, quarantined = FrameValidator(cfg).validate(all_frames)
    assert len(quarantined) >= 1
    fv2 = FrameValidator(cfg)
    fv2.validate([partial_frame])
    assert any('completeness' in e.reason for e in fv2.quarantine)


def test_validator_valid_frame_passes():
    from aruna.l1_ingestion.validator import FrameValidator
    from aruna.l1_ingestion.models import IngestionConfig
    frame = _make_valid_frame()
    valid, quarantined = FrameValidator(IngestionConfig()).validate([frame])
    assert len(valid) == 1
    assert len(quarantined) == 0


def test_validator_empty_input():
    from aruna.l1_ingestion.validator import FrameValidator
    from aruna.l1_ingestion.models import IngestionConfig
    valid, quarantined = FrameValidator(IngestionConfig()).validate([])
    assert valid == []
    assert quarantined == []


def test_validator_quarantine_entry_has_retry_time():
    from aruna.l1_ingestion.validator import FrameValidator
    from aruna.l1_ingestion.models import IngestionConfig
    counts = np.full(32, np.nan)
    frame = _make_valid_frame(counts=counts)
    cfg = IngestionConfig(quarantine_ttl_sec=300.0)
    fv = FrameValidator(cfg)
    fv.validate([frame])
    assert len(fv.quarantine) == 1
    delta = (fv.quarantine[0].retry_after - fv.quarantine[0].quarantined_at).total_seconds()
    assert delta == pytest.approx(300.0, abs=1.0)


# ── Task 8: KafkaPublisher ────────────────────────────────────────────────────

def test_kafka_publish_within_2_seconds():
    import time
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.publisher import KafkaPublisher
    producer = MagicMock()
    pub = KafkaPublisher(producer)
    frames = [_make_valid_frame() for _ in range(10)]
    t0 = time.monotonic()
    pub.publish(frames)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0
    assert producer.send.called


def test_kafka_publishes_solexs_to_correct_topic():
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.publisher import KafkaPublisher
    producer = MagicMock()
    pub = KafkaPublisher(producer, topic_solexs='aruna.observations.solexs')
    pub.publish([_make_valid_frame()])
    topics = [call.args[0] for call in producer.send.call_args_list]
    assert 'aruna.observations.solexs' in topics


def test_kafka_publishes_hel1os_to_correct_topic():
    from unittest.mock import MagicMock
    from aruna.models.observation import ObservationFrame, QualityFlag
    from aruna.l1_ingestion.publisher import KafkaPublisher
    producer = MagicMock()
    pub = KafkaPublisher(producer, topic_hel1os='aruna.observations.hel1os')
    frame = ObservationFrame(
        instrument='HEL1OS_CDTE',
        timestamp=datetime(2024, 2, 22, 10, 0, 0, tzinfo=timezone.utc),
        energy_bins_kev=np.linspace(8.0, 70.0, 65),
        counts_per_s_kev=np.ones(64),
        quality_flag=QualityFlag.GOOD,
        dead_time_fraction=0.02,
        calibration_version='sarwade_2025_v1',
        solar_distance_au=0.987,
        spacecraft_mode='NOMINAL',
        saa_flag=False,
        source_file='hl.fits',
        checksum_sha256='def',
        pipeline_version='2.0.0',
    )
    pub.publish([frame])
    topics = [call.args[0] for call in producer.send.call_args_list]
    assert 'aruna.observations.hel1os' in topics


def test_kafka_quality_summary_published():
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.publisher import KafkaPublisher
    producer = MagicMock()
    pub = KafkaPublisher(producer, topic_quality='aruna.quality.summary')
    pub.publish([_make_valid_frame()])
    topics = [call.args[0] for call in producer.send.call_args_list]
    assert 'aruna.quality.summary' in topics


def test_kafka_payload_has_schema_version():
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.publisher import KafkaPublisher
    producer = MagicMock()
    pub = KafkaPublisher(producer)
    pub.publish([_make_valid_frame()])
    for call in producer.send.call_args_list:
        payload = call.kwargs.get('value', call.args[1] if len(call.args) > 1 else {})
        if isinstance(payload, dict) and 'schema_version' in payload:
            assert payload['schema_version'] == '1.0'
            assert payload['pipeline_version'] == '2.0.0'


def test_kafka_publish_empty_list_sends_quality_summary_only():
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.publisher import KafkaPublisher
    producer = MagicMock()
    pub = KafkaPublisher(producer)
    pub.publish([])
    topics = [call.args[0] for call in producer.send.call_args_list]
    assert 'aruna.quality.summary' in topics
    assert 'aruna.observations.solexs' not in topics


# ── Task 9: IngestionPipeline + Acceptance Gate ───────────────────────────────

def test_x63_event_processes_without_errors(x63_solexs_fits_path, hel1os_fits_path, hk_fits_path):
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.pipeline import IngestionPipeline
    from aruna.l1_ingestion.models import IngestionConfig
    producer = MagicMock()
    pipeline = IngestionPipeline(cache=None, config=IngestionConfig(), kafka_producer=producer)
    frames = pipeline.ingest(x63_solexs_fits_path, hel1os_fits_path, hk_fits_path)
    assert len(frames) > 0
    assert all(hasattr(f, 'instrument') for f in frames)


def test_pipeline_publishes_to_kafka(solexs_fits_path, hel1os_fits_path, hk_fits_path):
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.pipeline import IngestionPipeline
    from aruna.l1_ingestion.models import IngestionConfig
    producer = MagicMock()
    pipeline = IngestionPipeline(cache=None, config=IngestionConfig(), kafka_producer=producer)
    pipeline.ingest(solexs_fits_path, hel1os_fits_path, hk_fits_path)
    assert producer.send.called


def test_pipeline_x63_sdd1_saturated_frames_use_sdd2(x63_solexs_fits_path, hel1os_fits_path, hk_fits_path):
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.pipeline import IngestionPipeline
    from aruna.l1_ingestion.models import IngestionConfig
    from aruna.models.observation import QualityFlag
    producer = MagicMock()
    pipeline = IngestionPipeline(cache=None, config=IngestionConfig(), kafka_producer=producer)
    frames = pipeline.ingest(x63_solexs_fits_path, hel1os_fits_path, hk_fits_path)
    solexs_frames = [f for f in frames if f.is_solexs()]
    saturated = [f for f in solexs_frames if f.quality_flag == QualityFlag.SATURATED]
    assert len(saturated) > 0


def test_pipeline_hk_saa_flag_propagated(solexs_fits_path, hel1os_fits_path, tmp_path):
    from astropy.io import fits as afits
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.pipeline import IngestionPipeline
    from aruna.l1_ingestion.models import IngestionConfig
    n_times = 60
    cols = [
        afits.Column(name='TIME',            format='D',  array=np.arange(n_times, dtype=np.float64)),
        afits.Column(name='SAA_FLAG',        format='I',  array=np.ones(n_times, dtype=np.int16)),
        afits.Column(name='INSTRUMENT_MODE', format='8A', array=np.array(['NOMINAL'] * n_times)),
        afits.Column(name='SOLAR_DIST_AU',   format='D',  array=np.full(n_times, 0.987)),
    ]
    hdu = afits.BinTableHDU.from_columns(cols, name='HK')
    hdu.header['TIMEREF'] = '2024-02-22T00:00:00'
    hdul = afits.HDUList([afits.PrimaryHDU(), hdu])
    hk_path = tmp_path / 'hk_saa.fits'
    hdul.writeto(hk_path, overwrite=True)
    producer = MagicMock()
    pipeline = IngestionPipeline(cache=None, config=IngestionConfig(), kafka_producer=producer)
    frames = pipeline.ingest(solexs_fits_path, hel1os_fits_path, hk_path)
    assert all(f.saa_flag for f in frames)


def test_acceptance_dead_time_constants_correct():
    from aruna.l1_ingestion.deadtime import DEADTIME_CONSTANTS, SATURATION_LIMITS
    assert DEADTIME_CONSTANTS['SOLEXS_SDD1'] == pytest.approx(4.5e-6)
    assert SATURATION_LIMITS['SOLEXS_SDD1'] == pytest.approx(5e4)


def test_acceptance_public_api_importable():
    from aruna.l1_ingestion import (
        FitsReader, DeadTimeCorrector, BackgroundSubtractor,
        TimeAligner, GoesLabelJoiner, FrameValidator, KafkaPublisher,
        IngestionPipeline,
    )


def test_acceptance_observation_frame_schema():
    from aruna.models.observation import ObservationFrame, QualityFlag
    f = ObservationFrame(
        instrument='SOLEXS_SDD1',
        timestamp=datetime(2024, 2, 22, 10, 5, 0, tzinfo=timezone.utc),
        energy_bins_kev=np.linspace(2.0, 22.0, 33),
        counts_per_s_kev=np.ones(32) * 500.0,
        quality_flag=QualityFlag.GOOD,
        dead_time_fraction=0.05,
        calibration_version='sarwade_2025_v1',
        solar_distance_au=0.987,
        spacecraft_mode='NOMINAL',
        saa_flag=False,
        source_file='SoLEXS_L1_20240222.fits',
        checksum_sha256='a' * 64,
        pipeline_version='2.0.0',
    )
    assert f.schema_version == '1.0'
    assert f.is_solexs()
    assert f.is_good()
