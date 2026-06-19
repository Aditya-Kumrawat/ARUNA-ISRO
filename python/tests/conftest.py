"""
Shared pytest fixtures and configuration for the ARUNA test suite.
"""

import pathlib
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from astropy.io import fits as astropy_fits


@pytest.fixture(scope="session")
def x63_fits_dir(tmp_path_factory):
    """
    Path to X6.3 event (2024-02-22) FITS fixtures.
    Populated with real data before Phase 1 testing begins.
    """
    import pathlib
    fixture_dir = pathlib.Path(__file__).parent / "fixtures" / "x63"
    if not any(fixture_dir.glob("*.fits")):
        pytest.skip("X6.3 FITS fixtures not yet populated — add files to tests/fixtures/x63/")
    return fixture_dir


@pytest.fixture(scope="session")
def config():
    """Load the ARUNA config for use in tests."""
    import pathlib
    import yaml

    config_path = pathlib.Path(__file__).parent.parent.parent / "config" / "config.yaml"
    if not config_path.exists():
        config_path = config_path.parent / "config.example.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Synthetic FITS fixtures for L1 ingestion tests ────────────────────────────

def _make_bintable(ext_name, n_times, n_bins, counts,
                   timeref='2024-02-22T00:00:00', instrume=''):
    times = np.arange(n_times, dtype=np.float64)
    livetime = np.full(n_times, 0.95)
    dt_frac = np.full(n_times, 0.05)
    cols = [
        astropy_fits.Column(name='TIME', format='D', unit='s', array=times),
        astropy_fits.Column(name='COUNTS', format=f'{n_bins}D', unit='counts/s', array=counts),
        astropy_fits.Column(name='LIVETIME', format='D', array=livetime),
        astropy_fits.Column(name='DEADTIME_FRAC', format='D', array=dt_frac),
    ]
    hdu = astropy_fits.BinTableHDU.from_columns(cols, name=ext_name)
    hdu.header['TIMEREF'] = timeref
    hdu.header['INSTRUME'] = instrume
    return hdu


def _make_ebounds(name, e_min, e_max):
    cols = [
        astropy_fits.Column(name='E_MIN', format='D', unit='keV', array=e_min),
        astropy_fits.Column(name='E_MAX', format='D', unit='keV', array=e_max),
    ]
    return astropy_fits.BinTableHDU.from_columns(cols, name=name)


@pytest.fixture
def solexs_fits_path(tmp_path):
    """Synthetic SoLEXS FITS with SDD1, SDD2, EBOUNDS extensions. 60s x 32 bins."""
    n_times, n_bins = 60, 32
    rng = np.random.default_rng(42)
    counts = rng.poisson(500, size=(n_times, n_bins)).astype(np.float64)
    energy = np.linspace(2.0, 22.0, n_bins + 1)
    e_min, e_max = energy[:-1], energy[1:]

    hdul = astropy_fits.HDUList([
        astropy_fits.PrimaryHDU(),
        _make_bintable('SDD1', n_times, n_bins, counts, instrume='SOLEXS_SDD1'),
        _make_bintable('SDD2', n_times, n_bins, counts * 0.98, instrume='SOLEXS_SDD2'),
        _make_ebounds('EBOUNDS', e_min, e_max),
    ])
    path = tmp_path / 'SoLEXS_L1_20240222.fits'
    hdul.writeto(path, overwrite=True)
    return path


@pytest.fixture
def hel1os_fits_path(tmp_path):
    """Synthetic HEL1OS FITS with CDTE, CZT, EBOUNDS_CDTE, EBOUNDS_CZT. 60s."""
    n_times = 60
    rng = np.random.default_rng(43)

    n_cdte, n_czt = 64, 32
    counts_cdte = rng.poisson(200, size=(n_times, n_cdte)).astype(np.float64)
    counts_czt  = rng.poisson(50,  size=(n_times, n_czt)).astype(np.float64)

    e_cdte = np.linspace(8.0,  70.0,  n_cdte + 1)
    e_czt  = np.linspace(20.0, 150.0, n_czt  + 1)

    hdul = astropy_fits.HDUList([
        astropy_fits.PrimaryHDU(),
        _make_bintable('CDTE', n_times, n_cdte, counts_cdte, instrume='HEL1OS_CDTE'),
        _make_bintable('CZT',  n_times, n_czt,  counts_czt,  instrume='HEL1OS_CZT'),
        _make_ebounds('EBOUNDS_CDTE', e_cdte[:-1], e_cdte[1:]),
        _make_ebounds('EBOUNDS_CZT',  e_czt[:-1],  e_czt[1:]),
    ])
    path = tmp_path / 'HEL1OS_L1_20240222.fits'
    hdul.writeto(path, overwrite=True)
    return path


@pytest.fixture
def hk_fits_path(tmp_path):
    """Synthetic Aditya HK FITS. 60s, all NOMINAL, SAA_FLAG=0."""
    n_times = 60
    times = np.arange(n_times, dtype=np.float64)
    cols = [
        astropy_fits.Column(name='TIME',            format='D',  array=times),
        astropy_fits.Column(name='SAA_FLAG',        format='I',  array=np.zeros(n_times, dtype=np.int16)),
        astropy_fits.Column(name='INSTRUMENT_MODE', format='8A', array=np.array(['NOMINAL'] * n_times)),
        astropy_fits.Column(name='SOLAR_DIST_AU',   format='D',  array=np.full(n_times, 0.987)),
    ]
    hdu = astropy_fits.BinTableHDU.from_columns(cols, name='HK')
    hdu.header['TIMEREF'] = '2024-02-22T00:00:00'
    hdul = astropy_fits.HDUList([astropy_fits.PrimaryHDU(), hdu])
    path = tmp_path / 'Aditya_HK_20240222.fits'
    hdul.writeto(path, overwrite=True)
    return path


@pytest.fixture
def x63_solexs_fits_path(tmp_path):
    """
    X6.3 flare scenario: 120 seconds, SDD1 saturates (>5e4 counts/s) at t=60-90s.
    SDD2 remains unsaturated throughout.
    """
    n_times, n_bins = 120, 32
    rng = np.random.default_rng(63)
    energy = np.linspace(2.0, 22.0, n_bins + 1)

    # SDD1: quiet baseline, then spike to saturation
    counts_sdd1 = rng.poisson(1000, size=(n_times, n_bins)).astype(np.float64)
    counts_sdd1[60:90, :] = rng.poisson(2000, size=(30, n_bins)) + 60000.0 / n_bins

    # SDD2: lower-gain, never saturates
    counts_sdd2 = counts_sdd1 * 0.01

    e_min, e_max = energy[:-1], energy[1:]
    hdul = astropy_fits.HDUList([
        astropy_fits.PrimaryHDU(),
        _make_bintable('SDD1', n_times, n_bins, counts_sdd1, instrume='SOLEXS_SDD1'),
        _make_bintable('SDD2', n_times, n_bins, counts_sdd2, instrume='SOLEXS_SDD2'),
        _make_ebounds('EBOUNDS', e_min, e_max),
    ])
    path = tmp_path / 'SoLEXS_L1_X63_20240222.fits'
    hdul.writeto(path, overwrite=True)
    return path


# ── L2 Physics Engine fixtures ────────────────────────────────────────────────

@pytest.fixture
def solexs_x63_frames():
    """120 synthetic SoLEXS ObservationFrames for the X6.3 event (2024-02-22)."""
    import uuid
    from datetime import datetime, timezone, timedelta
    from aruna.models.observation import ObservationFrame, QualityFlag

    n_frames = 120
    n_bins   = 32
    rng      = np.random.default_rng(63)
    energy   = np.linspace(2.0, 22.0, n_bins + 1)
    base_ts  = datetime(2024, 2, 22, 4, 0, 0, tzinfo=timezone.utc)

    frames = []
    for i in range(n_frames):
        # SDD1 saturates during t=60-90 → use SATURATED quality flag
        if 60 <= i < 90:
            counts = rng.poisson(2000, size=n_bins).astype(np.float64) + 60000.0 / n_bins
            qflag  = QualityFlag.SATURATED
        else:
            counts = rng.poisson(1000, size=n_bins).astype(np.float64)
            qflag  = QualityFlag.GOOD

        frames.append(ObservationFrame(
            instrument="SOLEXS_SDD1",
            timestamp=base_ts + timedelta(seconds=i),
            energy_bins_kev=energy.copy(),
            counts_per_s_kev=counts,
            quality_flag=qflag,
            dead_time_fraction=0.05,
            calibration_version="sarwade_2025_v1",
            solar_distance_au=0.987,
            spacecraft_mode="SCIENCE",
            saa_flag=False,
            source_file="SoLEXS_L1_X63_20240222.fits",
            checksum_sha256="abc123",
            pipeline_version="2.0.0",
        ))
    return frames


@pytest.fixture
def hel1os_x63_frames():
    """120 synthetic HEL1OS ObservationFrames for the X6.3 event (2024-02-22)."""
    import uuid
    from datetime import datetime, timezone, timedelta
    from aruna.models.observation import ObservationFrame, QualityFlag

    n_frames = 120
    n_bins   = 64
    rng      = np.random.default_rng(64)
    energy   = np.linspace(8.0, 70.0, n_bins + 1)
    base_ts  = datetime(2024, 2, 22, 4, 0, 0, tzinfo=timezone.utc)

    frames = []
    for i in range(n_frames):
        counts = rng.poisson(200, size=n_bins).astype(np.float64)
        frames.append(ObservationFrame(
            instrument="HEL1OS_CDTE",
            timestamp=base_ts + timedelta(seconds=i),
            energy_bins_kev=energy.copy(),
            counts_per_s_kev=counts,
            quality_flag=QualityFlag.GOOD,
            dead_time_fraction=0.02,
            calibration_version="sarwade_2025_v1",
            solar_distance_au=0.987,
            spacecraft_mode="SCIENCE",
            saa_flag=False,
            source_file="HEL1OS_L1_X63_20240222.fits",
            checksum_sha256="def456",
            pipeline_version="2.0.0",
        ))
    return frames


@pytest.fixture
def physics_config():
    from aruna.l2_physics.models import PhysicsConfig
    return PhysicsConfig()


@pytest.fixture
def quiet_sun_extended_frames(solexs_x63_frames, hel1os_x63_frames):
    """700 quiet-sun frames prepended to X6.3 — fills HFD window before event."""
    import copy
    import uuid
    from datetime import timedelta
    from aruna.models.observation import ObservationFrame, QualityFlag

    sol_ref = solexs_x63_frames[0]
    hel_ref = hel1os_x63_frames[0]
    n_quiet = 700

    def _quiet_frame(ref, idx):
        f = copy.copy(ref)
        f.timestamp = ref.timestamp - timedelta(seconds=n_quiet - idx)
        f.counts_per_s_kev = np.abs(np.random.default_rng(idx).normal(
            1.0, 0.1, len(ref.counts_per_s_kev)))
        f.frame_id = str(uuid.uuid4())
        return f

    sol_quiet = [_quiet_frame(sol_ref, i) for i in range(n_quiet)]
    hel_quiet = [_quiet_frame(hel_ref, i + n_quiet) for i in range(n_quiet)]
    return (sol_quiet + solexs_x63_frames, hel_quiet + hel1os_x63_frames)


# ── Real SoLEXS X6.3 data fixtures ───────────────────────────────────────────
# Source: data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/
# SDD2 energy calibration: 340 channels, 1.6–21.0 keV (~0.057 keV/channel)
# X6.3 peak confirmed at index 81154 = 2024-02-22T22:32:34 UTC in SDD2 LC

_REAL_SDD2_PI = (
    pathlib.Path(__file__).parent.parent.parent
    / "data" / "solexs_20240222" / "AL1_SLX_L1_20240222_v1.0"
    / "SDD2" / "AL1_SOLEXS_20240222_SDD2_L1.pi.gz"
)
_SDD2_ENERGY_BINS = np.linspace(1.6, 21.0, 341)   # 341 edges for 340 bins
_X63_PEAK_IDX     = 81154                           # 2024-02-22T22:32:34 UTC
_X63_WIN          = 120                             # 60 s before peak + 60 s after


@pytest.fixture(scope="session")
def real_solexs_pi_path():
    """Real SoLEXS SDD2 L1 PI spectrum for 2024-02-22; skip if absent."""
    if not _REAL_SDD2_PI.exists():
        pytest.skip("Real SoLEXS SDD2 L1 PI file not found — skipping real-data tests")
    return _REAL_SDD2_PI


@pytest.fixture(scope="session")
def real_x63_solexs_frames(real_solexs_pi_path):
    """
    120 ObservationFrames from real SoLEXS SDD2 data centred on the X6.3 peak.
    Index 60 = 2024-02-22T22:32:34 UTC (peak).
    counts_per_s_kev = raw_counts / (1 s × 0.0571 keV/ch).
    """
    from datetime import datetime, timezone
    from aruna.models.observation import ObservationFrame, QualityFlag

    start = _X63_PEAK_IDX - 60
    bin_w = float(_SDD2_ENERGY_BINS[1] - _SDD2_ENERGY_BINS[0])

    with astropy_fits.open(real_solexs_pi_path) as h:
        s        = h["SPECTRUM"]
        tstarts  = s.data["TSTART"] [start: start + _X63_WIN]
        counts   = s.data["COUNTS"] [start: start + _X63_WIN]
        exposure = s.data["EXPOSURE"][start: start + _X63_WIN]

    frames = []
    for i in range(_X63_WIN):
        raw = np.where(np.isfinite(counts[i]), counts[i], 0.0)
        exp = float(exposure[i]) if float(exposure[i]) > 0.0 else 1.0
        csk = raw / (exp * bin_w)
        ts  = datetime.fromtimestamp(float(tstarts[i]), tz=timezone.utc)
        frames.append(ObservationFrame(
            instrument          = "SOLEXS_SDD2",
            timestamp           = ts,
            energy_bins_kev     = _SDD2_ENERGY_BINS.copy(),
            counts_per_s_kev    = csk,
            quality_flag        = QualityFlag.GOOD,
            dead_time_fraction  = 0.02,
            calibration_version = "sarwade_2025_v1",
            solar_distance_au   = 0.987,
            spacecraft_mode     = "SCIENCE",
            saa_flag            = False,
            source_file         = str(real_solexs_pi_path),
            checksum_sha256     = "real_data",
            pipeline_version    = "2.0.0",
        ))
    return frames


@pytest.fixture(scope="session")
def real_x63_hel1os_frames(real_x63_solexs_frames):
    """
    120 synthetic HEL1OS CDTE frames with timestamps matched 1:1 to
    real_x63_solexs_frames.  Used when the PhysicsEngine needs paired frames
    but no real HEL1OS data is available for 2024-02-22.
    """
    from aruna.models.observation import ObservationFrame, QualityFlag

    n_bins = 64
    energy = np.linspace(8.0, 70.0, n_bins + 1)
    rng    = np.random.default_rng(seed=99)

    frames = []
    for sol in real_x63_solexs_frames:
        c = rng.poisson(200, size=n_bins).astype(np.float64)
        frames.append(ObservationFrame(
            instrument          = "HEL1OS_CDTE",
            timestamp           = sol.timestamp,
            energy_bins_kev     = energy.copy(),
            counts_per_s_kev    = c,
            quality_flag        = QualityFlag.GOOD,
            dead_time_fraction  = 0.02,
            calibration_version = "sarwade_2025_v1",
            solar_distance_au   = 0.987,
            spacecraft_mode     = "SCIENCE",
            saa_flag            = False,
            source_file         = "synthetic_hel1os_for_real_data_test.fits",
            checksum_sha256     = "synthetic",
            pipeline_version    = "2.0.0",
        ))
    return frames


# ── L3 Nowcasting fixtures ────────────────────────────────────────────────────

@pytest.fixture
def test_nowcast_config():
    """Small-window NowcastConfig for offline integration tests."""
    from aruna.l3_nowcasting.models import NowcastConfig
    return NowcastConfig(
        baseline_window_sec=300,
        baseline_recompute_interval=10,
        pelt_flux_window=20,
        pelt_min_size=3,
        pelt_penalty=0.3,
        dual_trigger_window_sec=180,
        peak_decay_threshold=0.5,
        db_url="sqlite:///:memory:",
    )


@pytest.fixture
def synthetic_physics_frames():
    """
    450 PhysicsFrames simulating one M5-class flare:
      Frames   0-229  quiet Sun      (flux ~500 sol, ~100 hel)
      Frames 230-249  HOPE precursor (hope_flag=True)
      Frames 250-329  flare rise     (flux 500→50000 sol)
      Frames 330-449  decay          (exponential falloff)

    HOPE fires at frame 230. Peak flux reached at frame 329.
    3 consecutive declining frames start at 330 → FLARE_PEAK at frame 332.
    lead_time from HOPE to peak = 332 - 230 = 102s >= 90s (PRD §7.11).

    SHEI ramps at 2× speed during rise so hel_triggered fires within 30s of
    the GOES onset (frame 250), satisfying the 30s detection SLA.
    """
    import uuid
    from aruna.l2_physics.models import PhysicsFrame, PhysicsQuality
    from aruna.models.observation import QualityFlag

    base_ts = datetime(2024, 2, 22, 4, 0, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)
    quiet_sol, quiet_hel = 500.0, 100.0
    peak_sol, peak_hel   = 50000.0, 10000.0
    n_quiet, n_hope, n_rise, n_decay = 230, 20, 80, 120

    frames = []
    for i in range(n_quiet + n_hope + n_rise + n_decay):
        ts = base_ts + timedelta(seconds=i)
        if i < n_quiet:
            sol  = quiet_sol + rng.normal(0, 15)
            hel  = quiet_hel + rng.normal(0, 5)
            hope = False; hr = None; hl = None
            SHEI = -1.8; dS = 0.0; T = 8.0; EM = 1e47; DNI = 0.1
            phase = "QUIET"
        elif i < n_quiet + n_hope:
            sol  = quiet_sol * 1.1
            hel  = quiet_hel * 1.05
            hope = True
            hr   = "dT/dt>0 streak=5s | EM flat | HEL quiet | TNPSV ang=90.0°"
            hl   = 120
            SHEI = -1.6; dS = 0.02; T = 12.0; EM = 1e47; DNI = 0.3
            phase = "HOPE"
        elif i < n_quiet + n_hope + n_rise:
            frac = (i - n_quiet - n_hope) / n_rise
            sol  = quiet_sol + (peak_sol - quiet_sol) * frac
            hel  = quiet_hel + (peak_hel - quiet_hel) * frac
            hope = False; hr = None; hl = None
            # SHEI ramps at 2× speed: crosses -1.0 at frac≈0.3 (frame ~274)
            # so hel_triggered can fire within 30s of the GOES onset (frame 250)
            SHEI = -1.6 + 2.0 * frac; dS = 0.05
            T = 20.0 + frac * 5; EM = 5e48 * frac; DNI = 1.0 + frac
            phase = "IMPULSIVE"
        else:
            frac = (i - n_quiet - n_hope - n_rise) / n_decay
            sol  = peak_sol * float(np.exp(-3.0 * frac))
            hel  = peak_hel * float(np.exp(-3.0 * frac))
            hope = False; hr = None; hl = None
            SHEI = -0.6 - frac; dS = -0.03
            T = 25.0 - frac * 10; EM = 5e48 * (1 - frac * 0.8); DNI = 1.8 - frac
            phase = "DECAY"

        sol = max(sol, 1.0); hel = max(hel, 1.0)
        frames.append(PhysicsFrame(
            frame_id=str(uuid.uuid4()),
            timestamp=ts,
            source_frame_ids=["sol", "hel"],
            T_MK=float(T), T_uncertainty_MK=1.0,
            EM_cm3=float(EM), EM_uncertainty=1e46,
            fit_chi_squared=1.2,
            DNI=float(DNI),
            tnpsv_magnitude=0.1, tnpsv_angle_rad=1.57,
            SHEI=float(SHEI), dSHEI_dt=float(dS),
            HFD=float("nan"),
            QPP_period_sec=None, QPP_amplitude=None,
            FeKa_flux=0.0,
            flare_phase=phase,
            hope_flag=bool(hope),
            hope_lead_time_sec=hl,
            hope_reason=hr,
            engine_state="ACTIVE",
            data_quality_flag=QualityFlag.GOOD,
            physics_quality_flag=PhysicsQuality.FIT_OK,
            fit_backend="chianti_lookup",
            pipeline_version="2.0.0",
            solexs_3_6kev=float(sol) * 0.6,
            solexs_6_10kev=float(sol) * 0.4,
            hel1os_10_25kev=float(hel) * 0.5,
            hel1os_25_50kev=float(hel) * 0.3,
            hel1os_50_70kev=float(hel) * 0.2,
            provenance_hash=f"hash_{i:04d}",
            dT_dt=0.5 if phase in ("HOPE", "IMPULSIVE") else -0.1,
            dEM_dt=1e44 if phase == "IMPULSIVE" else 0.0,
            d2T_dt2=0.0,
        ))
    return frames
