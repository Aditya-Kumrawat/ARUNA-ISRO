"""
Shared pytest fixtures and configuration for the ARUNA test suite.
"""

import pytest
import numpy as np
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
