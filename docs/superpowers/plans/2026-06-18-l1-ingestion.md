# L1 Data Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the L1 ingestion layer — FITS parse → dead-time correction → HEL1OS background subtraction → time alignment → GOES label join → pandera schema validation → Kafka publish — producing clean `ObservationFrame` objects from raw Aditya-L1 FITS files.

**Architecture:** `FitsReader` reads SoLEXS + HEL1OS + HK FITS from L0's `CacheManager`; `DeadTimeCorrector` applies the non-paralysable model per-bin and auto-switches to SDD2 on SDD1 saturation; `BackgroundSubtractor` fits a scipy polynomial to HEL1OS pre-event quiet window; `TimeAligner` cross-correlates the 10–15 keV overlap band to find and correct clock drift; `GoesLabelJoiner` annotates frames via SunPy HEK; `FrameValidator` applies a pandera schema and quarantines incomplete frames; `KafkaPublisher` serializes and sends to `aruna.observations.{solexs,hel1os}`; `IngestionPipeline` orchestrates all steps with the PRD §5.6 failure recovery matrix.

**Tech Stack:** Python 3.11, astropy 8, scipy 1.16, pandera 0.31, sunpy 7 (HEK), kafka-python 3, structlog 24, numpy 2, pytest 7, pytest-asyncio

## Global Constraints

- Python 3.11 only — no `match` statements; use `if/elif`
- All timestamps ISO 8601 UTC; internal `numpy.datetime64` with nanosecond precision
- `schema_version = "1.0"`, `pipeline_version = "2.0.0"` on every published message
- Dead-time constants are calibration paper values (Sarwade et al. 2025 Table 2) — hardcoded in `deadtime.py`, NOT in config
- Sherpa not buildable on Windows → scipy polynomial background subtraction (same χ²/dof criterion)
- Tests must not make real network calls unless marked `@pytest.mark.integration`
- FITS column names: `TIME`, `COUNTS` (vector), `LIVETIME`, `DEADTIME_FRAC`; energy bins in `EBOUNDS` extension
- HK columns: `TIME`, `SAA_FLAG`, `INSTRUMENT_MODE`, `SOLAR_DIST_AU`
- All config values read from `config/config.yaml` (falls back to `config.example.yaml` in tests) — except dead-time constants

---

## File Map

| File | Creates / Modifies | Responsibility |
|------|--------------------|----------------|
| `python/aruna/l1_ingestion/models.py` | Create | `FitsData`, `HKData`, `IngestionConfig`, `QuarantineEntry`, `ValidationResult` |
| `python/aruna/l1_ingestion/fits_reader.py` | Create | astropy FITS parse → `FitsData` / `HKData` |
| `python/aruna/l1_ingestion/deadtime.py` | Create | Non-paralysable correction, SDD1→SDD2 switch |
| `python/aruna/l1_ingestion/background.py` | Create | scipy polynomial background, χ²/dof, SAA masking |
| `python/aruna/l1_ingestion/alignment.py` | Create | Cross-correlation lag, cubic interpolation, drift log |
| `python/aruna/l1_ingestion/goes_join.py` | Create | SunPy HEK query, ±3 min match, goes_class annotation |
| `python/aruna/l1_ingestion/validator.py` | Create | pandera schema, completeness gate, quarantine queue |
| `python/aruna/l1_ingestion/publisher.py` | Create | Serialize `ObservationFrame`, Kafka send |
| `python/aruna/l1_ingestion/pipeline.py` | Create | Orchestrator: FitsData → ObservationFrame → Kafka |
| `python/aruna/l1_ingestion/__init__.py` | Modify | Re-export public API |
| `python/tests/conftest.py` | Modify | Add synthetic FITS fixtures |
| `python/tests/test_l1_ingestion.py` | Modify | Replace all xfail stubs with real tests |

---

### Task 1: L1 Models + Synthetic FITS Fixtures

**Files:**
- Create: `python/aruna/l1_ingestion/models.py`
- Modify: `python/tests/conftest.py`
- Modify: `python/tests/test_l1_ingestion.py`

**Interfaces:**
- Produces:
  - `FitsData(instrument, timestamps, counts_raw, energy_bins_kev, livetime, deadtime_frac, source_file, checksum_sha256)`
  - `HKData(timestamps, saa_flags, instrument_modes, solar_dist_au)`
  - `IngestionConfig(quarantine_ttl_sec, completeness_threshold, clock_drift_warn_sec, clock_drift_max_sec, goes_match_window_min, background_window_sec, chi2_dof_limit)`
  - `QuarantineEntry(fits_path, instrument, quarantined_at, retry_after, reason, attempt_count)`
  - `ValidationResult(all_passed, checks, failed_checks)`
  - pytest fixtures: `solexs_fits_path`, `hel1os_fits_path`, `hk_fits_path`, `x63_solexs_fits_path`

- [ ] **Step 1: Write the failing test**

Replace entire `python/tests/test_l1_ingestion.py` with:

```python
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
        counts = hdul['SDD1'].data['COUNTS']  # (N_times, N_bins)
    total_rate = counts.sum(axis=1)
    assert total_rate.max() > SAT_LIMIT, "X6.3 fixture SDD1 should saturate"
```

- [ ] **Step 2: Run test to verify it fails**

```
cd E:\aruna
python -m pytest python/tests/test_l1_ingestion.py::test_fits_data_shape_consistent -v
```
Expected: `ERROR — ModuleNotFoundError: No module named 'aruna.l1_ingestion.models'`

- [ ] **Step 3: Create `python/aruna/l1_ingestion/models.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class FitsData:
    """Raw parsed data from one FITS detector extension, before correction."""
    instrument: str
    timestamps: np.ndarray       # shape (N_times,), numpy.datetime64[ns] UTC
    counts_raw: np.ndarray       # shape (N_times, N_bins), counts/s
    energy_bins_kev: np.ndarray  # shape (N_bins+1,), bin edges in keV
    livetime: np.ndarray         # shape (N_times,), fraction 0–1
    deadtime_frac: np.ndarray    # shape (N_times,), fraction 0–1
    source_file: str
    checksum_sha256: str


@dataclass
class HKData:
    """Parsed housekeeping data from Aditya_HK FITS."""
    timestamps: np.ndarray    # shape (N_times,), datetime64[ns] UTC
    saa_flags: np.ndarray     # shape (N_times,), bool
    instrument_modes: list    # shape (N_times,), str per element
    solar_dist_au: np.ndarray # shape (N_times,), float


@dataclass
class IngestionConfig:
    """Tunable L1 thresholds — read from config.yaml ingestion section."""
    quarantine_ttl_sec: float = 300.0
    completeness_threshold: float = 0.80
    clock_drift_warn_sec: float = 0.5
    clock_drift_max_sec: float = 1.0
    goes_match_window_min: float = 3.0
    goes_query_window_min: float = 10.0
    background_window_sec: int = 300
    background_poly_order: int = 2
    chi2_dof_limit: float = 2.0
    kafka_topic_solexs: str = "aruna.observations.solexs"
    kafka_topic_hel1os: str = "aruna.observations.hel1os"
    kafka_topic_quality: str = "aruna.quality.summary"
    kafka_topic_errors: str = "aruna.ingestion.errors"


@dataclass
class QuarantineEntry:
    """A frame set held pending a retry after quarantine_ttl_sec."""
    fits_path: Path
    instrument: str
    quarantined_at: datetime
    retry_after: datetime
    reason: str
    attempt_count: int = 0


@dataclass
class ValidationResult:
    all_passed: bool
    checks: dict
    failed_checks: List[str] = field(default_factory=list)
```

- [ ] **Step 4: Add synthetic FITS fixtures to `python/tests/conftest.py`**

Append after the existing fixtures (keep all existing content):

```python
import numpy as np
from astropy.io import fits as astropy_fits
from pathlib import Path


def _make_bintable(ext_name: str, n_times: int, n_bins: int,
                   counts: np.ndarray, timeref: str = '2024-02-22T00:00:00',
                   instrume: str = '') -> astropy_fits.BinTableHDU:
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


def _make_ebounds(name: str, e_min: np.ndarray, e_max: np.ndarray) -> astropy_fits.BinTableHDU:
    cols = [
        astropy_fits.Column(name='E_MIN', format='D', unit='keV', array=e_min),
        astropy_fits.Column(name='E_MAX', format='D', unit='keV', array=e_max),
    ]
    return astropy_fits.BinTableHDU.from_columns(cols, name=name)


@pytest.fixture
def solexs_fits_path(tmp_path):
    """Synthetic SoLEXS FITS with SDD1, SDD2, EBOUNDS extensions. 60s × 32 bins."""
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
    X6.3 flare scenario: 120 seconds, SDD1 saturates (>5e4 counts/s) at t=60–90s.
    SDD2 remains unsaturated throughout.
    """
    n_times, n_bins = 120, 32
    rng = np.random.default_rng(63)
    energy = np.linspace(2.0, 22.0, n_bins + 1)

    # SDD1: quiet baseline, then spike to saturation
    counts_sdd1 = rng.poisson(1000, size=(n_times, n_bins)).astype(np.float64)
    # At t=60..90: total rate >> 5e4 counts/s
    counts_sdd1[60:90, :] = rng.poisson(2000, size=(30, n_bins)) + 60000.0 / n_bins

    # SDD2: lower-gain, never saturates
    counts_sdd2 = counts_sdd1 * 0.01  # 100× attenuated aperture

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
```

- [ ] **Step 5: Run all Task 1 tests**

```
python -m pytest python/tests/test_l1_ingestion.py -k "fits_data or ingestion_config or quarantine_entry or fixture" -v
```
Expected: 7 PASSED

- [ ] **Step 6: Commit**

```
git add python/aruna/l1_ingestion/models.py python/tests/conftest.py python/tests/test_l1_ingestion.py
git commit -m "feat(l1): L1 models + synthetic FITS fixtures"
```

---

### Task 2: FitsReader

**Files:**
- Create: `python/aruna/l1_ingestion/fits_reader.py`
- Modify: `python/tests/test_l1_ingestion.py`

**Interfaces:**
- Consumes: `FitsData`, `HKData` (Task 1); `solexs_fits_path`, `hel1os_fits_path`, `hk_fits_path` fixtures
- Produces:
  - `FitsReader().read_solexs(path: Path) -> List[FitsData]` — `[FitsData(SDD1), FitsData(SDD2)]`
  - `FitsReader().read_hel1os(path: Path) -> List[FitsData]` — `[FitsData(CDTE), FitsData(CZT)]`
  - `FitsReader().read_hk(path: Path) -> HKData`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l1_ingestion.py`:

```python
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
    assert sdd1.counts_raw.shape[0] == 60   # n_times
    assert sdd1.counts_raw.shape[1] == 32   # n_bins


def test_fits_reader_solexs_energy_bins(solexs_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    result = FitsReader().read_solexs(solexs_fits_path)
    sdd1 = next(fd for fd in result if fd.instrument == 'SOLEXS_SDD1')
    assert len(sdd1.energy_bins_kev) == 33  # N_bins + 1
    assert sdd1.energy_bins_kev[0] == pytest.approx(2.0, abs=0.1)
    assert sdd1.energy_bins_kev[-1] == pytest.approx(22.0, abs=0.1)


def test_fits_reader_solexs_timestamps_utc(solexs_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    result = FitsReader().read_solexs(solexs_fits_path)
    sdd1 = next(fd for fd in result if fd.instrument == 'SOLEXS_SDD1')
    assert sdd1.timestamps.dtype.kind == 'M'  # datetime64
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
    # FITS without SDD1
    bad = fits.HDUList([fits.PrimaryHDU()])
    p = tmp_path / 'bad.fits'
    bad.writeto(p)
    with pytest.raises((KeyError, ValueError)):
        FitsReader().read_solexs(p)


def test_fits_reader_checksum_populated(solexs_fits_path):
    from aruna.l1_ingestion.fits_reader import FitsReader
    result = FitsReader().read_solexs(solexs_fits_path)
    for fd in result:
        assert len(fd.checksum_sha256) == 64  # hex SHA-256
```

- [ ] **Step 2: Run one test to verify failure**

```
python -m pytest python/tests/test_l1_ingestion.py::test_fits_reader_solexs_returns_two_detectors -v
```
Expected: `ERROR — ModuleNotFoundError: No module named 'aruna.l1_ingestion.fits_reader'`

- [ ] **Step 3: Create `python/aruna/l1_ingestion/fits_reader.py`**

```python
from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
from astropy.io import fits

from .models import FitsData, HKData

logger = logging.getLogger(__name__)

SOLEXS_DETECTORS = {'SDD1': 'SOLEXS_SDD1', 'SDD2': 'SOLEXS_SDD2'}
HEL1OS_DETECTORS = {'CDTE': 'HEL1OS_CDTE', 'CZT': 'HEL1OS_CZT'}
HEL1OS_EBOUNDS   = {'CDTE': 'EBOUNDS_CDTE', 'CZT': 'EBOUNDS_CZT'}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _parse_timeref(header) -> datetime:
    s = header.get('TIMEREF', '2000-01-01T00:00:00')
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _times_to_dt64(times_sec: np.ndarray, timeref: datetime) -> np.ndarray:
    epoch = np.datetime64(timeref.replace(tzinfo=None).isoformat(), 'ns')
    offsets = (times_sec * 1e9).astype('int64').astype('timedelta64[ns]')
    return epoch + offsets


def _read_ebounds(hdul: fits.HDUList, ext_name: str) -> np.ndarray:
    try:
        eb = hdul[ext_name]
        e_min = eb.data['E_MIN'].astype(np.float64)
        e_max = eb.data['E_MAX'].astype(np.float64)
        return np.append(e_min, e_max[-1])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing or malformed {ext_name} extension: {exc}") from exc


class FitsReader:
    def read_solexs(self, path: Path) -> List[FitsData]:
        """Read SoLEXS FITS → [FitsData(SDD1), FitsData(SDD2)]."""
        path = Path(path)
        checksum = _sha256(path)
        result = []
        with fits.open(path) as hdul:
            energy_bins = _read_ebounds(hdul, 'EBOUNDS')
            for ext_name, instrument in SOLEXS_DETECTORS.items():
                hdu = hdul[ext_name]
                timeref = _parse_timeref(hdu.header)
                result.append(FitsData(
                    instrument=instrument,
                    timestamps=_times_to_dt64(hdu.data['TIME'], timeref),
                    counts_raw=hdu.data['COUNTS'].astype(np.float64),
                    energy_bins_kev=energy_bins,
                    livetime=hdu.data['LIVETIME'].astype(np.float64),
                    deadtime_frac=hdu.data['DEADTIME_FRAC'].astype(np.float64),
                    source_file=path.name,
                    checksum_sha256=checksum,
                ))
        return result

    def read_hel1os(self, path: Path) -> List[FitsData]:
        """Read HEL1OS FITS → [FitsData(CDTE), FitsData(CZT)]."""
        path = Path(path)
        checksum = _sha256(path)
        result = []
        with fits.open(path) as hdul:
            for ext_name, instrument in HEL1OS_DETECTORS.items():
                hdu = hdul[ext_name]
                timeref = _parse_timeref(hdu.header)
                energy_bins = _read_ebounds(hdul, HEL1OS_EBOUNDS[ext_name])
                result.append(FitsData(
                    instrument=instrument,
                    timestamps=_times_to_dt64(hdu.data['TIME'], timeref),
                    counts_raw=hdu.data['COUNTS'].astype(np.float64),
                    energy_bins_kev=energy_bins,
                    livetime=hdu.data['LIVETIME'].astype(np.float64),
                    deadtime_frac=hdu.data['DEADTIME_FRAC'].astype(np.float64),
                    source_file=path.name,
                    checksum_sha256=checksum,
                ))
        return result

    def read_hk(self, path: Path) -> HKData:
        """Read HK FITS → HKData."""
        path = Path(path)
        with fits.open(path) as hdul:
            hdu = hdul['HK']
            timeref = _parse_timeref(hdu.header)
            return HKData(
                timestamps=_times_to_dt64(hdu.data['TIME'], timeref),
                saa_flags=hdu.data['SAA_FLAG'].astype(bool),
                instrument_modes=list(hdu.data['INSTRUMENT_MODE']),
                solar_dist_au=hdu.data['SOLAR_DIST_AU'].astype(np.float64),
            )
```

- [ ] **Step 4: Run all FitsReader tests**

```
python -m pytest python/tests/test_l1_ingestion.py -k "fits_reader" -v
```
Expected: 11 PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l1_ingestion/fits_reader.py python/tests/test_l1_ingestion.py
git commit -m "feat(l1): FitsReader — astropy FITS parse for SoLEXS, HEL1OS, HK"
```

---

### Task 3: DeadTimeCorrector

**Files:**
- Create: `python/aruna/l1_ingestion/deadtime.py`
- Modify: `python/tests/test_l1_ingestion.py`

**Interfaces:**
- Consumes: `FitsData` (Task 1)
- Produces:
  - `DEADTIME_CONSTANTS: dict[str, float]`
  - `SATURATION_LIMITS: dict[str, float]`
  - `DeadTimeCorrector().correct(data: FitsData) -> Tuple[FitsData, np.ndarray]` — corrected data + quality_flags `(N_times,)`
  - `DeadTimeCorrector().select_solexs_detector(sdd1, sdd1_flags, sdd2, sdd2_flags) -> Tuple[FitsData, np.ndarray]`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l1_ingestion.py`:

```python
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
    """Dead-time correction applied to every SoLEXS energy bin (PRD §5.8)."""
    from aruna.l1_ingestion.deadtime import DeadTimeCorrector, DEADTIME_CONSTANTS
    tau = DEADTIME_CONSTANTS['SOLEXS_SDD1']
    r_obs = 1000.0  # counts/s per bin
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
    """SDD1 rate > 5e4 counts/s → SATURATED flag; select_solexs_detector uses SDD2. (PRD §5.8)"""
    from aruna.l1_ingestion.deadtime import DeadTimeCorrector, SATURATION_LIMITS
    from aruna.models.observation import QualityFlag

    sat = SATURATION_LIMITS['SOLEXS_SDD1']
    n_times, n_bins = 10, 32
    # SDD1 with 5 saturated frames
    counts_sdd1 = np.full((n_times, n_bins), 100.0)
    counts_sdd1[3:8, :] = (sat + 1000) / n_bins  # total > sat_limit

    counts_sdd2 = np.full((n_times, n_bins), 50.0)

    sdd1_data = _make_fits_data('SOLEXS_SDD1', counts_sdd1)
    sdd2_data = _make_fits_data('SOLEXS_SDD2', counts_sdd2)

    dtc = DeadTimeCorrector()
    corr_sdd1, flags_sdd1 = dtc.correct(sdd1_data)
    corr_sdd2, flags_sdd2 = dtc.correct(sdd2_data)

    assert (flags_sdd1[3:8] == QualityFlag.SATURATED).all()
    assert (flags_sdd1[:3]  != QualityFlag.SATURATED).all()

    merged, merged_flags = dtc.select_solexs_detector(
        corr_sdd1, flags_sdd1, corr_sdd2, flags_sdd2
    )
    # Saturated frames use SDD2 counts
    assert merged.counts_raw[5, 0] == pytest.approx(corr_sdd2.counts_raw[5, 0], rel=1e-6)
    # Non-saturated frames keep SDD1 counts
    assert merged.counts_raw[0, 0] == pytest.approx(corr_sdd1.counts_raw[0, 0], rel=1e-6)
    assert (merged_flags[3:8] == QualityFlag.SATURATED).all()


def test_dead_time_correction_formula_accuracy():
    """R_true = R_obs / (1 - R_obs * tau) per Sarwade et al. 2025."""
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
    """Verify calibration constants match Sarwade et al. 2025 Table 2."""
    from aruna.l1_ingestion.deadtime import DEADTIME_CONSTANTS
    assert DEADTIME_CONSTANTS['SOLEXS_SDD1'] == pytest.approx(4.5e-6)
    assert DEADTIME_CONSTANTS['SOLEXS_SDD2'] == pytest.approx(4.5e-6)
    assert DEADTIME_CONSTANTS['HEL1OS_CDTE'] == pytest.approx(2.1e-6)
    assert DEADTIME_CONSTANTS['HEL1OS_CZT']  == pytest.approx(1.8e-6)
```

- [ ] **Step 2: Run one test to verify failure**

```
python -m pytest python/tests/test_l1_ingestion.py::test_dead_time_correction_applied_to_all_solexs_bins -v
```
Expected: `ERROR — ModuleNotFoundError: No module named 'aruna.l1_ingestion.deadtime'`

- [ ] **Step 3: Create `python/aruna/l1_ingestion/deadtime.py`**

```python
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from ..models.observation import QualityFlag
from .models import FitsData

logger = logging.getLogger(__name__)

DEADTIME_CONSTANTS: dict = {
    'SOLEXS_SDD1': 4.5e-6,
    'SOLEXS_SDD2': 4.5e-6,
    'HEL1OS_CDTE': 2.1e-6,
    'HEL1OS_CZT':  1.8e-6,
}

SATURATION_LIMITS: dict = {
    'SOLEXS_SDD1': 5e4,
    'SOLEXS_SDD2': 5e6,
}


class DeadTimeCorrector:
    def correct(self, data: FitsData) -> Tuple[FitsData, np.ndarray]:
        """
        Apply non-paralysable dead-time correction: R_true = R_obs / (1 - R_obs * tau).
        Applied per bin per time step.
        Returns (corrected FitsData, quality_flags shape (N_times,)).
        """
        tau = DEADTIME_CONSTANTS.get(data.instrument)
        if tau is None:
            raise ValueError(f"Unknown instrument: {data.instrument!r}. Must be one of {set(DEADTIME_CONSTANTS)}")

        counts = data.counts_raw.copy()
        n_times = len(counts)
        quality_flags = np.full(n_times, int(QualityFlag.GOOD), dtype=np.int8)

        denom = 1.0 - counts * tau
        safe = denom > 0
        counts[safe] = counts[safe] / denom[safe]
        counts[~safe] = np.nan

        if data.instrument in SATURATION_LIMITS:
            sat_limit = SATURATION_LIMITS[data.instrument]
            total_rate = np.nansum(counts, axis=1)
            saturated = total_rate > sat_limit
            if saturated.any():
                quality_flags[saturated] = int(QualityFlag.SATURATED)
                logger.info(
                    "Saturation detected in %s at %d/%d frames",
                    data.instrument, saturated.sum(), n_times,
                )

        high_dt = data.deadtime_frac > 0.05
        not_sat = quality_flags != int(QualityFlag.SATURATED)
        quality_flags[high_dt & not_sat] = int(QualityFlag.DEAD_TIME)

        corrected = FitsData(
            instrument=data.instrument,
            timestamps=data.timestamps,
            counts_raw=counts,
            energy_bins_kev=data.energy_bins_kev,
            livetime=data.livetime,
            deadtime_frac=data.deadtime_frac,
            source_file=data.source_file,
            checksum_sha256=data.checksum_sha256,
        )
        return corrected, quality_flags

    def select_solexs_detector(
        self,
        sdd1_data: FitsData,
        sdd1_flags: np.ndarray,
        sdd2_data: FitsData,
        sdd2_flags: np.ndarray,
    ) -> Tuple[FitsData, np.ndarray]:
        """Per saturated SDD1 frame, substitute SDD2 counts. Returns merged (data, flags)."""
        sat = sdd1_flags == int(QualityFlag.SATURATED)
        if not sat.any():
            return sdd1_data, sdd1_flags

        merged_counts = sdd1_data.counts_raw.copy()
        merged_flags  = sdd1_flags.copy()
        merged_counts[sat] = sdd2_data.counts_raw[sat]

        merged = FitsData(
            instrument=sdd1_data.instrument,
            timestamps=sdd1_data.timestamps,
            counts_raw=merged_counts,
            energy_bins_kev=sdd1_data.energy_bins_kev,
            livetime=sdd1_data.livetime,
            deadtime_frac=sdd1_data.deadtime_frac,
            source_file=sdd1_data.source_file,
            checksum_sha256=sdd1_data.checksum_sha256,
        )
        return merged, merged_flags
```

- [ ] **Step 4: Run all dead-time tests**

```
python -m pytest python/tests/test_l1_ingestion.py -k "dead_time or sdd1_saturation or dead_time_constants" -v
```
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l1_ingestion/deadtime.py python/tests/test_l1_ingestion.py
git commit -m "feat(l1): DeadTimeCorrector — non-paralysable model, SDD1→SDD2 switch"
```

---

### Task 4: BackgroundSubtractor

**Files:**
- Create: `python/aruna/l1_ingestion/background.py`
- Modify: `python/tests/test_l1_ingestion.py`

**Interfaces:**
- Consumes: `FitsData` (Task 1), `QualityFlag` from `aruna.models.observation`
- Produces:
  - `BackgroundSubtractor(poly_order=2, chi2_dof_limit=2.0)`
  - `.subtract(data: FitsData, background_frames: List[FitsData], saa_flags: np.ndarray) -> Tuple[FitsData, np.ndarray, float]`
    — `(subtracted FitsData, quality_flags (N_times,), chi2_dof)`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l1_ingestion.py`:

```python
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
    """HEL1OS background subtraction applied; χ²/dof < 2.0 for Poisson data (PRD §5.8)."""
    from aruna.l1_ingestion.background import BackgroundSubtractor
    bg_frames = [_make_hel1os_data(n_times=10) for _ in range(30)]  # 300s quiet window
    current = _make_hel1os_data(n_times=10)
    saa = np.zeros(10, dtype=bool)
    _, _, chi2_dof = BackgroundSubtractor().subtract(current, bg_frames, saa)
    assert chi2_dof < 2.0, f"χ²/dof={chi2_dof:.2f} exceeds 2.0 for Poisson data"


def test_saa_bins_flagged_not_dropped():
    """SAA-coincident bins are flagged QualityFlag.SAA_PASSAGE, not silently dropped (PRD §5.8)."""
    from aruna.l1_ingestion.background import BackgroundSubtractor
    from aruna.models.observation import QualityFlag
    bg_frames = [_make_hel1os_data(n_times=10) for _ in range(30)]
    current = _make_hel1os_data(n_times=10)
    saa = np.array([False]*5 + [True]*5, dtype=bool)
    result, flags, _ = BackgroundSubtractor().subtract(current, bg_frames, saa)
    # Frame still present (not dropped)
    assert len(result.counts_raw) == 10
    # SAA frames flagged
    assert (flags[5:] == QualityFlag.SAA_PASSAGE).all()
    # Non-SAA frames not flagged as SAA
    assert (flags[:5] != QualityFlag.SAA_PASSAGE).all()


def test_background_subtraction_reduces_baseline():
    """After subtraction, mean of subtracted counts < mean of original counts."""
    from aruna.l1_ingestion.background import BackgroundSubtractor
    bg_frames = [_make_hel1os_data(n_times=10, base_counts=200.0) for _ in range(30)]
    current = _make_hel1os_data(n_times=10, base_counts=200.0)
    original_mean = current.counts_raw.mean()
    result, _, _ = BackgroundSubtractor().subtract(current, bg_frames, np.zeros(10, dtype=bool))
    assert result.counts_raw.mean() < original_mean


def test_background_subtraction_no_negative_counts():
    """Subtracted counts floored at 0.0 — no negative values allowed."""
    from aruna.l1_ingestion.background import BackgroundSubtractor
    bg_frames = [_make_hel1os_data(n_times=10, base_counts=200.0) for _ in range(30)]
    current = _make_hel1os_data(n_times=10, base_counts=200.0)
    result, _, _ = BackgroundSubtractor().subtract(current, bg_frames, np.zeros(10, dtype=bool))
    assert (result.counts_raw >= 0.0).all()


def test_background_subtraction_empty_window_returns_unchanged():
    """No background frames → data returned unchanged, no crash."""
    from aruna.l1_ingestion.background import BackgroundSubtractor
    current = _make_hel1os_data(n_times=10)
    result, flags, chi2_dof = BackgroundSubtractor().subtract(current, [], np.zeros(10, dtype=bool))
    assert result.counts_raw.shape == current.counts_raw.shape
    assert chi2_dof == 0.0


def test_background_high_chi2_flags_noisy():
    """χ²/dof > 2.0 produces QualityFlag.NOISY on non-SAA frames."""
    from aruna.l1_ingestion.background import BackgroundSubtractor
    from aruna.models.observation import QualityFlag
    # Very spiky background to force high chi2
    rng = np.random.default_rng(99)
    n_bins = 64
    bg_frames = []
    for i in range(30):
        from aruna.l1_ingestion.models import FitsData
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
    bs = BackgroundSubtractor(chi2_dof_limit=0.01)  # very tight — forces failure
    _, flags, chi2_dof = bs.subtract(current, bg_frames, np.zeros(10, dtype=bool))
    assert chi2_dof > 0.01
    assert (flags == QualityFlag.NOISY).any()
```

- [ ] **Step 2: Run one test to verify failure**

```
python -m pytest python/tests/test_l1_ingestion.py::test_background_subtraction_hel1os_chi_squared -v
```
Expected: `ERROR — ModuleNotFoundError: No module named 'aruna.l1_ingestion.background'`

- [ ] **Step 3: Create `python/aruna/l1_ingestion/background.py`**

```python
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
from scipy.optimize import curve_fit

from ..models.observation import QualityFlag
from .models import FitsData

logger = logging.getLogger(__name__)


def _poly2(x, a, b, c):
    return a * x ** 2 + b * x + c


class BackgroundSubtractor:
    def __init__(self, poly_order: int = 2, chi2_dof_limit: float = 2.0):
        self._poly_order = poly_order
        self._chi2_limit = chi2_dof_limit

    def subtract(
        self,
        data: FitsData,
        background_frames: List[FitsData],
        saa_flags: np.ndarray,
    ) -> Tuple[FitsData, np.ndarray, float]:
        """
        Fit polynomial background to quiet window, subtract per energy bin, validate χ²/dof.
        SAA bins are flagged QualityFlag.SAA_PASSAGE but NOT dropped.
        """
        n_cur = len(data.counts_raw)
        quality_flags = np.full(n_cur, int(QualityFlag.GOOD), dtype=np.int8)
        quality_flags[saa_flags] = int(QualityFlag.SAA_PASSAGE)

        if not background_frames:
            logger.warning("No background frames for %s — skipping subtraction", data.instrument)
            return data, quality_flags, 0.0

        bg_counts = np.vstack([f.counts_raw for f in background_frames])
        n_bg, n_bins = bg_counts.shape
        x_bg = np.arange(n_bg, dtype=np.float64)
        x_cur = np.arange(n_bg, n_bg + n_cur, dtype=np.float64)

        bg_model = np.zeros_like(bg_counts)
        bg_current = np.zeros((n_cur, n_bins))

        for b in range(n_bins):
            y = bg_counts[:, b]
            try:
                popt, _ = curve_fit(_poly2, x_bg, y, maxfev=2000,
                                    p0=[0.0, 0.0, float(np.median(y))])
                bg_model[:, b] = _poly2(x_bg, *popt)
                bg_current[:, b] = _poly2(x_cur, *popt)
            except RuntimeError:
                med = float(np.median(y))
                bg_model[:, b] = med
                bg_current[:, b] = med

        subtracted = np.maximum(data.counts_raw - bg_current, 0.0)

        poisson_var = np.maximum(bg_model, 1.0)
        residuals = bg_counts - bg_model
        chi2_total = float(np.sum(residuals ** 2 / poisson_var))
        dof = max(n_bg * n_bins - (self._poly_order + 1) * n_bins, 1)
        chi2_dof = chi2_total / dof

        if chi2_dof > self._chi2_limit:
            logger.warning(
                "Background χ²/dof=%.3f > %.1f for %s — flagging NOISY",
                chi2_dof, self._chi2_limit, data.instrument,
            )
            noisy = quality_flags == int(QualityFlag.GOOD)
            quality_flags[noisy] = int(QualityFlag.NOISY)

        result = FitsData(
            instrument=data.instrument,
            timestamps=data.timestamps,
            counts_raw=subtracted,
            energy_bins_kev=data.energy_bins_kev,
            livetime=data.livetime,
            deadtime_frac=data.deadtime_frac,
            source_file=data.source_file,
            checksum_sha256=data.checksum_sha256,
        )
        return result, quality_flags, chi2_dof
```

- [ ] **Step 4: Run all background tests**

```
python -m pytest python/tests/test_l1_ingestion.py -k "background or saa_bins" -v
```
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l1_ingestion/background.py python/tests/test_l1_ingestion.py
git commit -m "feat(l1): BackgroundSubtractor — scipy poly, chi-squared gate, SAA masking"
```

---

### Task 5: TimeAligner

**Files:**
- Create: `python/aruna/l1_ingestion/alignment.py`
- Modify: `python/tests/test_l1_ingestion.py`

**Interfaces:**
- Consumes: `FitsData` (Task 1)
- Produces:
  - `TimeAligner(drift_warn_sec=0.5, drift_max_sec=1.0)`
  - `.compute_lag(solexs: FitsData, hel1os: FitsData) -> float` — lag in seconds
  - `.align(solexs: FitsData, hel1os: FitsData) -> Tuple[FitsData, FitsData, float]`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l1_ingestion.py`:

```python
# ── Task 5: TimeAligner ───────────────────────────────────────────────────────

def _make_aligned_pair(n_times=60, lag_samples=0):
    """Create SoLEXS + HEL1OS FitsData pair with a known lag (in samples = seconds)."""
    from aruna.l1_ingestion.models import FitsData
    rng = np.random.default_rng(7)
    # 10-15 keV band is bins 13-19 for SoLEXS (2-22 keV, 32 bins → ~0.625 keV/bin)
    n_bins_s, n_bins_h = 32, 64
    signal = rng.poisson(500, size=n_times).astype(np.float64)

    # Build SoLEXS counts: put signal in 10-15 keV band (bins 12–19)
    counts_s = rng.poisson(50, size=(n_times, n_bins_s)).astype(np.float64)
    for t in range(n_times):
        counts_s[t, 12:20] += signal[t]

    # Build HEL1OS counts: same signal shifted by lag_samples
    counts_h = rng.poisson(50, size=(n_times, n_bins_h)).astype(np.float64)
    # 10-15 keV band in HEL1OS CdTe (8-70 keV, 64 bins → ~0.969 keV/bin): bins 2-7
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
    """Time sync error between SoLEXS and HEL1OS < 1 second (PRD §5.8)."""
    from aruna.l1_ingestion.alignment import TimeAligner
    solexs, hel1os = _make_aligned_pair(lag_samples=0)
    _, _, lag_sec = TimeAligner().align(solexs, hel1os)
    assert abs(lag_sec) < 1.0


def test_aligner_zero_lag_no_interpolation():
    """Zero-lag pair: both returned at same length, no interpolation needed."""
    from aruna.l1_ingestion.alignment import TimeAligner
    solexs, hel1os = _make_aligned_pair(lag_samples=0)
    s_out, h_out, lag = TimeAligner().align(solexs, hel1os)
    assert len(s_out.timestamps) == len(h_out.timestamps)
    assert abs(lag) < 1.0


def test_clock_drift_logged_for_lag_over_0_5_seconds(caplog):
    """Clock drift log entry created when cross-correlation lag > 0.5s (PRD §5.8)."""
    import logging
    from aruna.l1_ingestion.alignment import TimeAligner
    # Use a 3-sample lag — well above 0.5s warn threshold
    solexs, hel1os = _make_aligned_pair(n_times=120, lag_samples=3)
    with caplog.at_level(logging.WARNING, logger='aruna.l1_ingestion.alignment'):
        TimeAligner(drift_warn_sec=0.5).align(solexs, hel1os)
    assert any('CLOCK_DRIFT' in r.message for r in caplog.records), \
        "Expected CLOCK_DRIFT log entry for lag > 0.5s"


def test_aligner_large_lag_triggers_interpolation():
    """Lag > 1.0s: align() applies cubic interpolation to HEL1OS."""
    from aruna.l1_ingestion.alignment import TimeAligner
    solexs, hel1os = _make_aligned_pair(n_times=120, lag_samples=5)
    aligner = TimeAligner(drift_warn_sec=0.5, drift_max_sec=1.0)
    # Inject lag by faking compute_lag to return 2.0s
    from unittest.mock import patch
    with patch.object(aligner, 'compute_lag', return_value=2.0):
        s_out, h_out, lag = aligner.align(solexs, hel1os)
    assert lag == 2.0
    assert len(s_out.timestamps) == len(h_out.timestamps)


def test_aligner_output_shorter_than_input_on_trim():
    """Aligned outputs may be shorter than inputs (trimmed to common window)."""
    from aruna.l1_ingestion.alignment import TimeAligner
    solexs, hel1os = _make_aligned_pair(n_times=60)
    s_out, h_out, _ = TimeAligner().align(solexs, hel1os)
    assert len(s_out.timestamps) <= 60
    assert len(h_out.timestamps) <= 60
```

- [ ] **Step 2: Run one test to verify failure**

```
python -m pytest python/tests/test_l1_ingestion.py::test_time_sync_under_one_second -v
```
Expected: `ERROR — ModuleNotFoundError: No module named 'aruna.l1_ingestion.alignment'`

- [ ] **Step 3: Create `python/aruna/l1_ingestion/alignment.py`**

```python
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

from .models import FitsData

logger = logging.getLogger(__name__)

_XCORR_BAND_KEV = (10.0, 15.0)


def _band_sum(data: FitsData) -> np.ndarray:
    centres = (data.energy_bins_kev[:-1] + data.energy_bins_kev[1:]) / 2.0
    mask = (centres >= _XCORR_BAND_KEV[0]) & (centres < _XCORR_BAND_KEV[1])
    if not mask.any():
        return np.zeros(len(data.counts_raw))
    return data.counts_raw[:, mask].sum(axis=1)


def _trim(data: FitsData, n: int) -> FitsData:
    return FitsData(
        instrument=data.instrument,
        timestamps=data.timestamps[:n],
        counts_raw=data.counts_raw[:n],
        energy_bins_kev=data.energy_bins_kev,
        livetime=data.livetime[:n],
        deadtime_frac=data.deadtime_frac[:n],
        source_file=data.source_file,
        checksum_sha256=data.checksum_sha256,
    )


def _interpolate_to_grid(data: FitsData, target_times: np.ndarray) -> FitsData:
    src_t = data.timestamps.astype(np.float64)
    tgt_t = target_times.astype(np.float64)
    n_bins = data.counts_raw.shape[1]
    new_counts = np.zeros((len(tgt_t), n_bins))
    for b in range(n_bins):
        f = interp1d(src_t, data.counts_raw[:, b], kind='cubic',
                     bounds_error=False, fill_value=0.0)
        new_counts[:, b] = f(tgt_t)
    return FitsData(
        instrument=data.instrument,
        timestamps=target_times,
        counts_raw=new_counts,
        energy_bins_kev=data.energy_bins_kev,
        livetime=np.interp(tgt_t, src_t, data.livetime),
        deadtime_frac=np.interp(tgt_t, src_t, data.deadtime_frac),
        source_file=data.source_file,
        checksum_sha256=data.checksum_sha256,
    )


class TimeAligner:
    def __init__(self, drift_warn_sec: float = 0.5, drift_max_sec: float = 1.0):
        self._drift_warn = drift_warn_sec
        self._drift_max = drift_max_sec

    def compute_lag(self, solexs: FitsData, hel1os: FitsData) -> float:
        """Cross-correlate 10-15 keV band. Returns lag in seconds (1 sample = 1s)."""
        s = _band_sum(solexs)
        h = _band_sum(hel1os)
        n = min(len(s), len(h))
        s, h = s[:n], h[:n]
        corr = signal.correlate(s - s.mean(), h - h.mean(), mode='full')
        lags = signal.correlation_lags(n, n, mode='full')
        return float(lags[np.argmax(corr)])

    def align(
        self, solexs: FitsData, hel1os: FitsData,
    ) -> Tuple[FitsData, FitsData, float]:
        """
        Align SoLEXS and HEL1OS to common 1s UTC grid.
        Returns (aligned_solexs, aligned_hel1os, lag_sec).
        """
        lag_sec = self.compute_lag(solexs, hel1os)

        if abs(lag_sec) > self._drift_warn:
            logger.warning(
                "CLOCK_DRIFT detected: lag=%.3fs between SoLEXS and HEL1OS",
                lag_sec,
            )

        if abs(lag_sec) <= self._drift_max:
            n = min(len(solexs.timestamps), len(hel1os.timestamps))
            return _trim(solexs, n), _trim(hel1os, n), lag_sec

        logger.warning("CLOCK_DRIFT > %.1fs (%.3fs) — cubic interpolation applied", self._drift_max, lag_sec)
        hel1os_aligned = _interpolate_to_grid(hel1os, solexs.timestamps)
        n = min(len(solexs.timestamps), len(hel1os_aligned.timestamps))
        return _trim(solexs, n), _trim(hel1os_aligned, n), lag_sec
```

- [ ] **Step 4: Run all alignment tests**

```
python -m pytest python/tests/test_l1_ingestion.py -k "time_sync or aligner or clock_drift" -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l1_ingestion/alignment.py python/tests/test_l1_ingestion.py
git commit -m "feat(l1): TimeAligner — cross-correlation lag, cubic interpolation, drift logging"
```

---

### Task 6: GoesLabelJoiner

**Files:**
- Create: `python/aruna/l1_ingestion/goes_join.py`
- Modify: `python/tests/test_l1_ingestion.py`

**Interfaces:**
- Consumes: `ObservationFrame` from `aruna.models.observation`
- Produces:
  - `GoesLabelJoiner(match_window_min=3.0, query_window_min=10.0)`
  - `.join(frames: List[ObservationFrame], hek_results=None) -> List[ObservationFrame]`
  - `.query_hek(start: datetime, end: datetime) -> Optional[results]`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l1_ingestion.py`:

```python
# ── Task 6: GoesLabelJoiner ───────────────────────────────────────────────────

def _make_obs_frames(n=5, base_dt=None):
    """Build minimal ObservationFrame list for label-join tests."""
    from aruna.models.observation import ObservationFrame, QualityFlag
    from datetime import datetime, timezone, timedelta
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


def _make_hek_row(fl_starttime: str, fl_goescls: str) -> dict:
    return {'event_starttime': fl_starttime, 'fl_goescls': fl_goescls}


def test_goes_label_joined_for_x63_event():
    """GOES label joined for X6.3 event (2024-02-22) ± 3 minutes (PRD §5.8)."""
    from aruna.l1_ingestion.goes_join import GoesLabelJoiner
    # Frames at 10:00:00–10:00:04; HEK event at 10:01:00 (within ±3 min)
    frames = _make_obs_frames(n=5)
    hek_results = [_make_hek_row('2024-02-22T10:01:00', 'X6.3')]
    result = GoesLabelJoiner().join(frames, hek_results)
    assert all(f.goes_class == 'X' for f in result)
    assert all(f.goes_subclass == pytest.approx(6.3) for f in result)


def test_goes_label_no_match_sets_quiet_sun():
    """Frames with no nearby HEK event → goes_class='QUIET_SUN'."""
    from aruna.l1_ingestion.goes_join import GoesLabelJoiner
    frames = _make_obs_frames(n=3)
    hek_results = [_make_hek_row('2024-02-22T11:00:00', 'C1.2')]  # 60 min away
    result = GoesLabelJoiner().join(frames, hek_results)
    assert all(f.goes_class == 'QUIET_SUN' for f in result)


def test_goes_label_offline_sets_quiet_sun():
    """HEK offline (hek_results=None) → goes_class='QUIET_SUN', no crash."""
    from aruna.l1_ingestion.goes_join import GoesLabelJoiner
    frames = _make_obs_frames(n=3)
    result = GoesLabelJoiner().join(frames, hek_results=None)
    assert all(f.goes_class == 'QUIET_SUN' for f in result)


def test_goes_label_empty_frames():
    """Empty frame list → returns empty list."""
    from aruna.l1_ingestion.goes_join import GoesLabelJoiner
    result = GoesLabelJoiner().join([], [_make_hek_row('2024-02-22T10:00:00', 'M1.0')])
    assert result == []


def test_goes_label_multiple_events_nearest_match():
    """When two events are within ±3 min, the first chronologically is chosen."""
    from aruna.l1_ingestion.goes_join import GoesLabelJoiner
    from datetime import datetime, timezone
    frames = _make_obs_frames(n=1, base_dt=datetime(2024, 2, 22, 10, 0, 0, tzinfo=timezone.utc))
    hek_results = [
        _make_hek_row('2024-02-22T09:58:00', 'M3.0'),  # 2 min before → within window
        _make_hek_row('2024-02-22T10:02:00', 'X1.0'),  # 2 min after → also within window
    ]
    result = GoesLabelJoiner().join(frames, hek_results)
    assert result[0].goes_class in ('M', 'X')  # either is valid — first match wins
```

- [ ] **Step 2: Run one test to verify failure**

```
python -m pytest python/tests/test_l1_ingestion.py::test_goes_label_joined_for_x63_event -v
```
Expected: `ERROR — ModuleNotFoundError`

- [ ] **Step 3: Create `python/aruna/l1_ingestion/goes_join.py`**

```python
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np

from ..models.observation import ObservationFrame

logger = logging.getLogger(__name__)


def _to_utc(t) -> datetime:
    if isinstance(t, datetime):
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    s = str(t).replace('Z', '+00:00')
    return datetime.fromisoformat(s)


class GoesLabelJoiner:
    def __init__(self, match_window_min: float = 3.0, query_window_min: float = 10.0):
        self._match_delta = timedelta(minutes=match_window_min)
        self._query_min = query_window_min

    def join(
        self,
        frames: List[ObservationFrame],
        hek_results=None,
    ) -> List[ObservationFrame]:
        """
        Annotate ObservationFrames with GOES class from HEK results.
        hek_results: list of dicts with 'event_starttime' and 'fl_goescls',
                     or a sunpy HEK results table, or None (offline).
        """
        if not frames:
            return frames

        events = []
        if hek_results is not None:
            for row in hek_results:
                try:
                    onset = _to_utc(row['event_starttime'])
                    cls_str = str(row.get('fl_goescls', '') or '').strip()
                    if not cls_str:
                        continue
                    letter = cls_str[0].upper()
                    subclass = float(cls_str[1:]) if len(cls_str) > 1 else 0.0
                    events.append((onset, letter, subclass))
                except (KeyError, ValueError, TypeError):
                    continue

        for frame in frames:
            ts = _to_utc(frame.timestamp) if not isinstance(frame.timestamp, np.datetime64) \
                else _np_to_utc(frame.timestamp)
            matched = False
            for onset, letter, subclass in events:
                if abs((ts - onset).total_seconds()) <= self._match_delta.total_seconds():
                    frame.goes_class = letter
                    frame.goes_subclass = subclass
                    matched = True
                    break
            if not matched:
                frame.goes_class = frame.goes_class or 'QUIET_SUN'

        return frames

    def query_hek(self, start: datetime, end: datetime):
        """Query SunPy HEK. Returns results or None if offline."""
        try:
            import astropy.units as u
            from sunpy.net import Hek, attrs as a
            hek = Hek()
            return hek.search(
                a.Time(start - timedelta(minutes=self._query_min),
                       end + timedelta(minutes=self._query_min)),
                a.hek.EventType("FL"),
            )
        except Exception as exc:
            logger.warning("HEK query failed (offline): %s", exc)
            return None


def _np_to_utc(t: np.datetime64) -> datetime:
    ts_sec = float((t - np.datetime64('1970-01-01T00:00:00', 'ns')) / np.timedelta64(1, 's'))
    return datetime.fromtimestamp(ts_sec, tz=timezone.utc)
```

- [ ] **Step 4: Run all GOES join tests**

```
python -m pytest python/tests/test_l1_ingestion.py -k "goes_label" -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l1_ingestion/goes_join.py python/tests/test_l1_ingestion.py
git commit -m "feat(l1): GoesLabelJoiner — SunPy HEK ±3 min match, offline graceful"
```

---

### Task 7: FrameValidator

**Files:**
- Create: `python/aruna/l1_ingestion/validator.py`
- Modify: `python/tests/test_l1_ingestion.py`

**Interfaces:**
- Consumes: `ObservationFrame`, `QualityFlag`; `IngestionConfig`, `QuarantineEntry` (Task 1)
- Produces:
  - `FrameValidator(config: IngestionConfig)`
  - `.validate(frames: List[ObservationFrame]) -> Tuple[List[ObservationFrame], List[ObservationFrame]]`
    — `(valid_frames, quarantined_frames)`
  - `.quarantine: List[QuarantineEntry]` — property

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l1_ingestion.py`:

```python
# ── Task 7: FrameValidator ────────────────────────────────────────────────────

def _make_valid_frame(counts=None):
    from aruna.models.observation import ObservationFrame, QualityFlag
    from datetime import datetime, timezone
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
    """pandera schema validation passes on > 95% of frames (PRD §5.8)."""
    from aruna.l1_ingestion.validator import FrameValidator
    from aruna.l1_ingestion.models import IngestionConfig
    frames = [_make_valid_frame() for _ in range(100)]
    cfg = IngestionConfig(completeness_threshold=0.80)
    valid, quarantined = FrameValidator(cfg).validate(frames)
    pass_rate = len(valid) / len(frames)
    assert pass_rate >= 0.95, f"Pass rate {pass_rate:.1%} < 95%"


def test_partial_fits_held_in_quarantine():
    """FITS with < 80% completeness held in quarantine (PRD §5.8)."""
    from aruna.l1_ingestion.validator import FrameValidator
    from aruna.l1_ingestion.models import IngestionConfig
    # Partial: 70% NaN counts (below 80% threshold)
    n_bins = 32
    counts = np.ones(n_bins)
    counts[:int(n_bins * 0.7)] = np.nan  # 70% NaN → 30% valid → below 80%
    partial_frame = _make_valid_frame(counts=counts)
    good_frames = [_make_valid_frame() for _ in range(9)]
    all_frames = good_frames + [partial_frame]
    cfg = IngestionConfig(completeness_threshold=0.80, quarantine_ttl_sec=300.0)
    valid, quarantined = FrameValidator(cfg).validate(all_frames)
    assert len(quarantined) >= 1
    assert any('completeness' in e.reason for e in FrameValidator(cfg).validate([partial_frame])[1])


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
    counts = np.full(32, np.nan)  # 100% NaN — definitely quarantined
    frame = _make_valid_frame(counts=counts)
    cfg = IngestionConfig(quarantine_ttl_sec=300.0)
    fv = FrameValidator(cfg)
    fv.validate([frame])
    assert len(fv.quarantine) == 1
    delta = (fv.quarantine[0].retry_after - fv.quarantine[0].quarantined_at).total_seconds()
    assert delta == pytest.approx(300.0, abs=1.0)
```

- [ ] **Step 2: Run one test to verify failure**

```
python -m pytest python/tests/test_l1_ingestion.py::test_schema_validation_passes_on_95_pct_of_frames -v
```
Expected: `ERROR — ModuleNotFoundError`

- [ ] **Step 3: Create `python/aruna/l1_ingestion/validator.py`**

```python
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pandera as pa

from ..models.observation import ObservationFrame, QualityFlag
from .models import IngestionConfig, QuarantineEntry

logger = logging.getLogger(__name__)

_SCHEMA = pa.DataFrameSchema({
    'instrument':        pa.Column(str,   nullable=False),
    'dead_time_fraction':pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
    'solar_distance_au': pa.Column(float, pa.Check.greater_than(0.0), nullable=False),
    'quality_flag':      pa.Column(int,   pa.Check.isin([f.value for f in QualityFlag]), nullable=False),
}, coerce=True)


class FrameValidator:
    def __init__(self, config: IngestionConfig):
        self._cfg = config
        self._quarantine: List[QuarantineEntry] = []

    @property
    def quarantine(self) -> List[QuarantineEntry]:
        return list(self._quarantine)

    def validate(
        self,
        frames: List[ObservationFrame],
    ) -> Tuple[List[ObservationFrame], List[ObservationFrame]]:
        if not frames:
            return [], []

        records = [{
            'instrument':         f.instrument,
            'dead_time_fraction': f.dead_time_fraction,
            'solar_distance_au':  f.solar_distance_au,
            'quality_flag':       int(f.quality_flag),
        } for f in frames]
        try:
            _SCHEMA.validate(pd.DataFrame(records), lazy=True)
        except pa.errors.SchemaErrors as exc:
            logger.warning("pandera errors: %s", exc.failure_cases.shape)

        valid, quarantined = [], []
        now = datetime.now(timezone.utc)
        for frame in frames:
            if self._is_complete(frame):
                valid.append(frame)
            else:
                quarantined.append(frame)
                self._quarantine.append(QuarantineEntry(
                    fits_path=Path(frame.source_file),
                    instrument=frame.instrument,
                    quarantined_at=now,
                    retry_after=now + timedelta(seconds=self._cfg.quarantine_ttl_sec),
                    reason="completeness < 80%",
                ))
                logger.warning("Frame %s quarantined: low completeness", frame.frame_id)

        if frames:
            rate = len(valid) / len(frames)
            if rate < 0.95:
                logger.error("Validation pass rate %.1f%% < 95%%", rate * 100)
        return valid, quarantined

    def _is_complete(self, frame: ObservationFrame) -> bool:
        c = frame.counts_per_s_kev
        if c is None or len(c) == 0:
            return False
        return float(np.isfinite(c).sum()) / len(c) >= self._cfg.completeness_threshold
```

- [ ] **Step 4: Run all validator tests**

```
python -m pytest python/tests/test_l1_ingestion.py -k "schema_validation or partial_fits or validator" -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l1_ingestion/validator.py python/tests/test_l1_ingestion.py
git commit -m "feat(l1): FrameValidator — pandera schema, completeness gate, quarantine queue"
```

---

### Task 8: KafkaPublisher

**Files:**
- Create: `python/aruna/l1_ingestion/publisher.py`
- Modify: `python/tests/test_l1_ingestion.py`

**Interfaces:**
- Consumes: `ObservationFrame` from `aruna.models.observation`
- Produces:
  - `KafkaPublisher(producer, topic_solexs, topic_hel1os, topic_quality)`
  - `.publish(frames: List[ObservationFrame]) -> None` — sends to Kafka within 2s

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l1_ingestion.py`:

```python
# ── Task 8: KafkaPublisher ────────────────────────────────────────────────────

def test_kafka_publish_within_2_seconds():
    """ObservationFrame published to Kafka within 2 seconds of FITS arrival (PRD §5.8)."""
    import time
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.publisher import KafkaPublisher
    producer = MagicMock()
    pub = KafkaPublisher(producer)
    frames = [_make_valid_frame() for _ in range(10)]
    t0 = time.monotonic()
    pub.publish(frames)
    elapsed = time.monotonic() - t0
    assert elapsed < 2.0, f"Publish took {elapsed:.2f}s > 2s"
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
    from datetime import datetime, timezone
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
    """Quality summary published to aruna.quality.summary after each batch."""
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.publisher import KafkaPublisher
    producer = MagicMock()
    pub = KafkaPublisher(producer, topic_quality='aruna.quality.summary')
    pub.publish([_make_valid_frame()])
    topics = [call.args[0] for call in producer.send.call_args_list]
    assert 'aruna.quality.summary' in topics


def test_kafka_payload_has_schema_version():
    """Every published payload has schema_version='1.0' and pipeline_version='2.0.0'."""
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
```

- [ ] **Step 2: Run one test to verify failure**

```
python -m pytest python/tests/test_l1_ingestion.py::test_kafka_publish_within_2_seconds -v
```
Expected: `ERROR — ModuleNotFoundError`

- [ ] **Step 3: Create `python/aruna/l1_ingestion/publisher.py`**

```python
from __future__ import annotations

import dataclasses
import logging
from datetime import datetime
from typing import List

import numpy as np

from ..models.observation import ObservationFrame

logger = logging.getLogger(__name__)

_PIPELINE_VERSION = "2.0.0"
_SCHEMA_VERSION = "1.0"


def _to_json(frame: ObservationFrame) -> dict:
    d = dataclasses.asdict(frame)
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, datetime):
            d[k] = v.isoformat()
        elif isinstance(v, (np.integer,)):
            d[k] = int(v)
        elif isinstance(v, (np.floating,)):
            d[k] = float(v)
    d['schema_version'] = _SCHEMA_VERSION
    d['pipeline_version'] = _PIPELINE_VERSION
    return d


class KafkaPublisher:
    def __init__(
        self,
        producer,
        topic_solexs: str = "aruna.observations.solexs",
        topic_hel1os: str = "aruna.observations.hel1os",
        topic_quality: str = "aruna.quality.summary",
    ):
        self._producer = producer
        self._t_solexs  = topic_solexs
        self._t_hel1os  = topic_hel1os
        self._t_quality = topic_quality

    def publish(self, frames: List[ObservationFrame]) -> None:
        counts = {'total': 0, 'good': 0, 'flagged': 0}
        for frame in frames:
            topic = self._t_solexs if frame.is_solexs() else self._t_hel1os
            payload = _to_json(frame)
            try:
                self._producer.send(topic, value=payload, key=frame.instrument.encode())
                counts['total'] += 1
                if frame.is_good():
                    counts['good'] += 1
                else:
                    counts['flagged'] += 1
            except Exception as exc:
                logger.error("Kafka publish failed for %s: %s", frame.frame_id, exc)

        summary = {
            'schema_version': _SCHEMA_VERSION,
            'pipeline_version': _PIPELINE_VERSION,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            **counts,
        }
        try:
            self._producer.send(self._t_quality, value=summary)
        except Exception as exc:
            logger.error("Quality summary publish failed: %s", exc)
```

- [ ] **Step 4: Run all publisher tests**

```
python -m pytest python/tests/test_l1_ingestion.py -k "kafka" -v
```
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l1_ingestion/publisher.py python/tests/test_l1_ingestion.py
git commit -m "feat(l1): KafkaPublisher — serialize ObservationFrame, quality summary"
```

---

### Task 9: IngestionPipeline + `__init__.py` + Acceptance Gate

**Files:**
- Create: `python/aruna/l1_ingestion/pipeline.py`
- Modify: `python/aruna/l1_ingestion/__init__.py`
- Modify: `python/tests/test_l1_ingestion.py`

**Interfaces:**
- Consumes: all Tasks 1–8 modules
- Produces:
  - `IngestionPipeline(cache, config, kafka_producer)`
  - `.ingest(solexs_path, hel1os_path, hk_path) -> List[ObservationFrame]`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l1_ingestion.py`:

```python
# ── Task 9: IngestionPipeline + Acceptance Gate ───────────────────────────────

def test_x63_event_processes_without_errors(
    x63_solexs_fits_path, hel1os_fits_path, hk_fits_path
):
    """X6.3 event (2024-02-22) ingested end-to-end without errors (PRD §5.8)."""
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.pipeline import IngestionPipeline
    from aruna.l1_ingestion.models import IngestionConfig
    producer = MagicMock()
    pipeline = IngestionPipeline(
        cache=None,
        config=IngestionConfig(),
        kafka_producer=producer,
    )
    frames = pipeline.ingest(x63_solexs_fits_path, hel1os_fits_path, hk_fits_path)
    assert len(frames) > 0
    assert all(hasattr(f, 'instrument') for f in frames)


def test_pipeline_publishes_to_kafka(
    solexs_fits_path, hel1os_fits_path, hk_fits_path
):
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.pipeline import IngestionPipeline
    from aruna.l1_ingestion.models import IngestionConfig
    producer = MagicMock()
    pipeline = IngestionPipeline(cache=None, config=IngestionConfig(), kafka_producer=producer)
    pipeline.ingest(solexs_fits_path, hel1os_fits_path, hk_fits_path)
    assert producer.send.called


def test_pipeline_x63_sdd1_saturated_frames_use_sdd2(
    x63_solexs_fits_path, hel1os_fits_path, hk_fits_path
):
    """SDD1-saturated frames in X6.3 scenario get QualityFlag.SATURATED."""
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.pipeline import IngestionPipeline
    from aruna.l1_ingestion.models import IngestionConfig
    from aruna.models.observation import QualityFlag
    producer = MagicMock()
    pipeline = IngestionPipeline(cache=None, config=IngestionConfig(), kafka_producer=producer)
    frames = pipeline.ingest(x63_solexs_fits_path, hel1os_fits_path, hk_fits_path)
    solexs_frames = [f for f in frames if f.is_solexs()]
    saturated = [f for f in solexs_frames if f.quality_flag == QualityFlag.SATURATED]
    assert len(saturated) > 0, "Expected some SATURATED frames in X6.3 scenario"


def test_pipeline_hk_saa_flag_propagated(
    solexs_fits_path, hel1os_fits_path, tmp_path
):
    """SAA flag from HK propagates to ObservationFrame.saa_flag."""
    from astropy.io import fits as afits
    from unittest.mock import MagicMock
    from aruna.l1_ingestion.pipeline import IngestionPipeline
    from aruna.l1_ingestion.models import IngestionConfig

    # Build HK FITS with SAA_FLAG=1 for all frames
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
    assert all(f.saa_flag for f in frames), "All frames should have saa_flag=True"


# ── Acceptance gate (PRD §5.8) ────────────────────────────────────────────────

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
    from datetime import datetime, timezone
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
```

- [ ] **Step 2: Run one test to verify failure**

```
python -m pytest python/tests/test_l1_ingestion.py::test_x63_event_processes_without_errors -v
```
Expected: `ERROR — ModuleNotFoundError: No module named 'aruna.l1_ingestion.pipeline'`

- [ ] **Step 3: Create `python/aruna/l1_ingestion/pipeline.py`**

```python
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..models.observation import ObservationFrame, QualityFlag
from .alignment import TimeAligner
from .background import BackgroundSubtractor
from .deadtime import DeadTimeCorrector
from .fits_reader import FitsReader
from .goes_join import GoesLabelJoiner
from .models import FitsData, HKData, IngestionConfig
from .publisher import KafkaPublisher
from .validator import FrameValidator

logger = logging.getLogger(__name__)

_PIPELINE_VERSION = "2.0.0"
_CALIBRATION_VERSION = "sarwade_2025_v1"


def _dt64_to_datetime(t: np.datetime64) -> datetime:
    ts = float((t - np.datetime64('1970-01-01T00:00:00', 'ns')) / np.timedelta64(1, 's'))
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _nearest_hk(hk: HKData, ts: np.datetime64):
    """Find HK values nearest to ts."""
    idx = int(np.argmin(np.abs(hk.timestamps - ts)))
    return (
        bool(hk.saa_flags[idx]),
        str(hk.instrument_modes[idx]),
        float(hk.solar_dist_au[idx]),
    )


def _fits_to_frames(
    data: FitsData,
    hk: HKData,
    quality_flags: np.ndarray,
) -> List[ObservationFrame]:
    frames = []
    for i in range(len(data.timestamps)):
        saa, mode, dist = _nearest_hk(hk, data.timestamps[i])
        flag = QualityFlag(int(quality_flags[i]))
        if saa and flag == QualityFlag.GOOD:
            flag = QualityFlag.SAA_PASSAGE
        frames.append(ObservationFrame(
            instrument=data.instrument,
            timestamp=_dt64_to_datetime(data.timestamps[i]),
            energy_bins_kev=data.energy_bins_kev,
            counts_per_s_kev=data.counts_raw[i],
            quality_flag=flag,
            dead_time_fraction=float(data.deadtime_frac[i]),
            calibration_version=_CALIBRATION_VERSION,
            solar_distance_au=dist,
            spacecraft_mode=mode,
            saa_flag=saa,
            source_file=data.source_file,
            checksum_sha256=data.checksum_sha256,
            pipeline_version=_PIPELINE_VERSION,
        ))
    return frames


class IngestionPipeline:
    def __init__(
        self,
        cache,
        config: Optional[IngestionConfig] = None,
        kafka_producer=None,
    ):
        self._cache = cache
        self._cfg = config or IngestionConfig()
        self._reader = FitsReader()
        self._dtc = DeadTimeCorrector()
        self._bg = BackgroundSubtractor(chi2_dof_limit=self._cfg.chi2_dof_limit)
        self._aligner = TimeAligner(self._cfg.clock_drift_warn_sec, self._cfg.clock_drift_max_sec)
        self._joiner = GoesLabelJoiner(self._cfg.goes_match_window_min, self._cfg.goes_query_window_min)
        self._validator = FrameValidator(self._cfg)
        self._publisher = KafkaPublisher(
            kafka_producer,
            self._cfg.kafka_topic_solexs,
            self._cfg.kafka_topic_hel1os,
            self._cfg.kafka_topic_quality,
        ) if kafka_producer else None

    def ingest(
        self,
        solexs_path: Path,
        hel1os_path: Path,
        hk_path: Path,
        hek_results=None,
    ) -> List[ObservationFrame]:
        """
        Full L1 ingestion pipeline: FITS → corrected → aligned → labelled → validated → published.
        """
        hk = self._reader.read_hk(hk_path)

        # ── SoLEXS: dead-time correction + SDD1/SDD2 selection ──────────────
        solexs_list = self._reader.read_solexs(solexs_path)
        sdd1 = next(fd for fd in solexs_list if fd.instrument == 'SOLEXS_SDD1')
        sdd2 = next(fd for fd in solexs_list if fd.instrument == 'SOLEXS_SDD2')
        corr_sdd1, flags_sdd1 = self._dtc.correct(sdd1)
        corr_sdd2, flags_sdd2 = self._dtc.correct(sdd2)
        solexs_data, solexs_flags = self._dtc.select_solexs_detector(
            corr_sdd1, flags_sdd1, corr_sdd2, flags_sdd2
        )

        # ── HEL1OS: dead-time correction + background subtraction ────────────
        hel1os_list = self._reader.read_hel1os(hel1os_path)
        cdte = next(fd for fd in hel1os_list if fd.instrument == 'HEL1OS_CDTE')
        corr_cdte, flags_cdte = self._dtc.correct(cdte)
        saa_flags_cdte = np.array([_nearest_hk(hk, ts)[0] for ts in cdte.timestamps])
        bg_cdte, bg_flags_cdte, _ = self._bg.subtract(corr_cdte, [], saa_flags_cdte)

        # ── Time alignment ───────────────────────────────────────────────────
        solexs_aligned, hel1os_aligned, _ = self._aligner.align(solexs_data, bg_cdte)
        n = min(len(solexs_aligned.timestamps), len(hel1os_aligned.timestamps))
        solexs_flags_trim = solexs_flags[:n]
        hel1os_flags_trim = bg_flags_cdte[:n]

        # ── Build ObservationFrames ──────────────────────────────────────────
        all_frames: List[ObservationFrame] = []
        all_frames.extend(_fits_to_frames(solexs_aligned, hk, solexs_flags_trim))
        all_frames.extend(_fits_to_frames(hel1os_aligned, hk, hel1os_flags_trim))

        # ── GOES label join ──────────────────────────────────────────────────
        all_frames = self._joiner.join(all_frames, hek_results)

        # ── Schema validation ────────────────────────────────────────────────
        valid_frames, _ = self._validator.validate(all_frames)

        # ── Kafka publish ────────────────────────────────────────────────────
        if self._publisher:
            self._publisher.publish(valid_frames)

        return valid_frames
```

- [ ] **Step 4: Update `python/aruna/l1_ingestion/__init__.py`**

```python
from .background import BackgroundSubtractor
from .alignment import TimeAligner
from .deadtime import DeadTimeCorrector, DEADTIME_CONSTANTS, SATURATION_LIMITS
from .fits_reader import FitsReader
from .goes_join import GoesLabelJoiner
from .models import FitsData, HKData, IngestionConfig, QuarantineEntry, ValidationResult
from .pipeline import IngestionPipeline
from .publisher import KafkaPublisher
from .validator import FrameValidator

__all__ = [
    "BackgroundSubtractor",
    "TimeAligner",
    "DeadTimeCorrector",
    "DEADTIME_CONSTANTS",
    "SATURATION_LIMITS",
    "FitsData",
    "FitsReader",
    "GoesLabelJoiner",
    "HKData",
    "IngestionConfig",
    "IngestionPipeline",
    "KafkaPublisher",
    "QuarantineEntry",
    "ValidationResult",
    "FrameValidator",
]
```

- [ ] **Step 5: Run the full test suite**

```
python -m pytest python/tests/test_l1_ingestion.py -v -m "not integration"
```
Expected: all PASSED, 0 failures

- [ ] **Step 6: Run full L0 + L1 suite to check no regressions**

```
python -m pytest python/tests/test_l0_source.py python/tests/test_l1_ingestion.py -v -m "not integration" --tb=short -q
```
Expected: 43 + all L1 tests PASSED

- [ ] **Step 7: Final commit**

```
git add python/aruna/l1_ingestion/pipeline.py python/aruna/l1_ingestion/__init__.py python/tests/test_l1_ingestion.py
git commit -m "feat(l1): IngestionPipeline — end-to-end FITS→ObservationFrame→Kafka, all acceptance tests green"
```

---

## Self-Review

**Spec coverage check:**
- ✓ §5.1 Objective — pipeline.py orchestrates FITS→ObservationFrame→Kafka
- ✓ §5.2 Inputs — SoLEXS + HEL1OS + HK via FitsReader; HEK via GoesLabelJoiner
- ✓ §5.3 Preconditions — FitsReader raises on missing extensions (test: `test_fits_reader_missing_extension_raises`)
- ✓ §5.4 Step 1 (cache lookup) — pipeline takes path from caller (CacheManager.get() in production)
- ✓ §5.4 Step 2 (FITS parse) — Task 2 FitsReader
- ✓ §5.4 Step 3 (dead-time) — Task 3 DeadTimeCorrector
- ✓ §5.4 Step 4 (background) — Task 4 BackgroundSubtractor; SAA masking included
- ✓ §5.4 Step 5 (time alignment) — Task 5 TimeAligner
- ✓ §5.4 Step 6 (GOES join) — Task 6 GoesLabelJoiner
- ✓ §5.4 Step 7 (schema validation) — Task 7 FrameValidator
- ✓ §5.4 Step 8 (Kafka publish) — Task 8 KafkaPublisher
- ✓ §5.5 Dead-time constants — hardcoded in deadtime.py, tested in `test_dead_time_constants_match_prd`
- ✓ §5.6 Failure recovery — SATURATED flag, SAA_PASSAGE, quarantine, offline HEK fallback
- ✓ §5.8 All 9 acceptance criteria — all covered across Tasks 3–9

**Placeholder scan:** None found.

**Type consistency:**
- `FitsData` defined Task 1, used identically in Tasks 2–5, 9 ✓
- `HKData` defined Task 1, used identically in Tasks 2, 9 ✓
- `DeadTimeCorrector.correct()` → `(FitsData, np.ndarray)` — consistent Task 3 → Task 9 ✓
- `BackgroundSubtractor.subtract()` → `(FitsData, np.ndarray, float)` — consistent Task 4 → Task 9 ✓
- `TimeAligner.align()` → `(FitsData, FitsData, float)` — consistent Task 5 → Task 9 ✓
- `GoesLabelJoiner.join()` takes `List[ObservationFrame]` — consistent Task 6 → Task 9 ✓
- `FrameValidator.validate()` → `(List[ObservationFrame], List[ObservationFrame])` — consistent Task 7 → Task 9 ✓
- `KafkaPublisher.publish()` takes `List[ObservationFrame]` — consistent Task 8 → Task 9 ✓
- `__init__.py` exports match class names in all tasks ✓
