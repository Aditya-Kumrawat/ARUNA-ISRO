# L1 Data Ingestion — Design Spec
**Date:** 2026-06-18
**PRD chapter:** §5
**Phase:** 2
**Status:** Approved

---

## 1. Objective

L1 is the transformation boundary between raw PRADAN telemetry and validated science-ready spectra. It reads FITS files from L0's CacheManager, applies all corrections and enrichments, and publishes clean `ObservationFrame` objects to Kafka so L2 can begin physics computation.

L1 never talks to PRADAN directly — all file I/O goes through `CacheManager.get()`.

---

## 2. Data Sources

### 2.1 Primary (required — all from L0 CacheManager)

| File pattern | Instrument | Content |
|---|---|---|
| `SoLEXS_L1_YYYYMMDD.fits` | SOLEXS_SDD1, SOLEXS_SDD2 | 1-second spectra, 2–22 keV |
| `HEL1OS_L1_YYYYMMDD.fits` | HEL1OS_CDTE, HEL1OS_CZT | 1-second spectra, 8–150 keV |
| `Aditya_HK_YYYYMMDD.fits` | Housekeeping | SAA_FLAG, INSTRUMENT_MODE, SOLAR_DIST_AU |

### 2.2 Secondary (optional enrichment)

| Source | Access | Used for |
|---|---|---|
| SunPy HEK | Direct HTTP query (mockable) | GOES flare label join ±3 min (Step 6) |
| NOAA SWPC JSON | Via L0 SupplementaryFetcher | Offline fallback — L0 handles, L1 reads flag |
| GOES XRS L1b NetCDF | Optional cross-validation | Not required for operational pipeline |

**VELC and SUIT** FITS files are downloaded by L0 but not processed in L1 — they are consumed by XL modules only.

---

## 3. Architecture

```
L0 CacheManager
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    L1 — IngestionPipeline                       │
│                                                                 │
│  FitsReader ──→ DeadTimeCorrector ──→ BackgroundSubtractor      │
│                      (SoLEXS)            (HEL1OS)               │
│                         │                    │                  │
│                         └────────┬───────────┘                  │
│                                  ▼                              │
│                         TimeAligner (cross-correlation)         │
│                                  │                              │
│                                  ▼                              │
│                         GoesLabelJoiner (SunPy HEK)             │
│                                  │                              │
│                                  ▼                              │
│                         FrameValidator (pandera)                │
│                                  │                              │
│                         QuarantineQueue (< 80% complete)        │
│                                  │                              │
│                                  ▼                              │
│                         KafkaPublisher                          │
└─────────────────────────────────────────────────────────────────┘
      │                        │                       │
      ▼                        ▼                       ▼
aruna.observations.solexs  aruna.observations.hel1os  aruna.quality.summary
```

---

## 4. Module Map

| File | Responsibility |
|---|---|
| `models.py` | `FitsData`, `IngestionConfig`, `QuarantineEntry`, `ValidationResult` |
| `fits_reader.py` | `FitsReader` — astropy parse, column extraction, HK flag read |
| `deadtime.py` | `DeadTimeCorrector` — non-paralysable model, SDD1→SDD2 auto-switch |
| `background.py` | `BackgroundSubtractor` — scipy polyfit, χ²/dof validation (sherpa fallback) |
| `alignment.py` | `TimeAligner` — cross-correlation lag, cubic interpolation, drift logging |
| `goes_join.py` | `GoesLabelJoiner` — SunPy HEK query, ±3 min match, goes_class annotation |
| `validator.py` | `FrameValidator` — pandera schema, physical range checks, quarantine gate |
| `publisher.py` | `KafkaPublisher` — serialize ObservationFrame, send to Kafka topics |
| `pipeline.py` | `IngestionPipeline` — orchestrates all steps, failure recovery matrix |
| `__init__.py` | Public API re-exports |

---

## 5. Key Algorithms

### 5.1 Dead-Time Correction (PRD §5.4 Step 3, §5.5)

Non-paralysable detector model per Sarwade et al. 2025:

```
R_true = R_obs / (1 - R_obs × τ)
```

Constants (hardcoded from paper — not in config, immutable calibration values):
```python
DEADTIME_CONSTANTS = {
    'SOLEXS_SDD1': 4.5e-6,
    'SOLEXS_SDD2': 4.5e-6,
    'HEL1OS_CDTE': 2.1e-6,
    'HEL1OS_CZT':  1.8e-6,
}
SATURATION_LIMITS = {
    'SOLEXS_SDD1': 5e4,   # counts/s → auto-switch to SDD2
    'SOLEXS_SDD2': 5e6,   # counts/s → X10+ approach limit
}
```

Applied per 1-second bin, per energy channel independently. Bins where corrected rate exceeds `SOLEXS_SDD1` limit get `QualityFlag.SATURATED` and the frame switches to SDD2.

### 5.2 HEL1OS Background Subtraction (PRD §5.4 Step 4)

Sherpa is not buildable on Windows. Implementation uses `scipy.optimize` polynomial fit (equivalent algorithm), consistent with the PRD risk mitigation for unavailable packages.

Steps:
1. Extract 5-minute pre-event quiet window (300 frames prior to current window)
2. Per energy bin: fit 2nd-order polynomial to quiet background counts
3. Subtract polynomial from observed counts
4. Compute χ²/dof on residuals vs Poisson noise: `χ² = Σ (obs - model)² / model`
5. If χ²/dof > 2.0: flag `QualityFlag.NOISY`, do not discard
6. SAA-coincident bins: set `QualityFlag.SAA_PASSAGE`, zero the counts, keep the frame

### 5.3 Time Alignment (PRD §5.4 Step 5)

1. Convert both FITS time columns to `numpy.datetime64` UTC
2. Extract 10–15 keV band from SoLEXS and HEL1OS (overlapping window)
3. `scipy.signal.correlate` to find peak cross-correlation lag
4. If lag > 0.5s: log `CLOCK_DRIFT` warning to structlog
5. If lag > 1.0s: apply `scipy.interpolate.interp1d` cubic interpolation to resample to common 1s UTC grid
6. If lag ≤ 1.0s: resample with linear interpolation to common grid, no flag

### 5.4 GOES Label Join (PRD §5.4 Step 6)

```python
from sunpy.net import Hek, attrs as a

results = Hek.search(
    a.Time(window_start - 10*u.min, window_end + 10*u.min),
    a.hek.EventType("FL")
)
```

Match criteria:
- Flare onset within ±3 minutes of ObservationFrame timestamp
- GOES class within 1 sub-class of frame flux level
- Annotate matched frames: `goes_class`, `goes_subclass`
- Unmatched: `goes_class = "QUIET_SUN"` if below B-class threshold, else `"UNKNOWN"`

HEK is always called with a timeout and try/except — offline mode skips annotation silently.

### 5.5 Schema Validation (PRD §5.4 Step 7)

pandera `DataFrameSchema` checks:
- All required `ObservationFrame` fields non-null
- `energy_bins_kev`: monotonically increasing, 2–150 keV range
- `counts_per_s_kev`: non-negative floats
- `dead_time_fraction`: [0.0, 1.0]
- `quality_flag`: valid `QualityFlag` enum value
- Completeness: > 80% non-null per 60-second window

Frames failing validation go to `QuarantineQueue` (in-memory, 5-minute TTL) then retry once. Still-failing frames get `QualityFlag.MISSING` and are published with that flag — never silently dropped.

---

## 6. FITS Column Convention

Since real FITS files from PRADAN aren't available for testing, synthetic FITS are generated in tests using `astropy.io.fits` with these column names (matching ISRO Aditya-L1 L1 data format documentation):

**SoLEXS / HEL1OS:**
- `TIME` — seconds since FITS epoch (converted to UTC datetime64)
- `COUNTS` — shape (N_times, N_energy_bins), raw detector counts
- `ENERGY_BINS` — shape (N_bins+1,), energy bin edges in keV
- `LIVETIME` — shape (N_times,), live time fraction per second
- `DEADTIME_FRAC` — shape (N_times,), dead-time fraction per second

**Housekeeping (HK):**
- `TIME` — same epoch
- `SAA_FLAG` — 0 or 1 per second
- `INSTRUMENT_MODE` — string: 'NOMINAL', 'CALIBRATION', 'SAFE'
- `SOLAR_DIST_AU` — float per second

---

## 7. Failure Recovery Matrix (PRD §5.6)

| Failure | Detection | Action | Quality Flag |
|---|---|---|---|
| FITS missing | `CacheManager.get()` returns None | Retry ×3, then GAP | `MISSING` |
| SHA-256 mismatch | CacheManager integrity check | Re-download ×1 | `GAP` |
| SDD1 saturated | rate > 5e4 counts/s | Switch to SDD2 | `SATURATED` |
| Clock drift > 1s | Cross-corr lag | Interpolate + log CLOCK_DRIFT | `GAP` |
| SAA passage | HK SAA_FLAG=1 | Mask bins, keep frame | `SAA_PASSAGE` |
| PRADAN down > 5m | L0 health monitor | Use cache (L0 handles) | `DATA_DEGRADED` (set by L0) |
| Both instruments offline | L0 health monitor | Fallback NOAA SWPC | L0 publishes `DATA_FALLBACK` |
| Partial FITS < 80% | pandera completeness | Quarantine 5 min, retry | Hold, then `MISSING` |

---

## 8. Testing Strategy

Same TDD pattern as L0: write failing test → implement → verify passing.

**Synthetic FITS generation:** A `conftest.py` fixture creates in-memory astropy FITS structures with realistic column shapes. No real PRADAN files needed.

**HEK mocking:** `unittest.mock.patch` on `sunpy.net.Hek.search` — returns synthetic flare event list.

**Kafka mocking:** `MagicMock()` producer — same pattern as L0 health monitor tests.

**Structlog capture:** `structlog.testing.capture_logs()` context manager to assert drift/saturation log entries.

**Test coverage targets:**
- `fits_reader.py`: roundtrip parse, missing columns, wrong shape
- `deadtime.py`: correction formula accuracy, SDD1→SDD2 switch, both-saturated edge case
- `background.py`: subtraction reduces background, χ²/dof gate, SAA masking
- `alignment.py`: zero-lag case, lag > 0.5s logging, lag > 1.0s interpolation
- `goes_join.py`: label match, no-match → QUIET_SUN, HEK offline graceful skip
- `validator.py`: valid frame passes, partial FITS quarantined, reject < 95%
- `publisher.py`: Kafka send called per instrument, quality summary published
- `pipeline.py`: X6.3 end-to-end with synthetic FITS, all acceptance criteria

---

## 9. Acceptance Criteria (PRD §5.8)

```
✓ Time sync error between SoLEXS and HEL1OS < 1 second
✓ Dead-time correction applied to all SoLEXS bins (SDD1 + SDD2)
✓ HEL1OS background subtracted, χ²/dof < 2.0
✓ SAA-coincident bins FLAGGED (QualityFlag.SAA_PASSAGE), not dropped
✓ Schema validation passes on > 95% of frames
✓ GOES label joined for X6.3 event (2024-02-22) ± 3 min
✓ ObservationFrame published to Kafka within 2 seconds of FITS arrival
✓ X6.3 event (2024-02-22) processes end-to-end without errors
✓ Clock drift > 0.5s logged (structlog CLOCK_DRIFT entry)
✓ SDD1 saturation > 5e4 counts/s → auto-switch SDD2 + SATURATED flag
✓ Partial FITS (< 80% complete) held in quarantine 5 min, then retried
```

---

## 10. Global Constraints (inherited from L0)

- Python 3.11 — no `match` statements for compat, use `if/elif`
- All timestamps ISO 8601 UTC; internal repr `numpy.datetime64` nanoseconds
- `schema_version = "1.0"`, `pipeline_version = "2.0.0"` on every published message
- All file writes atomic: write to `.tmp`, then `Path.replace()`
- Tests must not make real network calls unless marked `@pytest.mark.integration`
- Config values from `config/config.yaml`; dead-time constants are calibration constants (hardcoded from paper, never in config)
- Sherpa unavailable on Windows → scipy polynomial background subtraction (functionally equivalent, same χ²/dof criterion)
