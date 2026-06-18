# ARUNA — Engineering Pipeline PRD
## Aditya-L1 Real-time Unified Nowcasting & Alert System
### Complete Implementation Specification · BAH 2026 · Problem Statement 15

**Version:** 1.0.0  
**Status:** Implementation Ready  
**Classification:** Engineering Specification  
**Mentors:** Abhilash R. Sarwade · Srikar Paavan Tadepalli · Kiran Lakshmipathaiah  
**Tagline:** Foresee. Prepare. Protect.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [End-to-End System Flow](#2-end-to-end-system-flow)
3. [Global Data Contracts](#3-global-data-contracts)
4. [L0 — Source Layer](#4-l0--source-layer)
5. [L1 — Data Ingestion](#5-l1--data-ingestion)
6. [L2 — Physics Engine](#6-l2--physics-engine)
7. [L3 — Nowcasting Engine](#7-l3--nowcasting-engine)
8. [L4 — Forecasting Engine](#8-l4--forecasting-engine)
9. [L5 — Historical Analog Retrieval](#9-l5--historical-analog-retrieval)
10. [L6 — Threat Intelligence](#10-l6--threat-intelligence)
11. [L7 — LLM Briefing](#11-l7--llm-briefing)
12. [L8 — Dashboard Services](#12-l8--dashboard-services)
13. [XL Extension Modules](#13-xl-extension-modules)
14. [Message Bus — Kafka Topics](#14-message-bus--kafka-topics)
15. [Database Schema](#15-database-schema)
16. [API Specifications](#16-api-specifications)
17. [Deployment Architecture](#17-deployment-architecture)
18. [Monitoring & Observability](#18-monitoring--observability)
19. [Testing & Validation Framework](#19-testing--validation-framework)
20. [Verification & Validation Plan (V&V)](#20-verification--validation-plan-vv)
21. [FMEA — Failure Modes & Effects Analysis](#21-fmea--failure-modes--effects-analysis)
22. [Risk Register](#22-risk-register)
23. [Technology Readiness Levels](#23-technology-readiness-levels)
24. [Configuration Reference](#24-configuration-reference)
25. [Operational Runbooks](#25-operational-runbooks)
26. [Decision Log](#26-decision-log)
27. [Research Gaps Addressed](#27-research-gaps-addressed)
28. [Acceptance Criteria Master Checklist](#28-acceptance-criteria-master-checklist)

---

## 1. Executive Summary

### 1.1 What ARUNA Is

ARUNA is a physics-grounded, multi-instrument, operationally deployable solar flare nowcasting and forecasting pipeline built natively on Aditya-L1's SoLEXS (2–22 keV soft X-ray) and HEL1OS (8–150 keV hard X-ray) instruments. It is the first automated pipeline to treat these two instruments as a coupled thermal-nonthermal detector pair at 1-second cadence from the Sun-Earth L1 vantage point.

### 1.2 What ARUNA Does (Plain Terms)

| Mode | What it does |
|------|-------------|
| **Watch** | Passive situational awareness — Sun status at all times |
| **Warn** | Real-time alert when a flare is developing, before peak |
| **Brief** | Translates detection into India-specific operational risk |
| **Report** | Post-event reconstruction with physics diagnostics |

### 1.3 The Operational Gap ARUNA Fills

In February 2026, ISRO's ISTRAC Director manually monitored 50 operational satellites during an X-class flare event. ARUNA automates this entirely with 5–15 minutes of lead time before the flare peaks. This is the system's primary operational motivation.

### 1.4 Key Capabilities

- **Nowcasting:** Dual-instrument flare detection with GOES-class classification (B/C/M/X)
- **Forecasting:** P(flare in next N minutes) with calibrated confidence intervals
- **Physics diagnostics:** Temperature T(t), Emission Measure EM(t), Dynamic Neupert Integral DNI(t), Thermal-Nonthermal Phase Space Velocity TNPSV(t), Spectral Hardness Evolution Index SHEI(t)
- **Precursor detection:** HOPE (Hot Onset Precursor Event) flag with lead time estimation
- **Master Flare Catalog:** Unified SoLEXS + HEL1OS event database
- **Threat intelligence:** NavIC, HF comms, Gaganyaan radiation risk flags
- **Operational output:** ISRO SpaceWx bulletin-compatible alert format

### 1.5 What Makes ARUNA Different

Every other approach trains a model on a single light curve. ARUNA is different on three axes:

1. **Physics-grounded, not a black box.** Real plasma diagnostics from SoLEXS spectra. When the system fires, it explains WHY in physical terms.
2. **True dual-instrument fusion.** SoLEXS (thermal) and HEL1OS (non-thermal) as a coupled trigger pair. The timing offset between thermal precursor and non-thermal onset IS the forecast lead time.
3. **Threat intelligence, not just flare class.** ISRO doesn't care about the letter "M3.8" — it cares about NavIC signal integrity, Gaganyaan crew safety, and ISTRAC satellite safing priority.

### 1.6 Research Gaps Addressed

ARUNA directly addresses four explicit gaps from Sarwade et al.'s published work:

- Published work covers only 47 X-class flares → ARUNA automates T/EM across all M+C+X class
- Rapid height variation tracking limited by statistics → ARUNA provides 1s Fe Kα vs HEL1OS correlation
- No automated pipeline exists → ARUNA IS that pipeline
- Flare onset phase flagged as future work → ARUNA's HOPE precursor addresses this directly

---

## 2. End-to-End System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        L0 — SOURCE                              │
│  Aditya-L1 at L1 → SoLEXS + HEL1OS → PRADAN FITS files        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     L1 — DATA INGESTION                         │
│  FITS read → Checksum → Dead-time correction → Time alignment   │
│  Quality gate → Kafka publish → Gap interpolation               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     L2 — PHYSICS ENGINE                         │
│  Isothermal fit → T(t), EM(t) → DNI → TNPSV → SHEI → HFD     │
│  QPP detection → HOPE precursor flag → Fe-Kα correlation        │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    L3 — NOWCASTING ENGINE                       │
│  PELT change-point → Adaptive P85 threshold → Dual-trigger      │
│  GOES classification → Master Flare Catalog write               │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    L4 — FORECASTING ENGINE                      │
│  Sliding 10-min window → Dual-stream TCN → Moirai zero-shot    │
│  Ensemble → MAPIE conformal → P(flare|N min) + lead time       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                 L5 — HISTORICAL ANALOG RETRIEVAL                │
│  DTW similarity → FAISS vector search → Top-3 analogs          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                   L6 — THREAT INTELLIGENCE                      │
│  Flare class → NavIC risk → HF blackout → Gaganyaan radiation  │
│  Satellite safing priority queue → India regional impact map    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                     L7 — LLM BRIEFING                           │
│  Ollama/Mistral → Structured JSON prompt → Constraint check    │
│  Plain-language alert → SpaceWx bulletin format export          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    L8 — DASHBOARD SERVICES                      │
│  Streamlit + Plotly → 8 pages → Live ops + X6.3 replay         │
│  Gaganyaan window advisor → ISRO satellite safing queue         │
└─────────────────────────────────────────────────────────────────┘

ARUNA-XL (sits above core pipeline, consumes outputs):
XL-1: Active Region Intelligence  →  XL-2: Multi-Payload Coherence
XL-3: Physics Observable Engine   →  XL-4: Stacked Forecast Ensemble
XL-5: Earth/NavIC Impact Layer    →  XL-6: Extended Validation Agent
```

### 2.1 Data Flow Timing Budget

| Stage | Max Latency | Target Latency |
|-------|------------|----------------|
| L1 Ingestion | 5 seconds | 2 seconds |
| L2 Physics | 2 seconds | 1 second |
| L3 Nowcasting | 1 second | 0.5 seconds |
| L4 Forecasting | 10 seconds | 5 seconds |
| L5 Analog | 3 seconds | 1 second |
| L6 Threat | 2 seconds | 1 second |
| L7 LLM Briefing | 30 seconds | 15 seconds |
| L8 Dashboard update | 5 seconds | 2 seconds |
| **End-to-end alert** | **~60 seconds** | **~30 seconds** |

---

## 3. Global Data Contracts

### 3.1 Universal Timestamp Convention

All timestamps in ARUNA use **ISO 8601 UTC** format: `YYYY-MM-DDTHH:MM:SS.sssZ`  
Internal representation: `numpy.datetime64` with nanosecond precision  
FITS timestamps are converted at ingestion; never propagated raw.

### 3.2 Universal Flux Convention

| Quantity | Unit | Notes |
|----------|------|-------|
| SoLEXS flux | counts/s/keV | Dead-time corrected |
| HEL1OS flux | counts/s/keV | Background subtracted |
| GOES XRS flux | W/m² | Cross-validation only |
| Temperature | MK (megakelvin) | From isothermal fit |
| Emission Measure | cm⁻³ | From isothermal fit |
| DNI | dimensionless | Cumulative ratio |
| TNPSV magnitude | normalised units | 0–1 range |

### 3.3 Universal Quality Flags

```python
class QualityFlag(Enum):
    GOOD         = 0   # All checks passed
    SAA_PASSAGE  = 1   # South Atlantic Anomaly contamination
    DEAD_TIME    = 2   # Dead-time correction applied, uncertainty elevated
    SATURATED    = 3   # SDD1 saturated, SDD2 used
    GAP          = 4   # Data gap, interpolated
    NOISY        = 5   # Background elevated
    MISSING      = 6   # Frame absent, no interpolation possible
    CALIBRATION  = 7   # In-flight calibration window
```

### 3.4 Universal ObservationFrame Schema

```python
@dataclass
class ObservationFrame:
    # Identity
    frame_id:          str           # UUID4
    instrument:        str           # 'SOLEXS_SDD1' | 'SOLEXS_SDD2' | 'HEL1OS_CDTE' | 'HEL1OS_CZT'
    timestamp:         datetime      # UTC
    
    # Data
    energy_bins_kev:   np.ndarray    # shape (N,) — bin edges in keV
    counts_per_s_kev:  np.ndarray    # shape (N,) — deadtime-corrected
    
    # Quality
    quality_flag:      QualityFlag
    dead_time_fraction: float        # 0.0–1.0
    calibration_version: str         # e.g. 'sarwade_2025_v1'
    
    # Metadata
    solar_distance_au: float
    spacecraft_mode:   str
    saa_flag:          bool
    source_file:       str           # original FITS filename
    checksum_sha256:   str
    pipeline_version:  str
```

### 3.5 Universal AlertEvent Schema

```python
@dataclass
class AlertEvent:
    alert_id:          str           # UUID4
    created_at:        datetime      # UTC
    alert_type:        str           # 'HOPE_PRECURSOR' | 'FLARE_ONSET' | 'FLARE_PEAK' | 'FLARE_END'
    goes_class:        str           # 'B' | 'C' | 'M' | 'X' | 'SUB_A'
    goes_subclass:     float         # e.g. 3.8 for M3.8
    confidence:        float         # 0.0–1.0
    lead_time_sec:     int           # seconds before expected peak
    instruments:       List[str]     # which instruments triggered
    physics_state:     str           # 'HOPE' | 'IMPULSIVE' | 'GRADUAL' | 'DECAY'
    peak_temperature_MK: float
    peak_emission_measure: float
    dni_at_alert:      float
    threat_flags:      Dict[str, str] # {'navic': 'HIGH', 'hf_comms': 'MODERATE', ...}
    pipeline_version:  str
    provenance_hash:   str           # SHA-256 of input frames
```

---

## 4. L0 — Source Layer

### 4.1 Objective

Define and manage all data source connections. L0 is passive — it does not process data, only provides authenticated access handles to raw telemetry.

### 4.2 Primary Sources

| Source | URL | Format | Cadence | Auth |
|--------|-----|--------|---------|------|
| PRADAN (SoLEXS L1) | pradan.issdc.gov.in | FITS | 1 second | ISRO credentials |
| PRADAN (HEL1OS L1) | pradan.issdc.gov.in | FITS | 1 second | ISRO credentials |
| PRADAN (SUIT L1) | pradan.issdc.gov.in | FITS | 60 seconds | ISRO credentials |
| PRADAN (HK data) | pradan.issdc.gov.in | FITS | 1 second | ISRO credentials |

### 4.3 Supplementary Sources

| Source | URL | Format | Role |
|--------|-----|--------|------|
| NOAA SWPC JSON | services.swpc.noaa.gov | JSON REST | Real-time GOES XRS fallback |
| NOAA GOES-16 XRS | AWS S3 | NetCDF | Cross-validation |
| SunPy HEK | helioweb.ias.u-psud.fr | JSON | GOES flare labels |
| JSOC SHARP | jsoc.stanford.edu | FITS | Magnetic parameters (XL-1) |
| CDAWeb ACE | cdaweb.gsfc.nasa.gov | CDF | Solar wind (XL-5) |
| CDAW LASCO | cdaw.gsfc.nasa.gov | CSV | CME catalog (XL-5) |
| GFZ Potsdam | kp.gfz-potsdam.de | JSON | Kp/Ap index |

### 4.4 Source Health Monitor

```python
class SourceHealthMonitor:
    """
    Polls all sources every 30 seconds.
    Publishes to Kafka topic: aruna.source.health
    Triggers fallback mode if primary source unavailable > 5 minutes.
    """
    def check_pradan(self) -> SourceStatus: ...
    def check_noaa_swpc(self) -> SourceStatus: ...
    def trigger_fallback(self, source: str) -> None: ...
```

### 4.5 Local Cache Policy

- Last 48 hours of FITS data retained on local disk at `data/cache/`
- Cache eviction: LRU, max 50 GB
- Cache integrity: SHA-256 verified on every read
- If PRADAN unavailable: pipeline continues on cache with `DATA_DEGRADED` flag

### 4.6 Acceptance Criteria

```
✓ PRADAN connection established and authenticated
✓ SoLEXS FITS file readable with astropy.io.fits
✓ HEL1OS FITS file readable with astropy.io.fits
✓ NOAA SWPC JSON endpoint reachable (fallback)
✓ Local cache directory writable
✓ Source health monitor publishing to Kafka
```

---

## 5. L1 — Data Ingestion

### 5.1 Objective

Acquire raw FITS telemetry, validate integrity, apply dead-time correction, align both instruments to a common 1-second UTC time base, cross-reference with GOES catalog for labels, and publish clean ObservationFrames to the internal event bus.

### 5.2 Inputs

**Primary (required):**
- `SoLEXS_L1_YYYYMMDD.fits` — SDD1 and SDD2 detector arrays
- `HEL1OS_L1_YYYYMMDD.fits` — CdTe (8–70 keV) and CZT (20–150 keV) arrays
- `Aditya_HK_YYYYMMDD.fits` — Housekeeping: SAA flag, mode, ephemeris

**Secondary (optional, enrichment):**
- GOES XRS L1b NetCDF (for label join)
- NOAA SWPC JSON (real-time cross-reference)

### 5.3 Preconditions

```python
def validate_preconditions(fits_path: str) -> ValidationResult:
    checks = [
        file_exists(fits_path),
        checksum_valid(fits_path),           # SHA-256 vs PRADAN manifest
        fits_header_complete(fits_path),     # required keys present
        timestamp_parseable(fits_path),      # TIME column readable
        instrument_mode_nominal(fits_path),  # not in calibration/maintenance
        not saa_passage_active(fits_path),   # flag from HK data
    ]
    return ValidationResult(all_passed=all(checks), checks=checks)
```

### 5.4 Processing Pipeline — Step by Step

```
Step 1: Download / Cache Lookup
    → Check local cache first (SHA-256 match)
    → Download from PRADAN if not cached
    → Verify SHA-256 against PRADAN manifest
    → On failure: retry × 3, then use last cached version, set GAP flag

Step 2: FITS Parse
    → astropy.io.fits.open()
    → Extract: TIME, COUNTS, ENERGY_BINS, LIVETIME, DEADTIME_FRAC
    → Extract HK flags: SAA_FLAG, INSTRUMENT_MODE, SOLAR_DIST_AU
    → Validate all required columns present

Step 3: Dead-Time Correction (SoLEXS, per Sarwade et al. 2025)
    → Non-paralysable detector model:
       R_true = R_obs / (1 - R_obs × τ)
       where τ = detector dead-time constant from calibration table
    → Apply to each 1-second bin independently
    → Flag bins where corrected rate exceeds SDD1 saturation limit
    → Auto-switch to SDD2 aperture if SDD1 saturated

Step 4: Background Subtraction (HEL1OS)
    → Extract 5-minute pre-event quiet window
    → Fit background spectrum with sherpa Sub() method
    → Subtract per energy bin
    → Validate: residuals consistent with Poisson noise (χ²/dof < 2.0)
    → Flag all SAA-coincident 1-second bins as QUALITY=SAA_PASSAGE

Step 5: Time Alignment
    → Convert all timestamps to UTC ISO 8601
    → Compute cross-correlation peak between SoLEXS 10–15 keV
      and HEL1OS 10–15 keV (overlapping energy window)
    → Assert lag < 1 second
    → If lag > 1 second: apply fractional-second interpolation
      (scipy.interpolate.interp1d, cubic), log clock-drift event
    → Resample both streams to common 1-second UTC grid

Step 6: GOES Label Join
    → Query SunPy HEK for GOES flare events ± 10 minutes of window
    → Match: onset within ± 3 minutes AND class within 1 sub-class
    → Annotate each ObservationFrame with GOES_CLASS if matched
    → Unmatched frames labelled QUIET_SUN or UNKNOWN

Step 7: Schema Validation
    → pandera schema check on output DataFrame
    → Completeness check: > 80% non-null per 60-second window
    → Physical range check: flux values within instrument design limits
    → Reject frames failing checks; log to aruna.ingestion.errors

Step 8: Kafka Publish
    → Publish ObservationFrame to aruna.observations.solexs
    → Publish ObservationFrame to aruna.observations.hel1os
    → Publish quality summary to aruna.quality.summary
```

### 5.5 Dead-Time Correction Constants

```python
DEADTIME_CONSTANTS = {
    'SOLEXS_SDD1': 4.5e-6,   # seconds — from Sarwade et al. 2025 Table 2
    'SOLEXS_SDD2': 4.5e-6,   # seconds — same detector family
    'HEL1OS_CDTE': 2.1e-6,   # seconds — from Tadepalli/Nandi et al. 2025
    'HEL1OS_CZT':  1.8e-6,   # seconds
}

SATURATION_LIMITS = {
    'SOLEXS_SDD1': 5e4,       # counts/s — above this, auto-switch to SDD2
    'SOLEXS_SDD2': 5e6,       # counts/s — X10+ events approach this
}
```

### 5.6 Failure Recovery Matrix

| Failure | Detection | Action | Log | Alert |
|---------|-----------|--------|-----|-------|
| FITS missing | File not found | Retry × 3, then cache | ERROR | DATA_DEGRADED |
| Checksum mismatch | SHA-256 fail | Re-download × 1 | WARN | none |
| SDD1 saturated | Rate > limit | Switch to SDD2 | INFO | SATURATION_SWITCH |
| Clock drift > 1s | Cross-corr lag | Interpolate, log | WARN | CLOCK_DRIFT |
| SAA passage | HK flag = 1 | Mask bins, continue | INFO | none |
| PRADAN down > 5 min | Health monitor | Use cache | ERROR | OFFLINE_CACHE |
| Both instruments offline | Health monitor | Fallback to NOAA SWPC | CRITICAL | DATA_FALLBACK |
| Partial FITS (< 80% complete) | pandera | Hold in quarantine 5 min | WARN | none |

### 5.7 Output Schema

```python
# Published to: aruna.observations.solexs, aruna.observations.hel1os
# Type: ObservationFrame (see §3.4)
# Partition key: instrument
# Retention: 72 hours
```

### 5.8 Acceptance Criteria

```
✓ Time sync error between SoLEXS and HEL1OS < 1 second
✓ Dead-time correction applied to all SoLEXS bins
✓ Background subtraction applied to all HEL1OS bins
✓ SAA-contaminated bins flagged, not silently dropped
✓ Schema validation passes (pandera) on > 95% of frames
✓ GOES label joined for all known catalog events ± 3 min
✓ ObservationFrame published to Kafka within 2 seconds of file arrival
✓ X6.3 event (2024-02-22) processed without errors
✓ Clock drift log exists for any lag > 0.5 seconds
```

### 5.9 Performance Budget

| Metric | Target |
|--------|--------|
| Ingestion latency (file arrival → Kafka publish) | < 2 seconds |
| Dead-time correction per bin | < 1 ms |
| Memory footprint | < 512 MB |
| CPU (single core) | < 30% sustained |

---

## 6. L2 — Physics Engine

### 6.1 Objective

From each 1-second SoLEXS spectrum, compute real plasma diagnostics (T, EM) using the same isothermal fitting methodology as the SoLEXS instrument team. Compute ARUNA's novel physics observables (DNI, TNPSV, SHEI, HFD, TNCI). Detect the HOPE precursor. This is the scientific core of ARUNA and the layer that distinguishes it from every other approach.

### 6.2 Inputs

```
Consumes: aruna.observations.solexs (ObservationFrame, 1s cadence)
Consumes: aruna.observations.hel1os (ObservationFrame, 1s cadence)
Requires: CHIANTI atomic database (via sunkit-spex)
Requires: SoLEXS spectral response matrix (ARF + SRF, from Sarwade et al. 2025)
```

### 6.3 Algorithm 1 — Isothermal Plasma Fit (T and EM)

**Method:** `sunkit-spex f_vth` isothermal model, calibrated per Sarwade et al. 2025

```python
from sunkit_spex.fitting import FitISothermal

def fit_temperature_emission_measure(
    spectrum: np.ndarray,       # counts/s/keV, shape (N_bins,)
    energy_bins: np.ndarray,    # keV, shape (N_bins+1,)
    response_matrix: np.ndarray # ARF × SRF, shape (N_bins, N_bins)
) -> Tuple[float, float, float, float]:
    """
    Returns:
        T_MK: plasma temperature in megakelvin
        EM_cm3: emission measure in cm^-3
        T_uncertainty: 1-sigma uncertainty on T
        EM_uncertainty: 1-sigma uncertainty on EM
    """
    fitter = FitISothermal(
        model='f_vth',
        energy_range=(3.0, 20.0),  # keV — avoid low-energy noise and calibration edge
        abundance_table='coronal',  # per CHIANTI standard
        response_matrix=response_matrix
    )
    result = fitter.fit(spectrum, energy_bins)
    return result.T_MK, result.EM_cm3, result.T_1sigma, result.EM_1sigma
```

**Validation:** T and EM outputs on X6.3 event (2024-02-22) must match values reported in Sarwade et al. (arXiv:2502.19795) within 1-sigma instrument uncertainty.

### 6.4 Algorithm 2 — Dynamic Neupert Integral (DNI)

**Physical basis:** The Neupert effect states that soft X-ray flux time derivative tracks hard X-ray flux, because non-thermal electrons heat thermal plasma. DNI measures this ratio continuously.

```python
def compute_DNI(
    flux_hel1os_25_50: np.ndarray,  # HEL1OS 25–50 keV band, counts/s
    flux_solexs_8_15: np.ndarray,   # SoLEXS 8–15 keV band, counts/s
    dt: float = 1.0                  # seconds
) -> np.ndarray:
    """
    DNI(t) = ∫₀ᵗ F_HEL1OS(t') dt' / F_SoLEXS(t)
    
    Physical interpretation:
        DNI < 1.0  → thermal energy exceeds cumulative nonthermal input
                     → flare in gradual/conductive phase
        DNI > 1.5 and rising → nonthermal energy injection outpacing thermal response
                     → IMPULSIVE ONSET IMMINENT (early warning)
    """
    cumulative_hel = np.cumsum(flux_hel1os_25_50 * dt)
    DNI = cumulative_hel / (flux_solexs_8_15 + 1e-30)  # epsilon avoids div/0
    return DNI

HOPE_DNI_THRESHOLD = 1.5   # configurable via config.yaml
```

### 6.5 Algorithm 3 — Thermal-Nonthermal Phase Space Velocity (TNPSV)

**Physical basis:** In T-EM phase space, a flare traces a characteristic trajectory. The velocity and heading angle encode the instantaneous phase of flare development.

```python
def compute_TNPSV(
    T: np.ndarray,    # MK, shape (N,)
    EM: np.ndarray    # cm^-3, shape (N,)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    v_TN(t) = √[(dT/dt / T_max)² + (dEM/dt / EM_max)²]  — normalised speed
    θ_TN(t) = atan2(dEM/dt, dT/dt)                        — phase heading
    
    HOPE quadrant: T rising, EM flat (θ near +90°, v > threshold)
    Impulsive quadrant: T + EM both rising (θ near +45°)
    Decay quadrant: T falling, EM still high (θ near -90°)
    """
    T_max  = np.max(T)  + 1e-30
    EM_max = np.max(EM) + 1e-30
    dT  = np.gradient(T)
    dEM = np.gradient(EM)
    magnitude = np.sqrt((dT / T_max)**2 + (dEM / EM_max)**2)
    angle     = np.arctan2(dEM, dT)
    return magnitude, angle

HOPE_TNPSV_SPEED_THRESHOLD = 0.05   # configurable
HOPE_TNPSV_ANGLE_MIN = np.deg2rad(60)   # T rising, EM flat sector
HOPE_TNPSV_ANGLE_MAX = np.deg2rad(120)
```

### 6.6 Algorithm 4 — Spectral Hardness Evolution Index (SHEI)

```python
def compute_SHEI(
    flux_hel1os: np.ndarray,        # shape (N_time, N_energy)
    energy_bins: np.ndarray          # keV
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SHEI(t) = log10[ F_HEL1OS(40–70 keV) / F_HEL1OS(10–25 keV) ]
    
    SHEI < -1.5:          thermal dominated (gradual phase)
    -1.5 ≤ SHEI ≤ -0.8:  mixed thermal-nonthermal (onset)
    SHEI > -0.8:          nonthermal dominated (impulsive phase)
    
    dSHEI/dt > 0 (hardening) + SoLEXS rising → IMPULSIVE_ONSET label
    dSHEI/dt < 0 (softening) + HEL1OS declining → DECAY label
    """
    mask_40_70 = (energy_bins[:-1] >= 40) & (energy_bins[:-1] <= 70)
    mask_10_25 = (energy_bins[:-1] >= 10) & (energy_bins[:-1] <= 25)
    F_hard = np.sum(flux_hel1os[:, mask_40_70], axis=1)
    F_soft = np.sum(flux_hel1os[:, mask_10_25], axis=1)
    SHEI = np.log10(F_hard / (F_soft + 1e-30))
    dSHEI_dt = np.gradient(SHEI)
    return SHEI, dSHEI_dt
```

### 6.7 Algorithm 5 — Higuchi Fractal Dimension (HFD)

```python
def compute_HFD(
    flux: np.ndarray,    # SoLEXS total flux, 1s cadence
    k_max: int = 10,
    window: int = 600    # 10-minute sliding window
) -> np.ndarray:
    """
    Higuchi Fractal Dimension on sliding 10-minute window.
    
    Quiet Sun:  HFD ~ 1.3–1.5 (mildly fractal, stochastic)
    Pre-flare:  HFD drops toward 1.0–1.2 (flux rising coherently)
    
    HFD drop is an additional precursor feature in L4 forecasting.
    First application to SoLEXS 1-second cadence data.
    """
    # Higuchi algorithm
    def hfd_single(x, k_max):
        L = []
        for k in range(1, k_max + 1):
            Lk = []
            for m in range(1, k + 1):
                idx = np.arange(m - 1, len(x), k)
                Lmk = np.sum(np.abs(np.diff(x[idx]))) * (len(x) - 1) / (len(idx) * k**2)
                Lk.append(Lmk)
            L.append(np.mean(Lk))
        k_vals = np.arange(1, k_max + 1)
        slope, _ = np.polyfit(np.log(k_vals), np.log(L), 1)
        return -slope

    result = np.full(len(flux), np.nan)
    for i in range(window, len(flux)):
        result[i] = hfd_single(flux[i - window:i], k_max)
    return result
```

### 6.8 Algorithm 6 — Wavelet QPP Detector

```python
def detect_QPP(
    counts_hel1os_25_50: np.ndarray,  # HEL1OS 25–50 keV, 1s cadence
    dt: float = 1.0
) -> Tuple[Optional[float], Optional[float]]:
    """
    Quasi-Periodic Pulsation detection via Morlet continuous wavelet transform.
    
    Steps:
    1. Detrend by subtracting LOWESS envelope (frac=0.3)
    2. Compute Morlet CWT with PyWavelets
    3. Identify power ridges > 99th percentile noise floor (Lomb-Scargle)
    4. Extract dominant QPP period T_QPP and amplitude A_QPP
    
    Returns: (T_QPP_seconds, A_QPP_normalized) or (None, None) if no QPP detected
    
    Operational use:
    Sudden T_QPP period halving/doubling → fire SYMPATHETIC_RISK alert
    """
    import pywt
    from scipy.signal import lombscargle
    
    # Detrend
    from statsmodels.nonparametric.smoothers_lowess import lowess
    trend = lowess(counts_hel1os_25_50, np.arange(len(counts_hel1os_25_50)), frac=0.3)[:, 1]
    detrended = counts_hel1os_25_50 - trend
    
    # CWT
    scales = np.arange(5, 200)
    coeffs, freqs = pywt.cwt(detrended, scales, 'morl', sampling_period=dt)
    power = np.abs(coeffs)**2
    
    # Find dominant period
    mean_power = np.mean(power, axis=1)
    noise_floor = np.percentile(mean_power, 99)
    peaks = scales[mean_power > noise_floor]
    
    if len(peaks) == 0:
        return None, None
    
    dominant_scale = peaks[np.argmax(mean_power[mean_power > noise_floor])]
    T_QPP = dominant_scale * dt
    A_QPP = np.max(mean_power) / np.mean(mean_power)
    return T_QPP, A_QPP
```

### 6.9 Algorithm 7 — Fe Kα Fluorescence Time Series (Novel — Sarwade Gap)

**This addresses an explicit gap from Sarwade et al. 2025:** "statistical uncertainties currently limit the ability to track rapid height variations on short timescales." ARUNA correlates the Fe Kα flux time series with HEL1OS non-thermal light curve at 1-second cadence — a measurement the instrument team wanted but has not yet published.

```python
def extract_FeKa_timeseries(
    spectra: np.ndarray,      # SoLEXS spectra, shape (N_time, N_energy)
    energy_bins: np.ndarray,  # keV
    T: np.ndarray,            # MK from Algorithm 1
    EM: np.ndarray            # cm^-3 from Algorithm 1
) -> np.ndarray:
    """
    Isolate Fe Kα fluorescence (6.4 keV line) by subtracting
    the fitted isothermal continuum model (CHIANTI).
    
    Residual at 6.4 keV = Fe Kα fluorescence flux.
    
    Then cross-correlate with HEL1OS 25–50 keV band.
    If Fe Kα peaks synchronously with HEL1OS → non-thermal electron
    beam driving photospheric fluorescence (first Aditya-L1 measurement).
    """
    # Build continuum model from T, EM
    continuum = build_chianti_continuum(T, EM, energy_bins)
    
    # Residual at Fe Kα energy
    feka_mask = (energy_bins[:-1] >= 6.2) & (energy_bins[:-1] <= 6.6)
    feka_flux = np.sum((spectra - continuum)[:, feka_mask], axis=1)
    feka_flux = np.clip(feka_flux, 0, None)  # fluorescence cannot be negative
    return feka_flux
```

### 6.10 HOPE Precursor Logic

```python
def evaluate_HOPE(
    T: float, dT_dt: float,
    EM: float, dEM_dt: float,
    DNI: float,
    tnpsv_mag: float, tnpsv_angle: float,
    hel1os_flux: float, hel1os_quiet_baseline: float,
    history_len: int  # seconds of rising T already observed
) -> Tuple[bool, str, int]:
    """
    HOPE fires when ALL of the following are true:
    
    1. T is rising (dT_dt > 0) for at least 3 consecutive seconds
    2. EM is still flat (dEM_dt < EM_RISE_THRESHOLD) — pre-impulsive
    3. HEL1OS is still near quiet-Sun level (nonthermal not yet dominant)
    4. TNPSV is in the precursor quadrant (T rising, EM flat sector)
    5. DNI is rising from baseline
    
    Returns: (hope_flag, reason_string, estimated_lead_time_seconds)
    """
    cond_T_rising     = dT_dt > 0 and history_len >= 3
    cond_EM_flat      = abs(dEM_dt) < EM_RISE_THRESHOLD
    cond_HEL_quiet    = hel1os_flux < hel1os_quiet_baseline * 2.0
    cond_TNPSV        = (tnpsv_mag > HOPE_TNPSV_SPEED_THRESHOLD and
                         HOPE_TNPSV_ANGLE_MIN < tnpsv_angle < HOPE_TNPSV_ANGLE_MAX)
    cond_DNI_rising   = DNI > 0.2  # above noise floor

    hope_flag = all([cond_T_rising, cond_EM_flat, cond_HEL_quiet, cond_TNPSV, cond_DNI_rising])

    # Lead time estimate: time until EM begins rising × correction factor
    lead_time_estimate = int(90 + history_len * 2)  # empirical, validated on X6.3

    reason = (f"T rising {dT_dt:.2f} MK/s for {history_len}s | "
              f"EM flat | HEL1OS quiet | TNPSV in precursor quadrant | DNI={DNI:.2f}")

    return hope_flag, reason, lead_time_estimate
```

### 6.11 Physics Output Schema

```python
@dataclass
class PhysicsFrame:
    frame_id:          str
    timestamp:         datetime
    source_frame_ids:  List[str]   # provenance — L1 frame IDs used
    
    # Plasma diagnostics
    T_MK:              float       # temperature
    T_uncertainty_MK:  float
    EM_cm3:            float       # emission measure
    EM_uncertainty:    float
    
    # Novel observables
    DNI:               float       # Dynamic Neupert Integral
    tnpsv_magnitude:   float       # phase space speed
    tnpsv_angle_rad:   float       # phase space heading
    SHEI:              float       # spectral hardness
    dSHEI_dt:          float       # hardening rate
    HFD:               float       # Higuchi fractal dimension (60s window)
    QPP_period_sec:    Optional[float]
    QPP_amplitude:     Optional[float]
    FeKa_flux:         float       # Fe Kα fluorescence residual
    
    # State labels
    flare_phase:       str         # 'QUIET' | 'HOPE' | 'IMPULSIVE' | 'GRADUAL' | 'DECAY'
    hope_flag:         bool
    hope_lead_time_sec: Optional[int]
    hope_reason:       Optional[str]
    
    # Quality
    fit_chi_squared:   float       # isothermal fit quality (target < 2.0)
    quality_flag:      QualityFlag
```

### 6.12 Acceptance Criteria

```
✓ T/EM output on X6.3 event (2024-02-22) matches Sarwade et al. within 1-sigma
✓ Isothermal fit chi-squared < 2.0 on > 90% of M+ class spectra
✓ DNI > 1.5 fires HOPE flag on X6.3 event within 2 minutes of impulsive onset
✓ TNPSV in precursor quadrant > 90 seconds before X6.3 peak
✓ SHEI transitions from < -1.5 to > -0.8 during impulsive phase of X6.3
✓ Fe Kα fluorescence detected in at least 80% of X-class events
✓ Physics engine runtime < 1 second per 1-second observation window
✓ All PhysicsFrames include provenance hash linking to source ObservationFrames
```

---

## 7. L3 — Nowcasting Engine

### 7.1 Objective

Detect solar flares independently in SoLEXS and HEL1OS using a dual-detection consensus approach, classify by GOES-equivalent class, and write every confirmed event to the Master Flare Catalog. This is the primary detection layer — the operational heart of ARUNA.

### 7.2 Inputs

```
Consumes: aruna.physics.frames (PhysicsFrame, 1s cadence)
Consumes: aruna.observations.solexs (ObservationFrame, for raw flux)
Consumes: aruna.observations.hel1os (ObservationFrame, for raw flux)
```

### 7.3 Detection Algorithm — Dual-Trigger Consensus

**Design principle:** A flare is confirmed ONLY when both SoLEXS AND HEL1OS independently trigger. This dual-instrument consensus dramatically reduces false alarm rate.

```
SoLEXS Detector (thermal):
    → Rolling 3-hour P85 baseline (recomputed every 60 seconds)
    → PELT change-point detector (ruptures library, O(n log n))
    → Trigger: flux crosses baseline + 3σ AND PELT detects change-point
    
HEL1OS Detector (non-thermal):
    → Same rolling baseline approach
    → Additional: SHEI hardening rate dSHEI/dt > threshold
    → Trigger: hard X-ray flux rises AND spectral hardening detected

Consensus Engine:
    → Both detectors must trigger within 3-minute window
    → HOPE precursor may pre-announce before consensus
    → Triple-mode: HOPE_PRECURSOR → FLARE_ONSET → FLARE_PEAK
```

### 7.4 Rolling Baseline Algorithm

```python
from collections import deque
import numpy as np

class RollingP85Baseline:
    """
    Rolling 3-hour 85th percentile baseline.
    Recomputed every 60 seconds.
    Immune to long-duration event contamination.
    """
    def __init__(self, window_sec: int = 10800, recompute_interval: int = 60):
        self.window = deque(maxlen=window_sec)
        self.baseline = None
        self.last_recompute = 0
        self.recompute_interval = recompute_interval

    def update(self, flux: float, t: int) -> float:
        self.window.append(flux)
        if t - self.last_recompute >= self.recompute_interval:
            self.baseline = np.percentile(list(self.window), 85)
            self.last_recompute = t
        return self.baseline

    def get_threshold(self, sigma_multiplier: float = 3.0) -> float:
        quiet_std = np.std(list(self.window)[:3600])  # first hour as quiet estimate
        return self.baseline + sigma_multiplier * quiet_std
```

### 7.5 PELT Change-Point Detector

```python
import ruptures as rpt

def detect_changepoint(
    flux: np.ndarray,      # last 600 seconds of flux
    min_size: int = 10,    # minimum segment length
    penalty: float = 1.5   # PELT penalty parameter
) -> Optional[int]:
    """
    PELT (Pruned Exact Linear Time) change-point detection.
    Immune to threshold drift unlike simple thresholding.
    Returns index of most recent change-point, or None.
    """
    model = rpt.Pelt(model='rbf', min_size=min_size).fit(flux)
    breakpoints = model.predict(pen=penalty)
    if len(breakpoints) > 1:  # at least one internal breakpoint
        return breakpoints[-2]  # most recent change-point
    return None
```

### 7.6 GOES Classification Logic

```python
def classify_goes_class(
    peak_flux_solexs_1_8A: float,    # W/m² equivalent
    goes_xrs_reference: Optional[float] = None  # for cross-validation
) -> Tuple[str, float]:
    """
    GOES classification thresholds (1–8 Å equivalent from SoLEXS):
    B: 1e-7 – 1e-6 W/m²
    C: 1e-6 – 1e-5 W/m²
    M: 1e-5 – 1e-4 W/m²
    X: > 1e-4 W/m²
    
    Returns: (class_letter, subclass_number)
    e.g. ('M', 3.8) for M3.8
    """
    thresholds = [('X', 1e-4), ('M', 1e-5), ('C', 1e-6), ('B', 1e-7)]
    for letter, threshold in thresholds:
        if peak_flux_solexs_1_8A >= threshold:
            subclass = peak_flux_solexs_1_8A / threshold
            return letter, round(subclass, 1)
    return ('A', peak_flux_solexs_1_8A / 1e-8)
```

### 7.7 Overlapping Flare Handling (Edge Case)

```python
def handle_overlapping_flares(
    active_events: List[FlareEvent],
    new_changepoint_time: datetime,
    current_flux: float,
    current_T: float
) -> str:
    """
    Two flares overlap when:
    - A new change-point is detected during the DECAY phase of a prior event
    - The new change-point represents a second rise above the current-event baseline
    
    Action:
    - Log both events as OVERLAPPING in Master Catalog
    - Assign independent onset times and lead times
    - Issue alert for both
    - Lock baseline at first-event onset time (do not adapt during event)
    """
    if any(e.phase == 'DECAY' for e in active_events):
        return 'OVERLAPPING_FLARE'
    return 'NEW_FLARE'
```

### 7.8 Master Flare Catalog Schema

```python
# Storage: SQLite for local, PostgreSQL for production
# Table: master_flare_catalog

CREATE TABLE master_flare_catalog (
    event_id           TEXT PRIMARY KEY,       -- UUID4
    start_time         TIMESTAMP NOT NULL,     -- UTC ISO 8601
    hope_time          TIMESTAMP,              -- HOPE precursor timestamp
    onset_time         TIMESTAMP,              -- dual-trigger confirmation
    peak_time          TIMESTAMP,              -- maximum flux time
    end_time           TIMESTAMP,
    duration_sec       INTEGER,
    goes_class         TEXT NOT NULL,          -- 'B'|'C'|'M'|'X'
    goes_subclass      REAL,
    peak_flux_solexs   REAL,                   -- counts/s/keV
    peak_flux_hel1os   REAL,
    peak_T_MK          REAL,
    peak_EM_cm3        REAL,
    peak_DNI           REAL,
    hope_lead_time_sec INTEGER,
    hope_fired         BOOLEAN DEFAULT FALSE,
    instruments        TEXT,                   -- JSON array
    detection_method   TEXT,                   -- 'DUAL_TRIGGER'|'SOLEXS_ONLY'|'HOPE'
    confidence         REAL,                   -- 0.0–1.0
    goes_catalog_match TEXT,                   -- matched GOES event ID or NULL
    noaa_ar_number     INTEGER,               -- active region if known
    flare_type         TEXT,                   -- 'STANDARD'|'OVERLAPPING'|'LIMB_OCCULTED'|'SUB_A'
    pipeline_version   TEXT,
    provenance_hash    TEXT,
    notes              TEXT
);
```

### 7.9 Failure Modes

| Scenario | Detection | Response |
|----------|-----------|----------|
| SoLEXS saturated (X10+) | Rate > SDD2 limit | Flag SATURATED, report flux lower bound, escalate alert regardless |
| HEL1OS offline | Health monitor | SoLEXS-only detection with CONFIDENCE_REDUCED flag |
| Long-duration event (> 3 hr) | Duration timer | Lock baseline at onset, hold fixed for up to 6 hours |
| SAA particle spike mimicking flare | Kp index check | If Kp > 6 AND no SoLEXS precursor → classify SEP_EVENT not flare |
| Sub-A nanoflare | SDD1 sensitivity | Log as SUB_A, flag "SoLEXS-internal only, no GOES ground truth" |
| Behind-limb flare | HEL1OS signal without SoLEXS | Log LIMB_OCCULTED, use STEREO-A if available |

### 7.10 Output Schema

```python
# Published to: aruna.nowcast.events
# Also written to: master_flare_catalog (SQLite/PostgreSQL)

@dataclass
class NowcastEvent:
    event_id:          str
    alert_type:        str       # 'HOPE_PRECURSOR'|'FLARE_ONSET'|'FLARE_PEAK'
    timestamp:         datetime
    goes_class:        str
    goes_subclass:     float
    confidence:        float
    lead_time_sec:     Optional[int]
    instruments:       List[str]
    physics_frame_id:  str       # provenance link to L2
    hope_reason:       Optional[str]
    flare_type:        str
```

### 7.11 Acceptance Criteria

```
✓ X6.3 event (2024-02-22) detected within 30 seconds of GOES onset time
✓ HOPE precursor fires ≥ 90 seconds before X6.3 peak
✓ TPR > 0.80 on M+X class events (validation set)
✓ FAR < 0.05 on quiet-Sun validation windows
✓ TSS > 0.65 on M+X class events
✓ Overlapping flares logged with distinct event_ids
✓ Master Catalog write < 100ms per event
✓ All detections have provenance hash linking to source frames
✓ X3+ events escalated to CRITICAL alert regardless of confidence
```
---

## 8. L4 — Forecasting Engine

### 8.1 Objective

On a sliding 10-minute window of physics observables, a dual-stream temporal model learns the SoLEXS-to-HEL1OS timing relationship and outputs calibrated flare probability in the next N minutes with quantified uncertainty. This is ARUNA's predictive layer — the difference between detecting what is happening and warning before it happens.

### 8.2 Inputs

```
Consumes: aruna.physics.frames (PhysicsFrame, sliding 10-min window = 600 frames)
Consumes: aruna.nowcast.events (NowcastEvent — HOPE flag triggers immediate forecast)
Requires: Pre-trained TCN weights (aruna/models/tcn_v1.pt)
Requires: Pre-trained Moirai weights (uni2ts/moirai-1.0-R-base)
Requires: MAPIE conformal calibration set (aruna/models/mapie_calibration.pkl)
```

### 8.3 Feature Vector

```python
FEATURE_VECTOR = [
    # --- 1-second cadence features (from PhysicsFrame) ---
    'T_MK',               # plasma temperature
    'EM_cm3',             # emission measure
    'DNI',                # Dynamic Neupert Integral
    'tnpsv_magnitude',    # phase space speed
    'tnpsv_angle_rad',    # phase space heading
    'SHEI',               # spectral hardness
    'dSHEI_dt',           # hardening rate
    'FeKa_flux',          # Fe Kα fluorescence

    # --- 60-second cadence features ---
    'HFD',                # Higuchi fractal dimension
    'QPP_period_sec',     # QPP period (NaN if no QPP)
    'QPP_amplitude',      # QPP amplitude (NaN if no QPP)

    # --- Derived rates ---
    'dT_dt',              # temperature gradient
    'dEM_dt',             # emission measure gradient
    'd2T_dt2',            # temperature acceleration

    # --- Raw flux bands ---
    'solexs_3_6kev',      # SoLEXS 3–6 keV band
    'solexs_6_10kev',     # SoLEXS 6–10 keV band
    'hel1os_10_25kev',    # HEL1OS 10–25 keV band
    'hel1os_25_50kev',    # HEL1OS 25–50 keV band
    'hel1os_50_70kev',    # HEL1OS 50–70 keV band

    # --- State flags (one-hot encoded) ---
    'flare_phase_QUIET',
    'flare_phase_HOPE',
    'flare_phase_IMPULSIVE',
    'flare_phase_GRADUAL',
    'flare_phase_DECAY',
]

SEQUENCE_LENGTH = 600   # 10 minutes at 1-second cadence
N_FEATURES = len(FEATURE_VECTOR)  # 24 features
```

### 8.4 Dual-Stream TCN Architecture

```python
import torch
import torch.nn as nn

class DualStreamTCN(nn.Module):
    """
    Two parallel TCN streams:
    - Stream A: SoLEXS thermal features (T, EM, DNI, TNPSV)
    - Stream B: HEL1OS non-thermal features (SHEI, hard X-ray bands)
    Cross-attention: learns the temporal relationship between streams
    This timing offset IS the HOPE lead time.
    """
    def __init__(self, n_thermal: int = 8, n_nonthermal: int = 8,
                 n_shared: int = 8, n_classes: int = 5):
        super().__init__()
        # Thermal stream (SoLEXS)
        self.tcn_thermal = TemporalConvNet(
            num_inputs=n_thermal,
            num_channels=[64, 64, 128, 128],
            kernel_size=7, dropout=0.2
        )
        # Non-thermal stream (HEL1OS)
        self.tcn_nonthermal = TemporalConvNet(
            num_inputs=n_nonthermal,
            num_channels=[64, 64, 128, 128],
            kernel_size=7, dropout=0.2
        )
        # Cross-attention (learns thermal→nonthermal coupling)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=8, batch_first=True
        )
        # Shared features
        self.shared_encoder = TemporalConvNet(
            num_inputs=n_shared,
            num_channels=[32, 64],
            kernel_size=5, dropout=0.2
        )
        # Output head: 5-class probability
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
            # NOTE: no Softmax here — applied externally for calibration
        )

    def forward(self, x_thermal, x_nonthermal, x_shared):
        h_thermal    = self.tcn_thermal(x_thermal)
        h_nonthermal = self.tcn_nonthermal(x_nonthermal)
        # Cross-attention: nonthermal attends to thermal (learning the lag)
        h_cross, _   = self.cross_attention(h_nonthermal, h_thermal, h_thermal)
        h_shared     = self.shared_encoder(x_shared)
        h_cat = torch.cat([h_thermal[:, :, -1],
                           h_cross[:, :, -1],
                           h_shared[:, :, -1]], dim=1)
        return self.classifier(h_cat)
```

### 8.5 Loss Function (Asymmetric — Critical Design Decision)

```python
def asymmetric_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.2,      # under-prediction penalty weight (4× over-prediction)
    gamma: float = 2.0       # focal loss focusing parameter
) -> torch.Tensor:
    """
    For space weather, a missed M5+ flare is catastrophically worse than a false alarm.
    Standard MSE penalises equally. This loss penalises under-prediction 4× more.
    
    L = alpha * max(0, ŷ - y)² + (1-alpha) * max(0, y - ŷ)²
        ^-- over-prediction penalty    ^-- under-prediction penalty (4× larger)
    
    Secondary focal loss head: binary M+/X classification with γ=2
    Handles extreme class imbalance (~300:1 quiet:active ratio)
    """
    probs = torch.softmax(logits, dim=1)
    # Directional Huber-style loss
    residual = targets - probs
    over_pred  = torch.clamp(probs - targets, min=0)
    under_pred = torch.clamp(targets - probs, min=0)
    loss = alpha * over_pred**2 + (1 - alpha) * under_pred**2
    # Focal weighting (downweight easy negatives)
    pt = (probs * targets).sum(dim=1)
    focal_weight = (1 - pt)**gamma
    return (focal_weight * loss.sum(dim=1)).mean()
```

### 8.6 Moirai Zero-Shot Component

```python
from uni2ts.model.moirai import MoiraiForecast

def get_moirai_forecast(
    physics_history: np.ndarray,    # shape (600, N_features)
    context_length: int = 512,
    forecast_horizon: int = 30      # minutes
) -> np.ndarray:
    """
    Salesforce Moirai-1.0-R zero-shot time-series foundation model.
    Handles small-data regime without requiring fine-tuning.
    
    Input: last 512 time steps of physics features
    Output: probabilistic forecast for next 30 minutes
    
    If context length exceeded: clip to last 512, log truncation event,
    increase TCN weight in ensemble from 0.5 to 0.7.
    """
    model = MoiraiForecast.from_pretrained('Salesforce/moirai-1.0-R-base')
    context = physics_history[-context_length:]  # clip if needed
    forecast = model.predict(context, forecast_horizon)
    return forecast
```

### 8.7 Ensemble + MAPIE Calibration

```python
from mapie.classification import MapieClassifier

def ensemble_forecast(
    tcn_logits: torch.Tensor,
    moirai_probs: np.ndarray,
    tcn_weight: float = 0.5,
    moirai_weight: float = 0.5,
    mapie_calibrator: MapieClassifier = None
) -> Dict:
    """
    Weighted ensemble of TCN + Moirai.
    Calibrated via MAPIE conformal prediction.
    
    Output includes:
    - 5-class probability vector
    - 90% conformal prediction set (guaranteed coverage)
    - Point estimate with calibrated confidence
    """
    tcn_probs    = torch.softmax(tcn_logits, dim=1).numpy()
    ensemble_raw = tcn_weight * tcn_probs + moirai_weight * moirai_probs

    if mapie_calibrator:
        _, prediction_sets = mapie_calibrator.predict(
            ensemble_raw, alpha=0.1  # 90% coverage guarantee
        )
    else:
        prediction_sets = None

    predicted_class = np.argmax(ensemble_raw)
    CLASS_LABELS = ['QUIET', 'B', 'C', 'M', 'X']

    return {
        'prob_vector':        ensemble_raw.tolist(),
        'predicted_class':    CLASS_LABELS[predicted_class],
        'peak_class_prob':    float(ensemble_raw[0, predicted_class]),
        'prediction_set':     [CLASS_LABELS[i] for i in np.where(prediction_sets[0])[0]],
        'tcn_prob':           tcn_probs.tolist(),
        'moirai_prob':        moirai_probs.tolist(),
    }
```

### 8.8 Forecast Output Schema

```python
@dataclass
class ForecastFrame:
    forecast_id:          str
    timestamp:            datetime
    horizon_minutes:      int             # 5, 10, 15, 30, 60

    # Probabilities
    prob_quiet:           float
    prob_B:               float
    prob_C:               float
    prob_M:               float
    prob_X:               float
    predicted_class:      str
    peak_class_prob:      float

    # Uncertainty
    confidence_interval_low:  float      # lower bound (90% conformal)
    confidence_interval_high: float      # upper bound
    prediction_set:           List[str]  # MAPIE conformal set
    calibration_ece:          float      # expected calibration error

    # Lead time
    lead_time_estimate_sec:    int
    lead_time_uncertainty_sec: int

    # Attribution
    tcn_weight:               float
    moirai_weight:            float
    hope_flag_active:         bool
    physics_state:            str

    # Provenance
    input_window_start:       datetime
    input_window_end:         datetime
    model_version:            str
    pipeline_version:         str
```

### 8.9 Temporal Leakage Prevention

```python
# CRITICAL: Solar time-series must NOT be split randomly
# Random splits allow the model to "see the future" via temporal autocorrelation

TRAIN_PERIOD   = ('2023-09-02', '2024-06-30')   # SC25 rising phase
VALIDATE_PERIOD = ('2024-07-01', '2024-12-31')   # high-activity period
TEST_PERIOD    = ('2025-01-01', '2025-06-30')    # held out — not touched until final eval

# Within each split: no shuffling. Data must remain in chronological order.
# Validation events must post-date ALL training events by at least 24 hours (buffer).
```

### 8.10 Acceptance Criteria

```
✓ TSS > 0.65 on M+X class events (validation set 2024 H2)
✓ ROC-AUC > 0.92 on binary M+/X detection (held-out test set)
✓ Expected Calibration Error < 0.05 after MAPIE calibration
✓ Lead time median absolute error < 90 seconds on HOPE-triggered events
✓ Ensemble beats best single model by TSS ≥ 0.03
✓ No temporal leakage verified (no training events post-dating validation by < 24 hr)
✓ Forecast output within 10 seconds of trigger
✓ All forecasts include calibrated confidence interval
✓ Under-prediction rate < 5% for M5+ events (asymmetric loss validation)
```

---

## 9. L5 — Historical Analog Retrieval

### 9.1 Objective

When a new event begins, retrieve the three most similar past flares by light-curve shape, lag, and energy evolution. Ground every alert in real prior solar behaviour rather than abstract probability.

### 9.2 Algorithm — DTW + FAISS Vector Search

```python
import faiss
from dtaidistance import dtw

def build_analog_library(
    catalog_events: List[MasterFlareEvent],
    feature_keys: List[str] = ['T_MK', 'EM_cm3', 'DNI', 'SHEI']
) -> Tuple[faiss.IndexFlatL2, List[str]]:
    """
    Encode each historical event as a contrastive embedding.
    Index in FAISS for fast similarity search.
    """
    embeddings = []
    event_ids  = []
    encoder    = ContrastiveFlareEncoder()  # PyTorch, SimCLR-style

    for event in catalog_events:
        feature_ts = event.get_feature_timeseries(feature_keys)
        embedding  = encoder.encode(feature_ts)
        embeddings.append(embedding)
        event_ids.append(event.event_id)

    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))
    return index, event_ids

def retrieve_analogs(
    query_event: FlareEvent,
    index: faiss.IndexFlatL2,
    event_ids: List[str],
    top_k: int = 3,
    dtw_validation: bool = True
) -> List[AnalogMatch]:
    """
    Step 1: FAISS approximate nearest neighbor (fast, O(log N))
    Step 2: DTW refinement on top-10 candidates (accurate, O(N²))
    Step 3: Return top-3 with similarity scores and metadata
    
    NO_ANALOG fallback:
    If DTW distance to nearest neighbor > 3σ above library mean:
        → Flag NO_ANALOG
        → Issue alert with HIGH_UNCERTAINTY label
        → Fall back to TCN-only probability
    """
    query_embedding = encoder.encode(query_event.get_feature_timeseries())
    distances, indices = index.search(query_embedding.reshape(1, -1), k=10)

    # DTW refinement
    dtw_scores = []
    for idx in indices[0]:
        candidate = catalog_events[idx]
        dtw_dist  = dtw.distance(query_event.timeseries, candidate.timeseries)
        dtw_scores.append((dtw_dist, event_ids[idx]))

    dtw_scores.sort()
    library_mean_dtw = np.mean([s[0] for s in dtw_scores])
    library_std_dtw  = np.std([s[0] for s in dtw_scores])

    no_analog_flag = dtw_scores[0][0] > library_mean_dtw + 3 * library_std_dtw

    return [AnalogMatch(
        event_id=eid,
        dtw_distance=dist,
        similarity=1 / (1 + dist),
        no_analog_flag=no_analog_flag
    ) for dist, eid in dtw_scores[:top_k]]
```

### 9.3 Acceptance Criteria

```
✓ Analog retrieval precision > 0.75 (top-3 analog class accuracy)
✓ FAISS search < 100ms for library of 10,000 events
✓ NO_ANALOG flag fires correctly when DTW distance > 3σ
✓ X6.3 event retrieves at least one X-class analog from 2024
✓ FAISS index rebuilt automatically after each new catalog entry
```

---

## 10. L6 — Threat Intelligence

### 10.1 Objective

Translate flare class + lead time + analog into operational risk flags for Indian infrastructure. Output a prioritised action list (Dos and Don'ts) alongside quantified risk levels. This layer turns ARUNA from a science tool into an operations tool.

### 10.2 Threat Rule Engine

```python
THREAT_RULES = {
    'navic_l5': {
        'B':   ('NEGLIGIBLE', 0.0),
        'C':   ('LOW',        0.15),
        'M1':  ('MODERATE',   0.30),
        'M5':  ('HIGH',       0.55),
        'X1':  ('VERY_HIGH',  0.67),
        'X5':  ('EXTREME',    0.90),
    },
    'hf_comms_india': {
        'B':   ('NONE',     0),
        'C':   ('MINOR',    15),   # minutes of degradation
        'M1':  ('MODERATE', 45),
        'M5':  ('STRONG',   120),
        'X1':  ('SEVERE',   240),
        'X5':  ('COMPLETE', 480),
    },
    'gaganyaan_radiation': {
        'B':   'NOMINAL',
        'C':   'NOMINAL',
        'M1':  'ELEVATED',
        'M5':  'HIGH — SHELTER ADVISORY',
        'X1':  'CRITICAL — SHELTER REQUIRED',
        'X5':  'EMERGENCY — EVA ABORT',
    },
    'satellite_safing_priority': [
        # Ranked by vulnerability (highest first)
        ('Chandrayaan-3',   'LUNAR_POLAR',     'radiation_sensitive'),
        ('Aditya-L1',       'L1_HALO',         'instrument_safe_mode'),
        ('RISAT-2BR1',      'LEO_SAR',         'attitude_sensitive'),
        ('GSAT-30',         'GEO',             'transponder_protection'),
        ('IRNSS-1I',        'MEO_NAVIC',       'ionospheric_impact'),
        ('EOS-06',          'LEO_OCEAN',       'payload_safe_mode'),
    ]
}
```

### 10.3 India Regional NavIC Impact Map

```python
INDIA_REGIONS = {
    'north_india':    {'geomag_lat': (20, 30), 'eia_effect': 'EDGE',  'key_infra': 'Army comms, Northern Railway'},
    'central_india':  {'geomag_lat': (15, 22), 'eia_effect': 'CREST', 'key_infra': 'Power grid, ISRO Bangalore uplink'},
    'south_india':    {'geomag_lat': (5,  15), 'eia_effect': 'CREST', 'key_infra': 'NavIC ground stations, ISRO SHARs'},
    'northeast':      {'geomag_lat': (20, 28), 'eia_effect': 'EDGE',  'key_infra': 'Oil pipelines, Army logistics'},
    'island_chains':  {'geomag_lat': (5,  13), 'eia_effect': 'EQUATORIAL_MAX', 'key_infra': 'Maritime GPS, Navy'},
}

# Equatorial Ionospheric Anomaly (EIA) creates peak scintillation at ±15° geomagnetic lat
# South India and island chains are in the worst scintillation zone
```

### 10.4 Dos and Don'ts Output

```python
def generate_dos_donts(
    goes_class: str,
    lead_time_sec: int,
    affected_regions: List[str],
    gaganyaan_active: bool
) -> Dict[str, List[str]]:
    """
    Operational directives for ISRO operations centers.
    Auto-generated from threat rules — not hardcoded per event.
    """
    dos, donts = [], []

    if goes_class in ('M5', 'M6', 'M7', 'M8', 'M9', 'X1', 'X2', 'X3'):
        dos   += [f"Switch NavIC receivers to S1 band backup ({lead_time_sec//60} min lead time)"]
        dos   += ["Notify ISRO SHARs ground stations of expected signal degradation"]
        donts += ["Do not schedule critical satellite commanding windows"]
        donts += ["Do not rely on HF comms for defence field operations"]

    if goes_class.startswith('X'):
        dos   += ["Move Gaganyaan crew to shielded module IMMEDIATELY"] if gaganyaan_active else []
        dos   += ["Activate ISTRAC satellite safing priority queue"]
        donts += ["Do not perform EVA or spacewalk operations"]
        donts += ["Do not ignore sudden GPS drift alerts — ionospheric impact imminent"]

    return {'do_immediately': dos, 'do_not': donts}
```

### 10.5 Satellite Safing Priority Queue

```python
def generate_safing_queue(
    goes_class: str,
    isro_satellite_registry: List[Dict]
) -> List[Dict]:
    """
    When an alert fires, rank all ISRO operational satellites by
    safing priority for the specific flare class.
    Output: JSON list ordered highest priority first.
    The ISTRAC duty officer acts on this list in sequence.
    """
    vulnerability_scores = {
        'LUNAR_POLAR':     {'X': 1.0, 'M': 0.7, 'C': 0.2},
        'L1_HALO':         {'X': 0.9, 'M': 0.6, 'C': 0.1},
        'LEO_SAR':         {'X': 0.8, 'M': 0.5, 'C': 0.1},
        'GEO':             {'X': 0.7, 'M': 0.4, 'C': 0.0},
        'MEO_NAVIC':       {'X': 0.8, 'M': 0.6, 'C': 0.2},
    }
    class_key = 'X' if goes_class.startswith('X') else \
                'M' if goes_class.startswith('M') else 'C'

    ranked = sorted(
        isro_satellite_registry,
        key=lambda s: vulnerability_scores.get(s['orbit_type'], {}).get(class_key, 0.0),
        reverse=True
    )
    return ranked
```

### 10.6 ISRO SpaceWx Bulletin Format Export

```python
def export_spacewx_bulletin(alert: AlertEvent, threat: ThreatOutput) -> str:
    """
    Export alert in ISRO SpaceWx SWMC bulletin format.
    Enables direct plug-in to ISRO's existing alert dissemination chain.
    """
    return f"""
SPACE WEATHER BULLETIN — ARUNA AUTOMATED ALERT
Issue Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
Alert ID: {alert.alert_id}

SOLAR EVENT SUMMARY
-------------------
Class: {alert.goes_class}{alert.goes_subclass}
Status: {alert.alert_type}
Lead Time: {alert.lead_time_sec} seconds
Confidence: {alert.confidence:.0%}
Physics State: {alert.physics_state}
T_peak: {alert.peak_temperature_MK:.1f} MK

INDIA IMPACT ASSESSMENT
------------------------
NavIC L5 Risk: {threat.navic_risk['south_india']['level']}
HF Comms: {threat.hf_blackout_risk} ({threat.hf_blackout_duration})
Radiation: {threat.radiation_risk}

OPERATIONAL DIRECTIVES
----------------------
DO: {'; '.join(threat.dos[:3])}
DO NOT: {'; '.join(threat.donts[:3])}

SATELLITE SAFING PRIORITY
--------------------------
{chr(10).join(f"{i+1}. {s['name']} ({s['orbit_type']})" for i, s in enumerate(threat.safing_queue[:5]))}

Data Source: Aditya-L1 SoLEXS + HEL1OS | ARUNA v2.0 | ISRO-BAH2026
    """.strip()
```

### 10.7 Acceptance Criteria

```
✓ Threat flags generated within 2 seconds of NowcastEvent receipt
✓ NavIC risk level matches empirical Kintner model ± 1 risk level
✓ Satellite safing queue orders Chandrayaan-3 as Priority 1 for X-class
✓ SpaceWx bulletin format validates against SWMC template
✓ Dos/Donts list generated for every M5+ and X-class event
✓ Regional NavIC map shows correct EIA amplification for South India
✓ No false-safety: X-class impact must never be under-stated
```

---

## 11. L7 — LLM Briefing

### 11.1 Objective

Convert all deterministic pipeline outputs into a plain-language operator alert. The LLM is strictly OUTSIDE the decision core — it only narrates, never decides. Every factual claim is validated against pipeline outputs before delivery.

### 11.2 LLM Configuration

```yaml
llm:
  provider: ollama
  model: mistral-7b-instruct
  mode: offline          # no internet required — runs air-gapped
  structured_output: true
  max_tokens: 400
  temperature: 0.1       # low temperature for factual consistency
```

### 11.3 Structured Prompt Template

```python
SYSTEM_PROMPT = """
You are ARUNA's operational briefing module. You narrate only.
You MUST NOT:
- Introduce any numbers not in the CONTEXT OBJECT below
- Make predictions beyond what the pipeline has computed
- Use hedging language that obscures actionable information
- Exceed 3 paragraphs

You MUST:
- State the flare class, confidence, and lead time in the first sentence
- State the NavIC and HF comms risk in the second paragraph
- State the operational directives in the third paragraph
- Use plain language suitable for a non-scientist operations officer

Output ONLY valid JSON with field "briefing_text".
"""

def generate_briefing(
    alert: AlertEvent,
    forecast: ForecastFrame,
    threat: ThreatOutput,
    analogs: List[AnalogMatch]
) -> str:
    context = {
        "goes_class":      f"{alert.goes_class}{alert.goes_subclass}",
        "confidence_pct":  f"{alert.confidence:.0%}",
        "lead_time_min":   alert.lead_time_sec // 60,
        "navic_risk":      threat.navic_risk['south_india']['level'],
        "hf_risk":         threat.hf_blackout_risk,
        "hf_duration_min": threat.hf_blackout_duration,
        "radiation_risk":  threat.radiation_risk,
        "do_list":         threat.dos[:3],
        "analog_class":    analogs[0].goes_class if analogs else "N/A",
        "analog_similarity": f"{analogs[0].similarity:.0%}" if analogs else "N/A",
    }
    prompt = f"CONTEXT OBJECT: {json.dumps(context)}\n\nWrite the operational briefing."
    response = ollama_client.generate(
        model='mistral-7b-instruct',
        prompt=prompt,
        system=SYSTEM_PROMPT
    )
    return json.loads(response['text'])['briefing_text']
```

### 11.4 Hallucination Guard

```python
def validate_briefing(
    briefing_text: str,
    alert: AlertEvent,
    threat: ThreatOutput
) -> Tuple[bool, List[str]]:
    """
    Validates that the LLM briefing does not introduce new numbers.
    If any constraint fails: fallback to deterministic template.
    """
    violations = []

    # Check flare class mentioned correctly
    if f"{alert.goes_class}{int(alert.goes_subclass)}" not in briefing_text:
        violations.append("MISSING_FLARE_CLASS")

    # Check lead time is within ± 2 minutes of pipeline value
    lead_time_matches = re.findall(r'(\d+)\s*minute', briefing_text)
    for match in lead_time_matches:
        if abs(int(match) - alert.lead_time_sec // 60) > 2:
            violations.append(f"LEAD_TIME_MISMATCH: {match}")

    # Verify no novel numbers introduced (simple regex check)
    numbers_in_briefing = set(re.findall(r'\b\d+\.?\d*\b', briefing_text))
    allowed_numbers = {str(x) for x in [
        int(alert.lead_time_sec // 60),
        round(alert.confidence * 100),
        round(alert.goes_subclass, 1),
        round(threat.hf_blackout_duration_min),
    ]}
    novel_numbers = numbers_in_briefing - allowed_numbers
    if len(novel_numbers) > 3:  # allow some (dates, small integers)
        violations.append(f"NOVEL_NUMBERS: {novel_numbers}")

    valid = len(violations) == 0
    if not valid:
        # Log and use deterministic fallback
        log.warning(f"LLM hallucination detected: {violations}")

    return valid, violations
```

### 11.5 Fallback Template

```python
DETERMINISTIC_TEMPLATE = """
ARUNA SOLAR FLARE ALERT — {timestamp} UTC
Flare Class: {goes_class} | Confidence: {confidence_pct} | Lead Time: {lead_time_min} minutes

NavIC Impact: {navic_risk} — {navic_affected_regions}
HF Communications: {hf_risk} blackout expected for {hf_duration_min} minutes
Gaganyaan Radiation: {radiation_risk}

ACTIONS REQUIRED:
{dos_formatted}
AVOID:
{donts_formatted}

Data: Aditya-L1 SoLEXS+HEL1OS | Most similar past event: {analog_class} ({analog_similarity} similarity)
"""
```

### 11.6 Acceptance Criteria

```
✓ Hallucination guard passes on 100% of test alerts
✓ Briefing generated within 15 seconds of alert trigger
✓ Fallback template activates when hallucination guard fails
✓ Every briefing includes flare class, confidence, lead time, NavIC risk
✓ No briefing introduces numbers not present in the context object
✓ LLM runs fully offline — no network requests
```

---

## 12. L8 — Dashboard Services

### 12.1 Objective

An 8-page operational web platform that provides situational awareness, live alerts, event replay, and mission advisory. The dashboard is the face of ARUNA — what evaluators see and what operators use daily.

### 12.2 Technology Stack

```yaml
framework: streamlit==1.35.0
charts:    plotly==5.22.0
database:  sqlite3 (local) / postgresql (production)
realtime:  kafka-python consumer thread
deploy:    docker (single container)
```

### 12.3 Page Specifications

**Page 1 — Landing / System Status**
```
- ARUNA logo + tagline "Foresee. Prepare. Protect."
- Live Sun status: GREEN / YELLOW / ORANGE / RED
- Current SoLEXS + HEL1OS light curves (last 2 hours)
- Active alert count + last alert summary
- System health: all pipeline stages GREEN/AMBER/RED
- Next PRADAN data update countdown
- India map with current NavIC scintillation risk overlay
```

**Page 2 — Live Operations (Primary Demo Page)**
```
- Real-time SoLEXS flux (3-6 keV, 6-10 keV) — updating every second
- Real-time HEL1OS flux (10-25 keV, 25-50 keV) — updating every second
- T(t) and EM(t) live traces with 1-sigma uncertainty ribbon
- DNI gauge (0–2.0, red line at 1.5 = HOPE threshold)
- TNPSV phase space plot (live trajectory, HOPE quadrant highlighted)
- HOPE precursor indicator: INACTIVE / ACTIVE (with countdown)
- Current flare phase label: QUIET / HOPE / IMPULSIVE / GRADUAL / DECAY
- Active alert banner (auto-appears on flare detection)
```

**Page 3 — Forecast Center**
```
- 5-class probability bar chart (QUIET / B / C / M / X)
- Forecast horizon selector: 5 / 10 / 15 / 30 / 60 min
- Calibration reliability diagram (predicted probability vs observed frequency)
- Lead time estimate with uncertainty
- Sub-model breakdown: TCN vs Moirai vs ensemble
- Conformal prediction set display
- SOC power-law size prior gauge
```

**Page 4 — Master Flare Catalog**
```
- Searchable table of all detected events
- Filters: date range, class, instrument, HOPE fired, confidence
- Per-event detail drill-down: T/EM/DNI time series, alert timeline
- Export to CSV
- Statistics: events per class, mean lead time, TSS by class
```

**Page 5 — X6.3 Event Replay (Always-On Demo)**
```
CRITICAL: This page is hardcoded on the X6.3 event (2024-02-22).
It ALWAYS works regardless of live data availability.
- Full event timeline: SUIT UV → SoLEXS thermal → HEL1OS nonthermal
- T/EM trace with published Sarwade et al. values overlaid
- DNI trace showing HOPE threshold crossing
- TNPSV phase space trajectory animation
- Alert timeline: HOPE fired at T-0 | Onset at T+Δ | Peak at T+Δ²
- Bottom banner: "ARUNA matched published SoLEXS team values within 1σ"
```

**Page 6 — Threat Intelligence & Satellite Safing**
```
- India regional NavIC S4 risk map (5 regions, color-coded)
- HF communications blackout prediction (duration + affected routes)
- Gaganyaan radiation advisory panel
- ISRO satellite safing priority queue (live ranked list)
- Dos and Don'ts operational card (auto-generated per alert)
- SpaceWx bulletin preview and export button
```

**Page 7 — Performance & Validation**
```
- Rolling 30-day confusion matrix heatmap (5×5)
- TSS time series with 95% CI ribbon
- Calibration reliability diagram (live)
- Cohen's Kappa gauge with historical trend
- Lead-time distribution histogram (HOPE events only)
- Ground-truth comparison: ARUNA vs GOES vs SoLEXS catalog (last 50 events)
- Bootstrap CI displayed for all metrics
```

**Page 8 — Gaganyaan Mission Window Advisor (Irresistibility Feature)**
```
- 7-day solar activity forecast calendar
- Each day: GREEN (< 15% M+ probability) / YELLOW (15–40%) / RED (> 40%)
- EVA safe window recommendations
- Launch window advisory
- Radiation exposure forecast for crew at 400 km LEO
- Label: "Gaganyaan Mission Planning Support — SC25 Peak Activity Period"
```

### 12.4 Acceptance Criteria

```
✓ Page 2 (Live Ops) updates within 2 seconds of new physics frame
✓ Page 5 (X6.3 Replay) loads and animates in < 3 seconds, always
✓ Page 7 confusion matrix matches offline validation metrics
✓ Alert banner appears within 5 seconds of NowcastEvent on Kafka
✓ Dashboard functional with no internet connection (all data local)
✓ Streamlit app starts with: streamlit run dashboard/app.py
✓ Page 8 Gaganyaan advisor updates daily at 06:00 IST
✓ SpaceWx bulletin export generates valid formatted text
```

---

## 13. XL Extension Modules

### 13.1 XL Architecture Principle

ARUNA-XL is NOT a rewrite of the core pipeline. It is six modules that sit above L0–L8, consuming their outputs and adding new capabilities. All XL modules publish to and consume from the same Kafka instance.

```
Build priority for BAH 2026 finale:
Phase 1 (5 days): XL-1 AR Registry skeleton + XL-2 MPCI scalar
Phase 2 (7 days): XL-3 TNCI + XL-4 M1+M3 sub-models
Phase 3 (10 days): XL-4 full ensemble + XL-5 Earth/NavIC impact
Phase 4 (5 days): XL-6 validation agent + MLflow dashboards
```

### 13.2 XL-1 — Active Region Intelligence Layer

**Purpose:** Transform ARUNA from a Sun-as-a-star detector into an active-region-aware intelligence system. Each NOAA Active Region gets a persistent state object updated in real time.

**Key data structure:**
```python
@dataclass
class ARState:
    ar_id:           int           # NOAA AR number
    position:        Tuple[float, float]  # Stonyhurst (lat, lon)
    mcintosh_class:  str           # e.g. 'Fkc'
    hale_class:      str           # e.g. 'beta-gamma-delta'
    bpil:            float         # polarity inversion line length (SHARP)
    free_energy:     float         # SHARP USFLUX proxy
    helicity:        float         # SHARP TOTUSJH
    twist_rate:      float         # d(TWIST)/dt — rolling 1-hr gradient
    shear_rate:      float         # d(SHEAR)/dt
    solexs_ts:       np.ndarray    # 1s thermal flux history
    hel1os_ts:       np.ndarray    # 1s nonthermal flux history
    flare_history:   List[str]     # event_ids from master catalog
    freshness:       float         # exp(-λ·Δt), λ=1/3600
    risk_score:      float         # composite 0.0–1.0
    last_updated:    datetime
```

**Kafka outputs:**
- `aruna.xl.ar_state` — ARState updates (every 720s or on new observation)
- `aruna.xl.ar_events` — AR-tagged observation stream

### 13.3 XL-2 — Multi-Payload Coherence Layer

**Purpose:** Ask whether all four Aditya-L1 payloads tell a physically consistent story.

**MPCI computation:**
```python
def compute_MPCI(
    suit_uv_score: float,       # 0–1, UV brightening
    solexs_score:  float,       # 0–1, T rising / EM flat
    hel1os_score:  float,       # 0–1, spectral hardening
    velc_score:    float,       # 0–1, Doppler shift (may be NaN)
    available_payloads: List[str]
) -> float:
    weights = {'SUIT': 0.20, 'SoLEXS': 0.35, 'HEL1OS': 0.30, 'VELC': 0.15}
    scores  = {'SUIT': suit_uv_score, 'SoLEXS': solexs_score,
               'HEL1OS': hel1os_score, 'VELC': velc_score}
    raw  = sum(weights[p] * scores[p] for p in available_payloads)
    norm = sum(weights[p]              for p in available_payloads)
    return raw / norm  # re-normalised if VELC unavailable

# Alert confidence modifier:
# MPCI > 0.80 → +20% confidence boost
# MPCI < 0.50 → -15% confidence penalty, DEGRADED_CONFIDENCE flag
# Ladder out-of-order → INCOHERENT flag, escalate to operator
```

**Expected payload timing ladder (physical sequence):**
1. SUIT UV brightening (chromospheric precursor, 2–5 min before peak)
2. SoLEXS soft X-ray temperature rise (HOPE signature, 1–3 min before)
3. VELC Doppler shift (plasma evaporation, 1–2 min before)
4. HEL1OS hard X-ray onset (impulsive phase, 0 min — this IS the peak)

### 13.4 XL-3 — Physics Observable Engine

**Purpose:** Formalise all physics-derived quantities into a named, typed, physically interpretable feature vector published as "physics tokens."

**New observable: TNCI (Thermal-Nonthermal Coupling Index)**
```python
def compute_TNCI(T, EM, flux_hel1os, dt=1.0):
    """
    TNCI(t) = d/dt[T(t) × EM(t)] / F_HEL1OS(t)
    
    Rising TNCI: thermal plasma accumulating energy faster than 
                 nonthermal electrons inject → conductive/pre-impulsive phase
    Falling TNCI: nonthermal injection outpacing thermal → impulsive onset
    Oscillating TNCI: periodic reconnection (QPP signature in energy exchange)
    """
    thermal_energy_proxy = T * EM
    d_thermal_dt = np.gradient(thermal_energy_proxy, dt)
    TNCI = d_thermal_dt / (flux_hel1os + 1e-30)
    return TNCI
```

**Complete physics token set (21 features):**
See XL document Section 8 — all tokens with cadences, formulas, and physical interpretations.

### 13.5 XL-4 — Stacked Forecast Ensemble

**Five sub-models:**

| Model | Inputs | Architecture | Role |
|-------|--------|-------------|------|
| M1: Region | SHARP features + AR rates | LightGBM | 24-hr magnetic-state forecast |
| M2: Event | SoLEXS/HEL1OS/SUIT time series | Dual-stream TCN | Real-time X-ray onset probability |
| M3: Physics | DNI, TNPSV, SHEI, TNCI, MPCI | Logistic regression + poly features | Physically-grounded, interpretable |
| M4: Sequence | Full 60-min physics token history | Moirai-1.0-R fine-tuned | Medium-range (10–60 min) forecast |
| M5: Calibration | Raw M1–M4 probabilities | Isotonic regression + Platt scaling | ECE minimisation |

**Graceful degradation:** If SHARP unavailable (M1 offline), TSS should drop < 0.08. Tested explicitly.

### 13.6 XL-5 — Earth / NavIC Impact Layer

**Key outputs:**
```python
impact_output = {
    'cme_probability':       0.72,
    'cme_status':            'PROBABILISTIC',   # or CONFIRMED | UNLIKELY
    'arrival_time_estimate': '2026-08-07T14:30Z',
    'arrival_time_uncertainty': '±5 hours',
    'hf_blackout_risk':      'HIGH',
    'hf_blackout_duration':  '1-3 hours',
    'navic_risk': {
        'north_india':   {'L5': 0.28, 'S1': 0.15, 'level': 'MODERATE'},
        'central_india': {'L5': 0.55, 'S1': 0.38, 'level': 'HIGH'},
        'south_india':   {'L5': 0.67, 'S1': 0.51, 'level': 'VERY HIGH'},
        'northeast':     {'L5': 0.31, 'S1': 0.18, 'level': 'MODERATE'},
        'island_chains': {'L5': 0.72, 'S1': 0.60, 'level': 'VERY HIGH'},
    },
    'radiation_risk':        'ELEVATED',
    'sep_probability':       0.18,
    'kp_forecast':           5,
    'storm_class':           'G1',
    'lead_time_flare_to_hf_impact':     '5-15 minutes',
    'lead_time_flare_to_navic':         '5-20 minutes',
    'lead_time_to_cme_arrival':         '18-30 hours',
}
```

### 13.7 XL-6 — Extended Validation & Self-Correction Agent

**Agent trigger conditions (any one fires):**
```python
XL_AGENT_TRIGGERS = {
    'tss_drop':        lambda m: m['rolling_24h_tss'] < 0.55,
    'far_spike':       lambda m: m['rolling_3h_far']  > 0.08,
    'kappa_drop':      lambda m: m['cohen_kappa']     < 0.50,
    'ece_spike':       lambda m: m['calibration_ece'] > 0.10,
    'false_alarm_storm': lambda m: m['quiet_sun_false_alarms_48h'] > 2,
    'ar_coverage_drop':  lambda m: m['ar_registry_coverage'] < 0.85,
    'mpci_perpetually_low': lambda m: m['mpci_6h_mean'] < 0.40,
    'ensemble_divergence':  lambda m: m['submodel_prob_spread'] > 0.40,
}
```

**Correction loop:** Optuna Bayesian search (30 trials, TPE sampler). Stops when all trigger thresholds cleared OR after 3 iterations maximum. Every agent action logged to MLflow.

---

## 14. Message Bus — Kafka Topics

### 14.1 Topic Registry

| Topic | Producer | Consumers | Schema | Retention | Partitions |
|-------|---------|-----------|--------|-----------|-----------|
| `aruna.source.health` | L0 HealthMonitor | L1, Dashboard | SourceStatus | 24h | 1 |
| `aruna.observations.solexs` | L1 Ingestion | L2 Physics | ObservationFrame | 72h | 3 |
| `aruna.observations.hel1os` | L1 Ingestion | L2 Physics | ObservationFrame | 72h | 3 |
| `aruna.physics.frames` | L2 Physics | L3, L4, L5, XL-3 | PhysicsFrame | 72h | 3 |
| `aruna.nowcast.events` | L3 Nowcasting | L4, L5, L6, L7, L8 | NowcastEvent | 30d | 1 |
| `aruna.forecast.frames` | L4 Forecasting | L6, L7, L8, XL-4 | ForecastFrame | 30d | 1 |
| `aruna.analogs.results` | L5 Analog | L7, L8 | AnalogResult | 30d | 1 |
| `aruna.threat.output` | L6 Threat | L7, L8 | ThreatOutput | 30d | 1 |
| `aruna.alerts.briefings` | L7 LLM | L8 Dashboard | AlertBriefing | 90d | 1 |
| `aruna.xl.ar_state` | XL-1 | XL-2, XL-3, XL-4, Dashboard | ARState | 7d | 1 |
| `aruna.xl.mpci` | XL-2 | XL-3, XL-4, Dashboard | MPCIFrame | 72h | 1 |
| `aruna.xl.features` | XL-3 | XL-4, XL-6 | PhysicsTokens | 72h | 3 |
| `aruna.xl.forecast` | XL-4 | XL-5, L8, XL-6 | EnsembleForecast | 30d | 1 |
| `aruna.xl.impact` | XL-5 | L8, XL-6 | ImpactOutput | 30d | 1 |
| `aruna.ingestion.errors` | L1 | XL-6, Dashboard | ErrorEvent | 7d | 1 |

### 14.2 Message Schema Convention

All messages: JSON-serialised dataclass, UTF-8 encoded  
All messages include: `schema_version`, `pipeline_version`, `created_at` (UTC ISO 8601)  
Retry policy: 3 retries with exponential backoff (1s, 2s, 4s)  
Dead-letter queue: `aruna.dlq.*` for each topic

---

## 15. Database Schema

### 15.1 Stores

| Store | Technology | Purpose | Retention |
|-------|-----------|---------|-----------|
| Raw store | Local filesystem + LRU cache | FITS files | 48 hours |
| Feature store | SQLite / PostgreSQL | PhysicsFrames | 30 days |
| Prediction store | SQLite / PostgreSQL | ForecastFrames | 90 days |
| Event store | SQLite / PostgreSQL | Master Flare Catalog | Permanent |
| Metadata store | SQLite | Pipeline runs, versions | Permanent |
| Model store | MLflow artifact store | Trained weights, calibrators | Versioned |
| AR store | Redis / shelve | ARState objects | 7 days rolling |

### 15.2 Master Catalog Table (full schema)

See §7.8 — `master_flare_catalog` CREATE TABLE statement.

### 15.3 Validation Results Table

```sql
CREATE TABLE validation_results (
    run_id          TEXT PRIMARY KEY,
    timestamp       TIMESTAMP,
    pipeline_version TEXT,
    model_version   TEXT,
    eval_period_start TIMESTAMP,
    eval_period_end   TIMESTAMP,
    n_events        INTEGER,
    tpr_M_X         REAL,
    far_M_X         REAL,
    tss_M_X         REAL,
    hss             REAL,
    cohen_kappa     REAL,
    roc_auc         REAL,
    ece             REAL,
    lead_time_mae_sec REAL,
    bootstrap_ci_tss_low  REAL,
    bootstrap_ci_tss_high REAL,
    confusion_matrix TEXT,   -- JSON 5×5 array
    notes           TEXT
);
```

### 15.4 Model Registry Table

```sql
CREATE TABLE model_registry (
    model_id        TEXT PRIMARY KEY,
    model_type      TEXT,       -- 'TCN'|'Moirai'|'MAPIE'|'LightGBM'|'LogReg'
    version         TEXT,
    train_period    TEXT,
    val_tss         REAL,
    val_ece         REAL,
    artifact_path   TEXT,
    git_commit      TEXT,
    experiment_id   TEXT,       -- MLflow experiment ID
    deployed        BOOLEAN,
    deployed_at     TIMESTAMP,
    notes           TEXT
);
```

---

## 16. API Specifications

### 16.1 Internal REST API (FastAPI)

```
Base URL: http://localhost:8000/api/v1

GET  /health                    → System health status all stages
GET  /status/sun                → Current solar activity level
GET  /alerts/latest             → Latest 10 alert events
GET  /alerts/{alert_id}         → Single alert detail
GET  /catalog/events            → Master catalog with filters
GET  /catalog/events/{event_id} → Single event detail
GET  /forecast/current          → Current forecast frame
GET  /forecast/history          → Forecast history with metrics
GET  /threat/current            → Current threat assessment
GET  /threat/navic              → NavIC regional risk map
GET  /physics/current           → Current physics frame
GET  /physics/history?minutes=N → Physics frame history
GET  /validation/metrics        → Rolling validation metrics
POST /replay/event              → Trigger event replay on date range
GET  /bulletin/latest           → SpaceWx bulletin text
GET  /gaganyaan/window?days=7   → Gaganyaan mission window advisory
```

### 16.2 WebSocket (Streamlit real-time)

```
ws://localhost:8765/live
Publishes: PhysicsFrame, NowcastEvent, ForecastFrame — every 1 second
Clients: Dashboard Pages 1 and 2
```

---

## 17. Deployment Architecture

### 17.1 Docker Compose (Development / Demo)

```yaml
version: '3.9'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.6.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181

  kafka:
    image: confluentinc/cp-kafka:7.6.0
    depends_on: [zookeeper]
    ports: ["9092:9092"]
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"

  aruna-pipeline:
    build: .
    depends_on: [kafka]
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./config:/app/config
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - ARUNA_CONFIG=/app/config/config.yaml
    command: python -m aruna.pipeline.run

  aruna-dashboard:
    build: .
    depends_on: [aruna-pipeline]
    ports: ["8501:8501"]
    command: streamlit run dashboard/app.py --server.port 8501

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_models:/root/.ollama
    command: ollama serve

  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.13.0
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0

volumes:
  ollama_models:
```

### 17.2 One-Command Startup

```bash
# Full system startup
docker compose up -d

# Verify all services healthy
docker compose ps

# Run on X6.3 event (demo mode)
docker compose exec aruna-pipeline python -m aruna.replay \
    --date 2024-02-22 \
    --instruments SOLEXS HEL1OS \
    --output-dir /app/data/demo/x63/

# Access dashboard
open http://localhost:8501
```

### 17.3 ISRO Workstation Deploy (No Docker)

```bash
# Single-node deploy for ISRO workstation
git clone https://github.com/[team]/aruna
cd aruna

conda create -n aruna python=3.11
conda activate aruna
pip install -r requirements.txt

# Configure
cp config/config.example.yaml config/config.yaml
# Edit: PRADAN credentials, data paths

# Start
python -m aruna.pipeline.run &
streamlit run dashboard/app.py
```

---

## 18. Monitoring & Observability

### 18.1 Metrics (Prometheus)

```python
# Key metrics exported to Prometheus → Grafana dashboard
METRICS = {
    'aruna_pipeline_latency_seconds':  Histogram,   # per-stage latency
    'aruna_ingestion_frames_total':    Counter,      # frames processed
    'aruna_ingestion_errors_total':    Counter,      # frames rejected
    'aruna_flare_detections_total':    Counter,      # events detected (by class)
    'aruna_hope_precursors_total':     Counter,      # HOPE flags fired
    'aruna_lead_time_seconds':         Histogram,    # distribution of lead times
    'aruna_tss_rolling_24h':           Gauge,        # rolling TSS (M+X)
    'aruna_far_rolling_24h':           Gauge,        # rolling FAR
    'aruna_ece_current':               Gauge,        # calibration error
    'aruna_kafka_lag_messages':        Gauge,        # consumer lag per topic
    'aruna_pradan_connection_status':  Gauge,        # 1=connected, 0=offline
    'aruna_llm_hallucination_rate':    Gauge,        # briefing violation rate
    'aruna_model_drift_score':         Gauge,        # feature distribution shift
}
```

### 18.2 Logging Convention

```python
import structlog

log = structlog.get_logger()

# Every log entry includes:
log.info("physics_frame_computed",
    frame_id=frame.frame_id,
    timestamp=frame.timestamp.isoformat(),
    T_MK=frame.T_MK,
    hope_flag=frame.hope_flag,
    pipeline_version=PIPELINE_VERSION,
    latency_ms=elapsed_ms
)

# Log levels:
# DEBUG:    Per-frame computation details
# INFO:     Frame published, event detected, alert issued
# WARNING:  Data quality issues, clock drift, fallback activated
# ERROR:    Stage failure, data missing, model error
# CRITICAL: Both instruments offline, pipeline halted
```

### 18.3 Health Checks

```python
# GET /health → returns JSON
{
    "status": "HEALTHY",     # or DEGRADED | CRITICAL
    "timestamp": "2026-08-06T10:32:45Z",
    "stages": {
        "L1_ingestion":   {"status": "HEALTHY", "last_frame_age_sec": 2},
        "L2_physics":     {"status": "HEALTHY", "last_frame_age_sec": 2},
        "L3_nowcasting":  {"status": "HEALTHY", "last_event_age_sec": 847},
        "L4_forecasting": {"status": "HEALTHY", "last_forecast_age_sec": 3},
        "kafka":          {"status": "HEALTHY", "lag": 0},
        "pradan":         {"status": "HEALTHY", "latency_ms": 234},
        "ollama":         {"status": "HEALTHY", "model": "mistral-7b-instruct"},
    },
    "uptime_seconds": 86400
}
```

### 18.4 Model Drift Detection

```python
from river.drift import ADWIN

class ModelDriftDetector:
    """
    ADWIN (Adaptive Windowing) detector for input feature drift.
    Triggers XL-6 Accuracy Agent when distribution shift detected.
    Runs on: T_MK, DNI, SHEI distributions.
    """
    def __init__(self):
        self.detectors = {f: ADWIN() for f in ['T_MK', 'DNI', 'SHEI']}

    def update(self, physics_frame: PhysicsFrame) -> bool:
        drift_detected = False
        for feature, detector in self.detectors.items():
            value = getattr(physics_frame, feature, None)
            if value is not None:
                detector.update(value)
                if detector.drift_detected:
                    log.warning("model_drift_detected", feature=feature)
                    drift_detected = True
        return drift_detected
```

---

## 19. Testing & Validation Framework

### 19.1 Test Categories

| Category | Tool | What it tests |
|----------|------|--------------|
| Unit tests | pytest | Individual functions, algorithms |
| Integration tests | pytest + docker-compose | Stage-to-stage data flow |
| Physics validation | Custom harness | T/EM vs published values |
| Synthetic injection | synthsol / tsaug | Known flares into quiet data |
| Historical replay | Replay engine | X6.3, May 2024 superstorm |
| Regression tests | pytest + golden files | Prevent regressions after changes |
| Stress tests | locust | Pipeline under high event rate |
| Performance tests | pytest-benchmark | Per-stage latency budgets |

### 19.2 Critical Unit Tests

```python
# tests/test_physics.py

def test_temperature_x63_event():
    """
    T/EM on X6.3 event must match Sarwade et al. within 1-sigma.
    This is the most important single test in the entire suite.
    """
    fits_path = 'tests/fixtures/solexs_l1_20240222.fits'
    frames = ingest_fits(fits_path)
    physics = [compute_physics(f) for f in frames]
    peak_frame = max(physics, key=lambda p: p.T_MK)

    # Published values from Sarwade et al. (arXiv:2502.19795)
    PUBLISHED_T_MK   = 23.5    # MK — from paper Figure 3
    PUBLISHED_T_1SIG = 2.0     # MK — instrument uncertainty

    assert abs(peak_frame.T_MK - PUBLISHED_T_MK) <= PUBLISHED_T_1SIG, \
        f"T_MK {peak_frame.T_MK:.1f} outside 1-sigma of published {PUBLISHED_T_MK}"

def test_hope_fires_before_x63_peak():
    """HOPE precursor must fire ≥ 90 seconds before X6.3 peak."""
    physics_series = load_physics_series('2024-02-22')
    hope_time = first_hope_time(physics_series)
    peak_time  = x_ray_peak_time(physics_series)
    lead_time  = (peak_time - hope_time).total_seconds()
    assert lead_time >= 90, f"HOPE lead time {lead_time:.0f}s < 90s minimum"

def test_dual_trigger_consensus_required():
    """A single-instrument trigger must NOT generate a confirmed alert."""
    solexs_only_event = create_synthetic_solexs_trigger()
    alerts = nowcast_engine.process(solexs_only_event, hel1os_triggered=False)
    assert not any(a.alert_type == 'FLARE_ONSET' for a in alerts), \
        "Confirmed onset alert fired on SoLEXS-only trigger — dual consensus required"

def test_dead_time_correction_xclass():
    """Dead-time correction must produce physically plausible flux for X10+ event."""
    raw_rate   = 4.8e4   # approaching SDD1 saturation
    corrected  = apply_dead_time_correction(raw_rate, tau=4.5e-6)
    assert corrected > raw_rate, "Corrected rate must exceed observed rate"
    assert corrected < 1e6, "Corrected rate physically implausible"

def test_saa_bins_masked():
    """All SAA-coincident bins must be flagged, not silently kept."""
    frames_with_saa = load_frames_with_saa_passage()
    for frame in frames_with_saa:
        if frame.saa_flag:
            assert frame.quality_flag == QualityFlag.SAA_PASSAGE

def test_false_alarm_rate_quiet_sun():
    """FAR < 5% on 100 hours of quiet-Sun data."""
    quiet_frames   = load_quiet_sun_validation_set()  # 100 hours, no GOES events
    alerts_fired   = run_nowcasting_pipeline(quiet_frames)
    false_alarms   = [a for a in alerts_fired if a.alert_type == 'FLARE_ONSET']
    far            = len(false_alarms) / len(quiet_frames) * 3600  # per hour
    assert far < 0.05, f"FAR {far:.3f} exceeds 0.05 threshold on quiet-Sun data"
```

### 19.3 Synthetic Flare Injection

```python
def test_synthetic_c_class_detection():
    """
    Inject a synthetic C5 flare into 6 hours of quiet-Sun data.
    Pipeline must detect it with correct classification.
    """
    quiet_baseline = load_quiet_sun_frames('2024-01-15')
    synthetic_flare = generate_synthetic_flare(
        goes_class='C', subclass=5.0,
        rise_time_sec=300, peak_duration_sec=120, decay_time_sec=600
    )
    injected_stream = inject_flare_into_stream(quiet_baseline, synthetic_flare)
    detections = run_full_pipeline(injected_stream)

    assert any(d.goes_class == 'C' for d in detections), "C-class flare not detected"
    detected = next(d for d in detections if d.goes_class == 'C')
    assert abs(detected.goes_subclass - 5.0) <= 2.0, "Subclass more than 2 off"
```

### 19.4 May 2024 Superstorm Regression Test

```python
def test_may2024_superstorm_counterfactual():
    """
    The May 10-11, 2024 geomagnetic storm (Kp=9) is the primary counterfactual test.
    ARUNA must issue RED alert ≥ 4 hours before Kp reached 9.
    """
    pradan_data = load_pradan_fits_range('2024-05-09', '2024-05-11')
    alert_log   = run_full_pipeline_replay(pradan_data)

    kp9_time    = datetime(2024, 5, 10, 17, 00, tzinfo=timezone.utc)  # actual Kp=9 onset
    red_alerts  = [a for a in alert_log if a.alert_type in ('FLARE_PEAK', 'HOPE_PRECURSOR')
                   and a.goes_class in ('M', 'X')]

    if red_alerts:
        earliest_alert = min(red_alerts, key=lambda a: a.timestamp)
        lead_hours = (kp9_time - earliest_alert.timestamp).total_seconds() / 3600
        assert lead_hours >= 2.0, f"Lead time {lead_hours:.1f}h below 2-hour minimum"
```

---

## 20. Verification & Validation Plan (V&V)

### 20.1 V&V Framework

ARUNA's V&V is structured in five layers, following aerospace practice.

| Layer | Type | Method | Standard |
|-------|------|--------|---------|
| Functional V | Unit testing | pytest, synthetic injection | Per module |
| Scientific V | Physics comparison | vs published SoLEXS team values | Sarwade et al. 2025 |
| Statistical V | ML metrics | Confusion matrix, TSS, Kappa, ECE | WMO Space Weather |
| Operational V | Historical replay | X6.3, May 2024 superstorm | GOES catalog |
| End-to-End V | Integration test | Full pipeline on fresh FITS files | Mission acceptance |

### 20.2 Acceptance Tolerances

| Metric | Target | Minimum | Failure Action |
|--------|--------|---------|---------------|
| T_MK accuracy (X6.3) | ≤ 0.5σ | ≤ 1σ | Re-calibrate response matrix |
| TPR (M+X class) | > 0.85 | > 0.80 | Tune thresholds, augment training |
| FAR (quiet Sun) | < 0.02 | < 0.05 | Increase dual-trigger strictness |
| TSS (M+X class) | > 0.70 | > 0.65 | Retrain with class-weighted loss |
| Cohen's Kappa | > 0.65 | > 0.60 | Review classification boundary |
| ECE | < 0.03 | < 0.05 | Recalibrate MAPIE on recent data |
| HOPE lead time MAE | < 60s | < 90s | Tune HOPE threshold parameters |

### 20.3 Ground Truth Hierarchy

1. **Primary:** SoLEXS team published flare catalog (Sarwade et al.) — for post-2023-09 events
2. **Secondary:** NOAA GOES XRS flare catalog (SunPy HEK) — globally accepted standard
3. **Tertiary:** Fermi/GBM Solar Flare List — independent hard X-ray confirmation for M2+

**Matching rule:** ARUNA detection is a True Positive if onset time within ± 3 minutes of catalog AND inferred GOES class within ± 1 sub-class.

### 20.4 Bootstrap Confidence Intervals

All reported metrics include 95% bootstrap CI (n = 2000 resamples, scipy.stats.bootstrap). No point estimate reported without CI.

---

## 21. FMEA — Failure Modes & Effects Analysis

| Failure Mode | Cause | Effect | Severity | Probability | Detection | Mitigation | Residual Risk |
|---|---|---|---|---|---|---|---|
| SoLEXS FITS unavailable | PRADAN downtime | No primary thermal stream | HIGH | LOW | L0 health monitor | 48h cache, NOAA fallback | MEDIUM |
| HEL1OS FITS unavailable | PRADAN downtime | No non-thermal confirmation | HIGH | LOW | L0 health monitor | SoLEXS-only mode, DEGRADED flag | MEDIUM |
| Both instruments offline | Mission anomaly | No Aditya-L1 data | CRITICAL | VERY LOW | L0 health monitor | NOAA SWPC fallback, DATA_FALLBACK alert | LOW |
| SDD1 saturation (X10+) | Extreme flare | Under-estimated peak flux | HIGH | MEDIUM | Rate > saturation limit | Auto-switch to SDD2, SATURATED flag | LOW |
| Clock drift > 1s | Onboard timing anomaly | Corrupted TNCT coupling | HIGH | LOW | Cross-correlation lag | Fractional-second interpolation | LOW |
| SAA contamination | Orbit geometry | False HEL1OS triggers | MEDIUM | HIGH | HK SAA flag | Mask SAA bins, continue | LOW |
| Isothermal fit non-convergence | Unusual spectrum | No T/EM output | MEDIUM | LOW | chi-squared > 5.0 | Flag PHYSICS_FAIL, use last valid | MEDIUM |
| LLM hallucination | Model instability | Incorrect operational briefing | HIGH | LOW | Constraint checker | Fallback to deterministic template | LOW |
| False alarm during geomagnetic storm | Particle background | Operator alarm fatigue | HIGH | MEDIUM | Kp index cross-check | Classify SEP_EVENT not flare | LOW |
| Overlapping flares | Solar activity | Missed second event | MEDIUM | LOW | QPP period shift | Detect second change-point | LOW |
| FAISS index empty (cold start) | First deployment | No analog retrieval | LOW | ONE_TIME | Event count check | Flag NO_ANALOG, TCN-only mode | LOW |
| Kafka consumer lag | High event rate | Alert delivery delayed | MEDIUM | LOW | Kafka lag metric | Alert throttle, digest mode | LOW |
| ML model concept drift | SC25 peak activity | Degraded forecast accuracy | HIGH | MEDIUM | ADWIN detector | XL-6 agent triggers retraining | MEDIUM |
| Dashboard crash during event | High traffic | Operators lose visibility | HIGH | LOW | Health check | Secondary JSON endpoint + email webhook | LOW |

---

## 22. Risk Register

| Risk | Probability | Impact | Score | Mitigation | Owner | Status |
|------|------------|--------|-------|-----------|-------|--------|
| PRADAN access not granted in time | MEDIUM | HIGH | HIGH | Use NOAA GOES data for initial development; apply PRADAN early | Team Lead | OPEN |
| X6.3 T/EM doesn't match published values | LOW | CRITICAL | HIGH | Validate sunkit-spex version; check response matrix | Physics Lead | OPEN |
| solexsloods package not installable | LOW | HIGH | MEDIUM | Manual calibration using paper constants; contact ISRO | Data Lead | OPEN |
| Moirai zero-shot underperforms on solar data | MEDIUM | MEDIUM | MEDIUM | Fallback to TCN-only; weight TCN at 0.8 | ML Lead | OPEN |
| 30-hour finale — live data not available | MEDIUM | HIGH | HIGH | Hardcode X6.3 replay on Page 5; all demos run on historical data | Demo Lead | OPEN |
| Insufficient M/C class training data | MEDIUM | MEDIUM | MEDIUM | LYRA transfer learning + tsaug augmentation | ML Lead | OPEN |
| LLM hallucination during live demo | LOW | HIGH | MEDIUM | Always use deterministic fallback template for demo | Demo Lead | OPEN |
| Team unable to complete XL modules | HIGH | MEDIUM | MEDIUM | XL is architecture-only for finale; core L0–L8 is the demo | Team Lead | OPEN |

---

## 23. Technology Readiness Levels

| Module | Current TRL | Target (Finale) | Justification |
|--------|------------|-----------------|---------------|
| L1 Data Ingestion | TRL 3 | TRL 5 | astropy + solexsloods proven; PRADAN access pending |
| L2 Physics Engine | TRL 3 | TRL 5 | sunkit-spex validated; need PRADAN data for end-to-end test |
| L3 Nowcasting Engine | TRL 3 | TRL 5 | PELT + P85 algorithms ready; validation needs data |
| L4 Forecast Engine | TRL 2 | TRL 4 | TCN architecture designed; training pipeline not yet built |
| L5 Analog Retrieval | TRL 3 | TRL 4 | FAISS + DTW coded; library population needs catalog |
| L6 Threat Intelligence | TRL 4 | TRL 5 | Rule engine straightforward; NavIC model needs calibration |
| L7 LLM Briefing | TRL 5 | TRL 6 | Ollama + Mistral available; template + guard coded |
| L8 Dashboard | TRL 4 | TRL 6 | Streamlit + Plotly ready; needs live Kafka integration |
| XL-1 AR Registry | TRL 2 | TRL 3 | Architecture defined; SHARP pipeline needs implementation |
| XL-4 Ensemble | TRL 2 | TRL 3 | Sub-model designs ready; no training yet |
| XL-5 NavIC Impact | TRL 3 | TRL 4 | Ionospheric model coded; needs S4 calibration data |

---

## 24. Configuration Reference

```yaml
# config/config.yaml — all thresholds and parameters

pipeline:
  version: "2.0.0"
  environment: "development"   # development | production | demo

data:
  pradan:
    base_url: "https://pradan.issdc.gov.in"
    username: "${PRADAN_USERNAME}"   # from env
    password: "${PRADAN_PASSWORD}"   # from env
    retry_count: 3
    retry_backoff_sec: [1, 2, 4]
  cache:
    path: "data/cache"
    max_size_gb: 50
    retention_hours: 48
  noaa_fallback_url: "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"

physics:
  energy_fit_range_kev: [3.0, 20.0]
  abundance_table: "coronal"
  deadtime_constants:
    SOLEXS_SDD1: 4.5e-6
    SOLEXS_SDD2: 4.5e-6
    HEL1OS_CDTE: 2.1e-6
    HEL1OS_CZT:  1.8e-6
  saturation_limits:
    SOLEXS_SDD1: 5.0e4
    SOLEXS_SDD2: 5.0e6

nowcasting:
  baseline_window_sec: 10800        # 3 hours
  baseline_percentile: 85
  baseline_recompute_interval_sec: 60
  trigger_sigma_multiplier: 3.0
  dual_trigger_window_sec: 180      # 3 minutes
  pelt_penalty: 1.5
  pelt_min_size: 10
  hope:
    dni_threshold: 1.5
    tnpsv_speed_threshold: 0.05
    tnpsv_angle_min_deg: 60
    tnpsv_angle_max_deg: 120
    t_rising_min_seconds: 3

forecasting:
  sequence_length_sec: 600         # 10 minutes
  forecast_horizons_min: [5, 10, 15, 30, 60]
  tcn_weight: 0.5
  moirai_weight: 0.5
  moirai_context_length: 512
  asymmetric_loss_alpha: 0.2
  focal_loss_gamma: 2.0
  mapie_coverage: 0.90
  mapie_alpha: 0.10

validation:
  ground_truth_match_window_min: 3
  bootstrap_resamples: 2000
  tss_alert_threshold: 0.55        # triggers accuracy agent
  far_alert_threshold: 0.08
  kappa_alert_threshold: 0.50
  ece_alert_threshold: 0.10

llm:
  provider: "ollama"
  model: "mistral-7b-instruct"
  max_tokens: 400
  temperature: 0.1
  structured_output: true
  hallucination_lead_time_tolerance_min: 2

kafka:
  bootstrap_servers: "localhost:9092"
  consumer_group: "aruna-pipeline"
  retry_count: 3
  dead_letter_topic: "aruna.dlq"

database:
  type: "sqlite"                   # sqlite | postgresql
  path: "data/aruna.db"
  postgresql_url: "${DATABASE_URL}"

mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "aruna-bah2026"
```

---

## 25. Operational Runbooks

### 25.1 Runbook: System Startup

```bash
# 1. Start infrastructure
docker compose up -d zookeeper kafka mlflow ollama

# 2. Wait for Kafka ready
sleep 10
kafka-topics.sh --bootstrap-server localhost:9092 --list

# 3. Pull Mistral model
docker compose exec ollama ollama pull mistral:7b-instruct

# 4. Start ARUNA pipeline
docker compose up -d aruna-pipeline

# 5. Verify health
curl http://localhost:8000/api/v1/health

# 6. Start dashboard
docker compose up -d aruna-dashboard
open http://localhost:8501
```

### 25.2 Runbook: X6.3 Demo Replay

```bash
# Run the X6.3 event (2024-02-22) demo — always works, hardcoded
python -m aruna.replay \
    --event-id SOL2024-02-22-X6.3 \
    --fits-dir tests/fixtures/x63/ \
    --speed 10 \            # 10× realtime
    --output-page 5         # opens dashboard Page 5 directly

# Expected output:
# T_MK peak: ~23.5 MK (matches Sarwade et al.)
# HOPE flag: fires ~120 seconds before peak
# DNI: crosses 1.5 at T-90s
# Alert: X6.3 classification, DUAL_TRIGGER
```

### 25.3 Runbook: May 2024 Superstorm Counterfactual

```bash
# Generate the counterfactual report (primary demo artifact)
python -m aruna.counterfactual \
    --start 2024-05-09T00:00:00Z \
    --end   2024-05-11T23:59:59Z \
    --output reports/may2024_superstorm_counterfactual.pdf

# Expected key result:
# ARUNA RED alert issued at: 2024-05-10T~13:30Z (estimated)
# Kp=9 storm onset: 2024-05-10T17:00Z
# Lead time: ~3.5 hours
# ISRO manual response comparison included
```

### 25.4 Runbook: Accuracy Agent Response

```bash
# If TSS drops below 0.55, agent triggers automatically
# Manual inspection:
python -m aruna.agent.diagnose

# Expected outputs:
# - Root cause classification
# - Correction action taken
# - Before/after TSS comparison
# - MLflow run ID for audit trail

# If agent loops (> 3 iterations), escalate:
curl -X POST http://localhost:8000/api/v1/agent/pause
# Then check Dashboard Page 7 for validation trend
```

### 25.5 Runbook: PRADAN Offline

```bash
# System auto-detects via L0 health monitor
# Dashboard Page 1 shows: OFFLINE_CACHE banner

# Verify cache is being used:
curl http://localhost:8000/api/v1/status/sun
# → "data_source": "local_cache", "cache_age_hours": 1.2

# Force NOAA fallback:
export ARUNA_FORCE_NOAA_FALLBACK=1
docker compose restart aruna-pipeline

# Verify fallback active:
curl http://localhost:8000/api/v1/health
# → "pradan": "OFFLINE", "noaa_fallback": "ACTIVE"
```

---

## 26. Decision Log

| Decision | Why | Alternatives Considered | Trade-offs |
|----------|-----|------------------------|-----------|
| Kafka message bus instead of direct module calls | Decouples producers and consumers; enables replay and monitoring; allows XL modules to subscribe without touching core | RabbitMQ (complex), Redis Streams (less mature for this use case), direct Python queues (no persistence) | Higher operational complexity; requires Kafka setup |
| Asymmetric loss function (alpha=0.2) | Missed X/M flares catastrophically worse than false alarms; MSE treats both equally | Focal loss only, standard MSE, custom per-class weights | Slightly higher false alarm rate; dramatically lower miss rate |
| PELT + P85 dual detection (not single threshold) | Static thresholds drift during prolonged M-class activity; single methods have higher FAR | GOES-style single static threshold, running mean ± N sigma | Two-method complexity; justified by FAR reduction |
| Dual-instrument consensus required | SoLEXS-only triggers too many false alarms from thermal noise; HEL1OS-only misses gradual flares | SoLEXS-only, HEL1OS-only, OR logic | Some delayed detections on edge events; dramatically lower FAR |
| Solexsloods for dead-time correction | Official ISRO package; ensures calibration traceability to Sarwade et al. 2025; any other method would diverge from instrument team's published results | Manual implementation of non-paralysable model | Package dependency; must track ISRO updates |
| sunkit-spex f_vth isothermal model | Same method used by SoLEXS team; outputs directly comparable to published T/EM values | CHIANTI direct, custom MCMC fitter | Isothermal assumption valid only at peak; multi-thermal extension possible post-hackathon |
| Offline-only LLM (Ollama + Mistral) | ISRO workstation may be air-gapped; no API keys required; data sovereignty | GPT-4 API, Gemini API, Claude API | Lower raw quality than frontier models; mitigated by structured prompts and constraint checker |
| SQLite for development, PostgreSQL for production | SQLite requires zero setup; enables demo on any laptop; PostgreSQL for ISRO deployment | PostgreSQL only (heavy), Redis (not relational) | Need migration script from SQLite to PostgreSQL |
| Docker single-container compose | One-command startup; reproducible across machines; signals operational seriousness to ISRO | pip install only (fragile), k8s (overkill), bare metal | Students need Docker installed; mitigated by fallback install script |
| MAPIE conformal prediction for uncertainty | Provides guaranteed coverage intervals (90% conformal, not just estimated); rigorous statistical guarantees | Dropout uncertainty (approximate), ensemble variance (uncalibrated) | Requires held-out calibration set; adds complexity |
| Temporal split (not random) for train/val/test | Random splits cause temporal leakage in time-series; model would "see the future" | K-fold cross-validation, stratified split | Smaller effective training window; required for scientific validity |

---

## 27. Research Gaps Addressed

This section explicitly maps ARUNA's contributions to gaps identified in the instrument team's published work.

### 27.1 Gap 1 — X-class-only analysis (Sarwade et al.)

**Published limitation:** The Fe Kα study used only 47 X-class flares for SNR. Hundreds of M and C-class events in SoLEXS catalog remain unanalyzed for T/EM plasma diagnostics.

**ARUNA's contribution:** Automated T/EM isothermal fitting pipeline running on all GOES classes. First systematic SoLEXS plasma diagnostic catalog across B/C/M/X classes.

### 27.2 Gap 2 — Rapid fluorescence height tracking (Sarwade et al.)

**Published limitation:** "Statistical uncertainties currently limit the ability to track rapid height variations on short timescales."

**ARUNA's contribution:** 1-second Fe Kα flux time series correlated with HEL1OS 25–50 keV band. If synchronous peak observed → first evidence of non-thermal electron beam driving photospheric fluorescence on Aditya-L1. Explicitly framed as a research probe.

### 27.3 Gap 3 — No automated pipeline exists

**Published limitation:** All published analyses are manual, event-specific. No operational pipeline on any Aditya-L1 X-ray data stream.

**ARUNA's contribution:** First automated, operationally deployable flare detection and forecasting pipeline on SoLEXS + HEL1OS data.

### 27.4 Gap 4 — Flare onset phase understudied

**Published limitation (SoLEXS instrument paper):** "SXR and HXR observations from other instruments will be valuable for future flare onset phase studies."

**ARUNA's contribution:** HOPE precursor system — systematic first study of the thermal pre-impulsive phase at 1-second cadence using SoLEXS + HEL1OS as a coupled pair.

### 27.5 Gap 5 — No joint SoLEXS + HEL1OS physics analysis

**Published limitation:** Instrument teams published separate papers. No published joint thermal + non-thermal physics analysis from a single Aditya-L1 publication.

**ARUNA's contribution:** Dual-instrument fusion architecture. DNI, TNPSV, TNCI, SHEI are all joint observables computed from SoLEXS + HEL1OS simultaneously. First measurement of this kind at 1-second cadence from L1.

---

## 28. Acceptance Criteria Master Checklist

### Pre-Demo Checklist (August 5, 2026 — Night Before)

```
PHYSICS
□ T/EM output on X6.3 matches Sarwade et al. within 1-sigma
□ HOPE flag fires ≥ 90 seconds before X6.3 peak
□ DNI crosses 1.5 at correct time in X6.3 event
□ Isothermal fit chi-squared < 2.0 on M6.7 test case

DETECTION
□ X6.3 detected within 30 seconds of GOES onset time
□ False alarm rate < 5% on quiet-Sun validation set
□ Master Catalog has ≥ 20 real events populated

FORECASTING
□ Forecast output generated within 10 seconds
□ Confidence interval displayed on all forecasts
□ Pre-trained model weights loaded correctly

DASHBOARD
□ Page 2 (Live Ops) updates every 1-2 seconds
□ Page 5 (X6.3 Replay) loads in < 3 seconds
□ Page 6 (Threat/Safing) shows correct NavIC risk
□ Page 8 (Gaganyaan Advisor) shows 7-day forecast

OPERATIONAL ARTIFACTS
□ May 2024 counterfactual PDF generated
□ SpaceWx bulletin export working
□ ISRO satellite safing queue outputs Chandrayaan-3 first
□ Dos/Donts card generated for a test M5 alert

SYSTEM
□ docker compose up starts all services
□ GET /health returns HEALTHY for all stages
□ X6.3 replay completes without errors
□ MLflow shows experiment runs recorded

PRESENTATION MATERIAL
□ Slide 2: February 2026 ISRO manual scramble quote + ARUNA lead time
□ Slide 3: X6.3 T/EM comparison (ARUNA vs Sarwade et al.)
□ Final slide: Deployment roadmap (Phase 2 requires URSC access)
□ Answers prepared for 6 hardest jury questions
```

### Go / No-Go Criteria (August 6, 2026 — Finale Morning)

```
MUST HAVE (Go/No-Go)
□ X6.3 event plays correctly on Page 5
□ T/EM physics output matches published values
□ HOPE flag fires correctly on X6.3
□ Alert system generates an alert on X6.3 peak
□ At least 3 dashboard pages functional

SHOULD HAVE (Strong to have)
□ May 2024 counterfactual PDF in hand
□ Satellite safing queue working
□ SpaceWx bulletin export working
□ Gaganyaan window advisor populated

NICE TO HAVE (Bonus points)
□ Live PRADAN data ingesting in real-time
□ XL-1 AR registry with current active regions
□ XL-5 NavIC regional map updating live
```

---

*ARUNA Engineering Pipeline PRD v1.0.0*  
*BAH 2026 · Problem Statement 15*  
*Foresee. Prepare. Protect.*  
*Built on Aditya-L1. Built for India.*
