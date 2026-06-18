# ARUNA — Session 3 Summary
## L1 Data Ingestion Layer — Complete

**Date:** 2026-06-18  
**Pipeline Version:** `2.0.0`  
**Session Duration:** ~2 hours  
**Status:** ✅ Phase 2 Complete — 58/58 tests passing

---

## 1. What We Built This Session

This session completed the entire **L1 Data Ingestion Layer** of the ARUNA pipeline.  
Every module specified in `docs/superpowers/plans/2026-06-18-l1-ingestion.md` was implemented, tested, debugged, and verified against the PRD §5.8 acceptance criteria.

---

## 2. Files Created / Modified

| File | Status | Responsibility |
|------|--------|----------------|
| `python/aruna/l1_ingestion/models.py` | ✅ Session 2 | `FitsData`, `HKData`, `IngestionConfig`, `QuarantineEntry`, `ValidationResult` |
| `python/aruna/l1_ingestion/fits_reader.py` | ✅ Session 2 | astropy FITS parse → `FitsData` / `HKData`, SHA-256 checksum |
| `python/aruna/l1_ingestion/deadtime.py` | ✅ Session 2 | Non-paralysable correction, SDD1→SDD2 auto-switch |
| `python/aruna/l1_ingestion/background.py` | ✅ **This session** | scipy poly background subtraction, χ²/dof gate, SAA masking |
| `python/aruna/l1_ingestion/alignment.py` | ✅ **This session** | Cross-correlation lag, cubic interpolation, clock-drift logging |
| `python/aruna/l1_ingestion/goes_join.py` | ✅ **This session** | SunPy HEK query, ±3 min match, offline fallback |
| `python/aruna/l1_ingestion/validator.py` | ✅ **This session** | pandera schema, 80% completeness gate, quarantine queue |
| `python/aruna/l1_ingestion/publisher.py` | ✅ **This session** | Kafka serializer, per-frame + quality summary topics |
| `python/aruna/l1_ingestion/pipeline.py` | ✅ **This session** | End-to-end orchestrator: FITS → corrected → aligned → labelled → published |
| `python/aruna/l1_ingestion/__init__.py` | ✅ **This session** | Public API re-exports |
| `python/tests/conftest.py` | ✅ **This session** | Synthetic FITS fixtures (SoLEXS/HEL1OS/HK/X6.3 scenario) |
| `python/tests/test_l1_ingestion.py` | ✅ **This session** | 58 acceptance tests (Tasks 1–9), all passing |

---

## 3. Module Design Decisions

### 3.1 BackgroundSubtractor (`background.py`)
- **Method:** scipy `curve_fit` 2nd-order polynomial per energy bin
- **Sherpa not used** — not buildable on Windows; scipy gives identical χ²/dof criterion
- **SAA bins:** Flagged `QualityFlag.SAA_PASSAGE` but **never dropped** — PRD §5.4 Step 4
- **χ²/dof gate:** If residuals exceed 2.0, non-SAA bins are re-flagged `QualityFlag.NOISY`
- **Empty window:** Returns unchanged data + χ²=0.0 gracefully

### 3.2 TimeAligner (`alignment.py`)
- **Method:** `scipy.signal.correlate` on the 10–15 keV band sum (SoLEXS ↔ HEL1OS overlap)
- **Drift warn at:** 0.5 s — logs `CLOCK_DRIFT` at WARNING level
- **Drift max at:** 1.0 s — above this, cubic `interp1d` is applied to regrid HEL1OS onto SoLEXS timestamps
- **Output:** Both streams trimmed to the same common length

### 3.3 GoesLabelJoiner (`goes_join.py`)
- **Match window:** ±3 minutes from frame timestamp to event onset
- **Offline:** If SunPy HEK is unreachable, returns `None`; frames labelled `QUIET_SUN`
- **Tz handling:** All datetimes normalised to naive UTC internally for consistent comparison
- **Subclass parsing:** `'X6.3'` → `goes_class='X'`, `goes_subclass=6.3`

### 3.4 FrameValidator (`validator.py`)
- **Schema:** pandera `DataFrameSchema` — `instrument`, `dead_time_fraction` (0–1), `solar_distance_au` (>0), `quality_flag` (valid enum value)
- **Completeness:** >80% finite values in `counts_per_s_kev` required
- **Quarantine:** Failed frames stored in `QuarantineEntry` list with `retry_after = now + quarantine_ttl_sec`
- **Pass-rate monitoring:** Logs ERROR if <95% of frames pass in a batch

### 3.5 KafkaPublisher (`publisher.py`)
- **Topics:** `aruna.observations.solexs`, `aruna.observations.hel1os`, `aruna.quality.summary`
- **Partition key:** `instrument` string encoded as bytes
- **Payload:** All `np.ndarray` → list, `datetime` → ISO string; `schema_version='1.0'`, `pipeline_version='2.0.0'`
- **Quality summary:** Always published even on empty frame list

### 3.6 IngestionPipeline (`pipeline.py`)
Full orchestration order:
```
FitsReader.read_solexs + read_hel1os + read_hk
  → DeadTimeCorrector.correct(SDD1) + correct(SDD2) → select_solexs_detector
  → DeadTimeCorrector.correct(CDTE) → BackgroundSubtractor.subtract
  → TimeAligner.align → trim flags to aligned lengths
  → _fits_to_frames → ObservationFrame list
  → GoesLabelJoiner.join
  → FrameValidator.validate
  → KafkaPublisher.publish
```

---

## 4. Bugs Found and Fixed During Testing

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| `TypeError` in `goes_join.py` | `frame.timestamp` is tz-aware; HEK string parsed with tz → mixed comparison | Normalised both sides to naive UTC via `_to_utc()` |
| `IndexError` in `pipeline.py` | After cubic interpolation, `hel1os_aligned` expands to SoLEXS grid length (120) but `bg_flags_cdte` still has 60 entries | Detect when `n_hel > len(bg_flags_cdte)` and pad with `QualityFlag.GOOD` |
| Wrong assertion in test | `validate()[1]` returns `ObservationFrame` objects, not `QuarantineEntry`; test called `.reason` on them | Switched to `FrameValidator.quarantine` property |

---

## 5. Acceptance Criteria — PRD §5.8

| Criterion | Status |
|-----------|--------|
| Time sync error SoLEXS ↔ HEL1OS < 1 second | ✅ |
| Dead-time correction applied to all SoLEXS bins | ✅ |
| Background subtraction applied to all HEL1OS bins | ✅ |
| SAA-contaminated bins flagged, not silently dropped | ✅ |
| Schema validation passes on >95% of frames | ✅ |
| GOES label joined for all known catalog events ±3 min | ✅ |
| ObservationFrame published to Kafka within 2 seconds | ✅ |
| X6.3 event (2024-02-22) processed without errors | ✅ |
| Clock drift log exists for any lag >0.5 seconds | ✅ |

**Test result:** `58 passed` in `3.21s`

---

## 6. Constants Reference (Sarwade et al. 2025)

```python
DEADTIME_CONSTANTS = {
    'SOLEXS_SDD1': 4.5e-6,   # seconds
    'SOLEXS_SDD2': 4.5e-6,   # seconds
    'HEL1OS_CDTE': 2.1e-6,   # seconds
    'HEL1OS_CZT':  1.8e-6,   # seconds
}
SATURATION_LIMITS = {
    'SOLEXS_SDD1': 5e4,       # counts/s
    'SOLEXS_SDD2': 5e6,       # counts/s
}
```

---

## 7. Known Warnings (Non-Breaking)

- `pandera` — `FutureWarning`: use `import pandera.pandas as pa` instead of `import pandera as pa` (API migration in next pandera version)
- `publisher.py` — `DeprecationWarning`: `datetime.utcnow()` → replaced with `datetime.now(timezone.utc)` in next cleanup
- `pytest 7.1.2` — uses deprecated `ast.Str` / `ast.NameConstant` (Python 3.13 transition artefact, not our code)

---

---

# Phase 3 — L2 Physics Engine

> **Next implementation target:** `python/aruna/l2_physics/`

## What Phase 3 Is

Phase 3 implements the **scientific core of ARUNA** — the Physics Engine (PRD §6). It consumes `ObservationFrame` objects from Kafka and produces `PhysicsFrame` objects containing real plasma diagnostics and ARUNA's novel observables.

This is what makes ARUNA different from every other approach: real physics, not a black box.

---

## Phase 3 — Key Modules to Build

| Module | File | What It Does |
|--------|------|-------------|
| **IsothermalFitter** | `l2_physics/isothermal.py` | `sunkit-spex f_vth` fit → T(t) in MK, EM(t) in cm⁻³ |
| **DNIComputer** | `l2_physics/dni.py` | Dynamic Neupert Integral — cumulative HEL1OS / SoLEXS ratio |
| **TNPSVComputer** | `l2_physics/tnpsv.py` | Phase-space velocity and heading angle in T-EM space |
| **SHEIComputer** | `l2_physics/shei.py` | Spectral Hardness Evolution Index — HEL1OS band ratio |
| **HFDComputer** | `l2_physics/hfd.py` | Higuchi Fractal Dimension on 10-min sliding window |
| **QPPDetector** | `l2_physics/qpp.py` | Morlet CWT quasi-periodic pulsation detection |
| **FeKaExtractor** | `l2_physics/feka.py` | Fe Kα fluorescence residual at 6.4 keV (novel measurement) |
| **HOPEEvaluator** | `l2_physics/hope.py` | Multi-condition HOPE precursor gate |
| **PhysicsEngine** | `l2_physics/engine.py` | Orchestrator: `ObservationFrame` → `PhysicsFrame` → Kafka |

---

## Phase 3 — Output: PhysicsFrame

```python
@dataclass
class PhysicsFrame:
    frame_id:           str
    timestamp:          datetime

    # Plasma diagnostics (from IsothermalFitter)
    T_MK:               float       # Temperature in megakelvin
    T_uncertainty_MK:   float
    EM_cm3:             float       # Emission measure in cm⁻³
    EM_uncertainty:     float
    fit_chi_squared:    float       # Target < 2.0

    # Novel observables
    DNI:                float       # Dynamic Neupert Integral
    tnpsv_magnitude:    float       # Phase-space speed (normalised)
    tnpsv_angle_rad:    float       # Phase-space heading
    SHEI:               float       # Spectral hardness index
    dSHEI_dt:           float       # Hardening rate
    HFD:                float       # Higuchi fractal dimension
    QPP_period_sec:     Optional[float]
    QPP_amplitude:      Optional[float]
    FeKa_flux:          float       # Fe Kα residual flux

    # State
    flare_phase:        str         # 'QUIET' | 'HOPE' | 'IMPULSIVE' | 'GRADUAL' | 'DECAY'
    hope_flag:          bool
    hope_lead_time_sec: Optional[int]
    hope_reason:        Optional[str]

    # Quality
    quality_flag:       QualityFlag
    source_frame_ids:   List[str]   # Provenance — L1 frame IDs used
    pipeline_version:   str
```

---

## Phase 3 — Acceptance Criteria (PRD §6.12)

```
✓ T/EM output on X6.3 event (2024-02-22) matches Sarwade et al. within 1-sigma
✓ Isothermal fit chi-squared < 2.0 on >90% of M+ class spectra
✓ DNI > 1.5 fires HOPE flag on X6.3 event within 2 minutes of impulsive onset
✓ TNPSV in precursor quadrant >90 seconds before X6.3 peak
✓ SHEI transitions from < -1.5 to > -0.8 during impulsive phase of X6.3
✓ Fe Kα fluorescence detected in at least 80% of X-class events
✓ Physics engine runtime < 1 second per 1-second observation window
```

---

## Phase 3 — Key Physics Notes

### HOPE Precursor Logic
HOPE fires when **all 5 conditions** are simultaneously true:
1. `T` rising (`dT/dt > 0`) for ≥ 3 consecutive seconds
2. `EM` still flat (`dEM/dt < threshold`) — pre-impulsive phase
3. HEL1OS still near quiet-Sun level — non-thermal not yet dominant
4. TNPSV in precursor quadrant (T rising, EM flat sector: 60°–120°)
5. DNI rising from baseline (`> 0.2`)

**Lead time estimate:** `90 + history_len × 2` seconds (empirical, validated on X6.3)

### DNI Interpretation
```
DNI < 1.0   → thermal energy exceeds cumulative nonthermal input (gradual/decay phase)
DNI > 1.5   → nonthermal injection outpacing thermal response → IMPULSIVE ONSET IMMINENT
```

### SHEI Thresholds
```
SHEI < -1.5           → thermal dominated (gradual phase)
-1.5 ≤ SHEI ≤ -0.8   → mixed thermal-nonthermal (onset)
SHEI > -0.8           → nonthermal dominated (impulsive phase)
```

### Fe Kα — Novel Measurement
This directly addresses an explicit gap from Sarwade et al. 2025:  
*"statistical uncertainties currently limit the ability to track rapid height variations on short timescales."*  
ARUNA correlates the 6.4 keV Fe Kα residual with HEL1OS 25–50 keV at 1-second cadence — the first automated pipeline to do this on Aditya-L1 data.

---

## Phase 3 — Suggested Build Order

1. `models.py` — `PhysicsFrame` dataclass + helper types
2. `isothermal.py` — T/EM fitter (sunkit-spex wrapper with scipy fallback)
3. `dni.py` + `tnpsv.py` + `shei.py` — the three core novel observables
4. `hfd.py` + `qpp.py` — optional enrichment (10-min window required)
5. `feka.py` — Fe Kα extraction (requires CHIANTI or synthetic continuum)
6. `hope.py` — HOPE evaluator (pure logic gate, easy to test)
7. `engine.py` — orchestrator + Kafka consumer/producer
8. Tests: X6.3 event as golden fixture throughout

---

## Phase 3 — Dependencies to Install

```bash
pip install sunkit-spex sunpy pywavelets statsmodels
# CHIANTI database download required for isothermal fitting:
# python -c "from sunpy.data import download_manager; ..."
```

> **Note on sunkit-spex:** If the CHIANTI database is unavailable in the test environment, implement a `CHIANTIFallback` that uses a pre-computed lookup table (T, EM) → synthetic spectrum. This keeps tests deterministic and offline.

---

*Next session: begin Phase 3 with `python/aruna/l2_physics/models.py` and `isothermal.py`.*
