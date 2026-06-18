# L2 Physics Engine — Design Spec
## ARUNA · BAH 2026 · Phase 3

**Date:** 2026-06-18
**Status:** Revised — v2
**Builds on:** L1 Ingestion (Session 3, 58/58 tests passing)
**PRD reference:** §6 — L2 Physics Engine

---

## 1. Objective

Consume paired 1-second `ObservationFrame` objects from Kafka (`aruna.observations.solexs` + `aruna.observations.hel1os`), compute real plasma diagnostics (T, EM) and ARUNA's novel physics observables (DNI, TNPSV, SHEI, HFD, QPP, Fe Kα), evaluate the HOPE precursor gate, and publish `PhysicsFrame` objects to `aruna.physics.frames` at 1-second cadence.

This is the scientific core of ARUNA — the layer that distinguishes it from every other approach.

---

## 2. Architecture — Stateful Engine with Pure-Function Modules

`PhysicsEngine` owns internal ring buffers and orchestrates calls to pure-function module classes. Each module class is independently instantiable and testable with synthetic numpy arrays. The engine handles all stateful coordination (buffers, streak counters, Kafka I/O).

This mirrors the `IngestionPipeline` pattern from L1 exactly.

**Engine state:** `PhysicsEngine` exposes a read-only `state: EngineState` property:
```python
class EngineState(Enum):
    WARMUP    = "WARMUP"    # HFD buffer not yet full (< 600 frames processed)
    ACTIVE    = "ACTIVE"    # all modules producing valid output
    DEGRADED  = "DEGRADED"  # ≥ 50% of last 60 frames had physics_quality_flag != FIT_OK
```
Every `PhysicsFrame` carries `engine_state: str` (the string value). Downstream consumers (L3 nowcasting, frontend) use this to know whether early `NaN`s are expected warm-up behaviour or indicate a fault. The transition `WARMUP → ACTIVE` fires exactly once, at frame 600. `ACTIVE → DEGRADED` is re-evaluated every 60 frames; recovery back to `ACTIVE` requires 60 consecutive `FIT_OK` frames.

**Concurrency rule:** `PhysicsEngine` is **not thread-safe**. One instance must be created per Kafka consumer instance (i.e. per partition assignment). Instances must never be shared across threads. If horizontal scaling is needed in future, each worker process creates its own `PhysicsEngine`; state is never shared via Redis or any other external store at this layer.

```
python/aruna/l2_physics/
  models.py        — PhysicsFrame dataclass + PhysicsConfig
  chianti_table.py — CHIANTILookupTable (precomputed .npz; deterministic fallback for isothermal fit)
  isothermal.py    — IsothermalFitter (sunkit-spex f_vth preferred; CHIANTILookupTable fallback)
  dni.py           — DNIComputer
  tnpsv.py         — TNPSVComputer
  shei.py          — SHEIComputer
  hfd.py           — HFDComputer
  qpp.py           — QPPDetector
  feka.py          — FeKaExtractor
  hope.py          — HOPEEvaluator
  engine.py        — PhysicsEngine (stateful orchestrator)
  __init__.py      — public re-exports
  data/
    chianti_table.npz   — precomputed (T, EM) → synthetic spectrum lookup table
```

---

## 3. Data Flow

```
Kafka aruna.observations.solexs  ──┐
Kafka aruna.observations.hel1os  ──┤
                                   ▼
                         PhysicsEngine.process(sol, hel)
                           │
                           ├─ ring_buffer.append(sol, hel)
                           │
                           ├─ IsothermalFitter.fit(sol.counts, sol.energy_bins)
                           │       → T_MK, EM_cm3, T_unc, EM_unc, chi2
                           │
                           ├─ DNIComputer.update(hel band 25–50, sol band 8–15)
                           │       → DNI (cumulative integral)
                           │
                           ├─ TNPSVComputer.update(T_history, EM_history)
                           │       → magnitude, angle_rad
                           │
                           ├─ SHEIComputer.update(hel.counts, hel.energy_bins)
                           │       → SHEI, dSHEI_dt
                           │
                           ├─ HFDComputer.update(sol total flux)
                           │       → HFD (nan until 600 frames buffered)
                           │
                           ├─ QPPDetector.update(hel band 25–50 buffer)
                           │       → QPP_period_sec, QPP_amplitude (or None, None)
                           │
                           ├─ FeKaExtractor.extract(sol.counts, T_MK, EM_cm3)
                           │       → FeKa_flux
                           │
                           ├─ HOPEEvaluator.evaluate(T, dT_dt, EM, dEM_dt,
                           │     DNI, tnpsv_mag, tnpsv_angle, hel_flux, streak)
                           │       → hope_flag, reason, lead_time_sec
                           │
                           └─ assemble PhysicsFrame → KafkaPublisher
                                   → aruna.physics.frames
```

---

## 4. Module Specifications

### 4.1 `models.py` — PhysicsFrame + PhysicsConfig

```python
@dataclass
class PhysicsFrame:
    frame_id:            str
    timestamp:           datetime
    source_frame_ids:    List[str]   # provenance: L1 frame IDs used

    # Plasma diagnostics (IsothermalFitter)
    T_MK:                float
    T_uncertainty_MK:    float
    EM_cm3:              float
    EM_uncertainty:      float
    fit_chi_squared:     float       # target < 2.0

    # Novel observables
    DNI:                 float
    tnpsv_magnitude:     float
    tnpsv_angle_rad:     float
    SHEI:                float
    dSHEI_dt:            float
    HFD:                 float       # nan until 600 frames available
    QPP_period_sec:      Optional[float]
    QPP_amplitude:       Optional[float]
    FeKa_flux:           float

    # State
    flare_phase:         str         # QUIET | HOPE | IMPULSIVE | GRADUAL | DECAY
    hope_flag:           bool
    hope_lead_time_sec:  Optional[int]
    hope_reason:         Optional[str]

    # Engine state (warm-up visibility for downstream)
    engine_state:        str              # 'WARMUP' | 'ACTIVE' | 'DEGRADED'

    # Quality — two independent dimensions
    data_quality_flag:   QualityFlag      # inherited from L1 ObservationFrame (telemetry quality)
    physics_quality_flag: PhysicsQuality  # generated by L2 (fit + observable quality)
    fit_backend:         str              # 'sunkit_spex' | 'chianti_lookup' — never omitted
    pipeline_version:    str
    schema_version:      str = "1.0"
    created_at:          datetime    = field(default_factory=...)
```

`PhysicsQuality` enum (L2-generated, lives in `models.py`):
```python
class PhysicsQuality(IntEnum):
    FIT_OK           = 0   # isothermal fit converged, chi2 < 2.0
    FIT_DIVERGED     = 1   # optimizer failed or chi2 > 5.0; T/EM are NaN
    LOW_CONFIDENCE   = 2   # fit converged but chi2 in 2.0–5.0 range
    BACKEND_FALLBACK = 3   # chianti_lookup used instead of sunkit_spex
```

A frame can have `data_quality_flag=GOOD` and `physics_quality_flag=FIT_DIVERGED` (valid telemetry, bad fit) or `data_quality_flag=SAA_PASSAGE` and `physics_quality_flag=FIT_OK` (contaminated data but fit still converged). Downstream consumers gate on both independently.

`PhysicsConfig` holds all tunable thresholds (mapped from `config.yaml`):
- `HOPE_DNI_THRESHOLD = 1.5`
- `HOPE_TNPSV_SPEED_THRESHOLD = 0.05`
- `HOPE_TNPSV_ANGLE_MIN = 60°`, `HOPE_TNPSV_ANGLE_MAX = 120°`
- `EM_RISE_THRESHOLD` (for HOPE condition 2)
- `HEL_QUIET_MULTIPLIER = 2.0` (for HOPE condition 3)
- `HFD_WINDOW = 600`, `QPP_WINDOW = 200`
- `FIT_ENERGY_MIN = 3.0 keV`, `FIT_ENERGY_MAX = 20.0 keV`

### 4.2 `chianti_table.py` — CHIANTILookupTable

The **deterministic fallback** for isothermal fitting when `sunkit-spex` is unavailable. Precomputes a `(T_grid × EM_grid → synthetic_spectrum)` table using Mewe-Gronenschild-Kaastra thermal bremsstrahlung with coronal abundances. Stored as `data/chianti_table.npz`.

- T grid: 50 log-spaced points, 1–100 MK
- EM grid: 50 log-spaced points, 1e44–1e52 cm⁻³
- Energy grid: matches SoLEXS bin structure (3–20 keV, N bins)
- `.generate(energy_bins)` regenerates the table for a given detector response
- `.lookup(T_MK, EM_cm3, energy_bins)` interpolates using `scipy.interpolate.RegularGridInterpolator`

If `data/chianti_table.npz` is absent at first import, `generate()` is called automatically (takes ~5s).

This is **not** the ground truth — it is an approximation used to keep tests deterministic and offline. `sunkit-spex f_vth` with the CHIANTI atomic database is the scientifically correct fit per Sarwade et al. 2025.

### 4.3 `isothermal.py` — IsothermalFitter

Backend resolution at import time (logged at INFO level so every run is traceable):
```python
try:
    from sunkit_spex.fitting import FitIsothermal
    _BACKEND = 'sunkit_spex'
    logger.info("IsothermalFitter: using sunkit-spex f_vth backend")
except ImportError:
    _BACKEND = 'chianti_lookup'
    logger.warning("IsothermalFitter: sunkit-spex unavailable, using CHIANTILookupTable fallback")
```

`fit(counts_per_s_kev, energy_bins_kev, response_matrix=None) -> FitResult`:
- Restrict to `FIT_ENERGY_MIN`–`FIT_ENERGY_MAX` keV window
- With `sunkit_spex`: delegate to `FitIsothermal(model='f_vth', abundance_table='coronal')`
- With lookup table: `scipy.optimize.minimize(Nelder-Mead)` minimising Poisson reduced chi-squared over (T, EM) 2D grid
- Returns `FitResult(T_MK, EM_cm3, T_unc, EM_unc, chi2_reduced, backend)` where `backend` is the string `'sunkit_spex'` or `'chianti_lookup'`
- On optimizer failure or chi2 > 5.0: return `FitResult(nan, nan, nan, nan, chi2, backend)`

The `backend` field is passed through to `PhysicsFrame.fit_backend` on every frame — no silent fallback.

### 4.4 `dni.py` — DNIComputer

Maintains `_cumulative_hel` running sum. `update(hel_25_50_flux, sol_8_15_flux, dt=1.0)`:
- `_cumulative_hel += hel_25_50_flux * dt`
- `DNI = _cumulative_hel / (sol_8_15_flux + 1e-30)`
- Returns current DNI value
- `reset()` zeroes the accumulator — called by `engine.py` when `flare_phase` transitions to `QUIET` after having been non-QUIET for ≥ 30 consecutive frames (prevents spurious reset on brief noise spikes)

### 4.5 `tnpsv.py` — TNPSVComputer

Maintains `deque(maxlen=30)` each for T and EM history. `update(T_MK, EM_cm3)`:
- Appends to histories
- With ≥ 3 frames: compute `np.gradient` over history arrays
- Normalise by running max T and EM
- Returns `(magnitude, angle_rad)` where `angle = atan2(dEM/dt, dT/dt)`
- With < 3 frames: return `(0.0, 0.0)`

### 4.6 `shei.py` — SHEIComputer

`update(counts_hel_2d, energy_bins)`:
- Maintains `deque(maxlen=10)` for SHEI history
- Band sums: F_hard = 40–70 keV, F_soft = 10–25 keV
- `SHEI = log10(F_hard / (F_soft + 1e-30))`
- `dSHEI_dt = gradient(SHEI_history)[-1]` with ≥ 2 frames, else 0.0
- Returns `(SHEI, dSHEI_dt)`

`update_single(hel_frame)` convenience wrapper accepting `ObservationFrame`.

### 4.7 `hfd.py` — HFDComputer

Maintains `deque(maxlen=600)` of SoLEXS total flux. `update(total_flux)`:
- With < 600 frames: return `nan` — this is expected behaviour during warm-up, not an error
- With 600 frames: run Higuchi algorithm (k_max=10) over window
- Returns HFD value

Pure Higuchi implementation — no external library required, numpy only.

**Warm-up note:** The 120-frame X6.3 fixture covers only 2 minutes of data. HFD will return `nan` for every frame in that test. That is correct and expected. HFD unit tests use a standalone 700-frame synthetic time series fed directly to `HFDComputer` (bypassing the engine) so the window fills. The integration test asserts that HFD is `nan` for all 120 X6.3 frames (confirming warm-up logic works, not that HFD is broken).

For a full HFD end-to-end test, `conftest.py` will provide a `quiet_sun_extended_frames` fixture: 700 frames of synthetic quiet-Sun SoLEXS data prepended to the X6.3 sequence, giving HFD a warm buffer before the event begins.

### 4.8 `qpp.py` — QPPDetector

Maintains `deque(maxlen=200)` of HEL1OS 25–50 keV flux. `update(hel_25_50_flux)`:
- With < 50 frames: return `(None, None)` — expected warm-up behaviour
- LOWESS detrend (statsmodels, frac=0.3)
- Morlet CWT via `pywt.cwt`, scales 5–200
- Dominant period = scale with max mean power above 99th percentile noise floor
- Returns `(T_QPP_seconds, A_QPP_normalised)` or `(None, None)` if no ridge found

**Warm-up note:** The 120-frame X6.3 fixture will produce `(None, None)` for the first 50 frames. QPP unit tests inject a 200-frame sinusoidal signal directly. Integration test accepts `(None, None)` for early frames and asserts a non-None period is returned once 50 frames of the impulsive phase have accumulated.

### 4.9 `feka.py` — FeKaExtractor

`extract(counts_per_s_kev, energy_bins_kev, T_MK, EM_cm3)`:
- Build CHIANTI continuum model via `CHIANTILookupTable.lookup(T_MK, EM_cm3, energy_bins)`
- Compute residual: `counts - continuum`
- Sum residual in 6.2–6.6 keV window
- `np.clip(..., 0, None)` — fluorescence cannot be negative
- Returns scalar `FeKa_flux`
- If T_MK is NaN (fit failed): return 0.0

### 4.10 `hope.py` — HOPEEvaluator

Pure function — no state. `evaluate(T, dT_dt, EM, dEM_dt, DNI, tnpsv_mag, tnpsv_angle, hel_flux, hel_quiet_baseline, streak_sec, config)`:

`streak_sec` is the **elapsed time in seconds** (not a frame count) during which T has been continuously rising without a gap. The engine computes it using timestamps: each frame where `dT_dt > 0` and `current_timestamp - prev_timestamp <= 1.5s` (allowing one missed frame) increments the streak by `delta_t`; any gap > 1.5s or `dT_dt <= 0` resets it to 0.0. This makes HOPE resilient to L1 gaps and interpolated frames.

Five conditions (all must be True):
1. `dT_dt > 0 and streak_sec >= 3.0` — T rising for ≥ 3 seconds of continuous observation
2. `abs(dEM_dt) < config.EM_RISE_THRESHOLD` — EM still flat
3. `hel_flux < hel_quiet_baseline * config.HEL_QUIET_MULTIPLIER` — non-thermal not dominant
4. `tnpsv_mag > config.HOPE_TNPSV_SPEED_THRESHOLD and ANGLE_MIN < tnpsv_angle < ANGLE_MAX`
5. `DNI > 0.2` — above noise floor

Returns `(hope_flag: bool, reason: str, lead_time_sec: int)`.
`lead_time_sec = int(90 + streak_sec * 2)` (empirical, validated on X6.3).

### 4.11 `engine.py` — PhysicsEngine

Stateful orchestrator.

**State:**
- `_sol_buffer: deque(maxlen=600)` — SoLEXS `ObservationFrame` history
- `_hel_buffer: deque(maxlen=600)` — HEL1OS `ObservationFrame` history
- `_hope_streak: int` — consecutive frames where T was rising
- `_hel_quiet_baseline: float` — rolling mean of HEL1OS flux in quiet period (updated when `flare_phase == QUIET`)
- `_dni_computer`, `_tnpsv_computer`, `_shei_computer`, `_hfd_computer`, `_qpp_detector` — module instances

**`process(sol_frame, hel_frame) -> PhysicsFrame`:**
1. Validate timestamp proximity (warn if > 1s apart)
2. Append to ring buffers
3. Call each module in data-flow order
4. Update `_hope_streak` (increment if `dT_dt > 0`, else reset to 0)
5. Update `_hel_quiet_baseline` if in QUIET phase
6. Assign `flare_phase` (priority order: IMPULSIVE → HOPE → DECAY → GRADUAL → QUIET)
7. Assemble and return `PhysicsFrame`
8. If `_publisher`: publish to `aruna.physics.frames`

**`reset()`:** Clears all buffers and resets streak/baseline. Called between replay events.

---

## 5. Flare Phase Assignment

Priority order (first match wins):

| Condition | Phase |
|-----------|-------|
| `DNI > 1.5 and SHEI > -0.8` | `IMPULSIVE` |
| `hope_flag == True` | `HOPE` |
| `SHEI < -1.5 and dT_dt < 0` | `DECAY` |
| `dT_dt > 0 or DNI > 0.2` | `GRADUAL` |
| else | `QUIET` |

---

## 6. Error Handling

| Failure Mode | Behaviour |
|---|---|
| Isothermal fit diverges (chi2 > 5.0 or optimizer error) | T_MK=NaN, EM_cm3=NaN, `physics_quality_flag=FIT_DIVERGED`, still publish frame |
| Isothermal fit converged but chi2 in 2.0–5.0 | T_MK/EM valid, `physics_quality_flag=LOW_CONFIDENCE` |
| `chianti_lookup` backend active | `physics_quality_flag=BACKEND_FALLBACK` (unless also diverged) |
| sol and hel timestamps differ by > 1s | Log WARNING, still process, `data_quality_flag=GAP` |
| Buffer has < 600 frames (HFD warm-up) | `engine_state=WARMUP`; HFD=NaN; QPP=(None,None) — not an error |
| Buffer has < 3 frames (TNPSV/DNI) | TNPSV returns (0.0, 0.0); DNI returns 0.0 |
| pywt not installed | QPPDetector returns (None, None) on every call; log WARNING once at init |
| FeKa extract with NaN T/EM | Return 0.0 |
| Any unhandled exception in module | Log ERROR with frame_id, `physics_quality_flag=FIT_DIVERGED`, still publish frame |

---

## 7. Testing Strategy

### 7.1 Unit Tests (per module, synthetic numpy arrays)

- `test_isothermal.py` — fit on known synthetic spectrum; assert T within 1 MK, chi2 < 2.0
- `test_dni.py` — step function input; assert cumulative integral correctness
- `test_tnpsv.py` — linear T ramp + flat EM; assert angle ≈ 90°
- `test_shei.py` — hard-band dominated spectrum; assert SHEI > -0.8
- `test_hfd.py` — pure sine wave (known HFD ≈ 1.0) vs random noise (HFD ≈ 1.5)
- `test_qpp.py` — injected sinusoidal pulsation; assert period recovered within 10%
- `test_feka.py` — spectrum with known 6.4 keV line; assert FeKa_flux > 0
- `test_hope.py` — parametric truth table across all 32 condition combinations

### 7.2 Integration Test — X6.3 Golden Fixture

Feed the 120-frame X6.3 sequence through `PhysicsEngine`. Assert:

| Criterion | Threshold |
|-----------|-----------|
| T_MK at peak frame | 23.5 ± 1.5 MK (Sarwade et al.) |
| HOPE flag first fires before peak | ≥ 90s before peak frame |
| SHEI at impulsive window | crosses -0.8 threshold |
| DNI at HOPE firing time | > 1.5 |
| chi2 < 2.0 | ≥ 90% of frames |
| FeKa_flux > 0 | ≥ 80% of frames in impulsive/gradual phase |
| Engine runtime per frame | < 1.0 second |
| All PhysicsFrames have source_frame_ids | 100% |

### 7.3 CHIANTI Table Test

- `test_chianti_table.py` — generate small table; assert `lookup` interpolation matches grid points exactly

---

## 8. Dependencies to Install

```bash
pip install pywavelets statsmodels
# sunkit-spex optional (attempted at import, skipped if unavailable)
# chianti_table.npz generated automatically on first import if absent
```

---

## 9. Build Order

1. `models.py` — PhysicsFrame + PhysicsConfig
2. `chianti_table.py` — lookup table generator + interpolator
3. `isothermal.py` — fitter with fallback chain
4. `dni.py`, `tnpsv.py`, `shei.py` — three core novel observables
5. `hfd.py`, `qpp.py` — enrichment modules (windowed)
6. `feka.py` — Fe Kα extraction (depends on chianti_table)
7. `hope.py` — pure-function gate (no state, easy to test)
8. `engine.py` — orchestrator
9. `__init__.py` — public API
10. Tests throughout, X6.3 fixture as anchor

---

## 10. Acceptance Criteria (PRD §6.12)

```
✓ T/EM output on X6.3 event matches Sarwade et al. within 1-sigma
✓ Isothermal fit chi-squared < 2.0 on > 90% of M+ class spectra
✓ DNI > 1.5 fires HOPE flag on X6.3 event within 2 minutes of impulsive onset
✓ TNPSV in precursor quadrant > 90 seconds before X6.3 peak
✓ SHEI transitions from < -1.5 to > -0.8 during impulsive phase of X6.3
✓ Fe Kα fluorescence detected in at least 80% of X-class events
✓ Physics engine runtime < 1 second per 1-second observation window
✓ All PhysicsFrames include provenance hash linking to source ObservationFrames
```
