# L3 Nowcasting Engine — Design Spec

**Date:** 2026-06-19
**Phase:** 4
**Status:** Approved — ready for implementation
**PRD reference:** §7 (L3 — Nowcasting Engine)
**Prerequisite:** L2 Physics Engine complete (153 tests passing, Session 4)

---

## 1. Objective

Detect solar flares in real time from `PhysicsFrame` objects produced by L2, confirm via dual-instrument consensus, classify by GOES-equivalent class, track the full flare lifecycle, and write every confirmed event to the Master Flare Catalog. This is the primary operational detection layer of ARUNA.

---

## 2. Architecture

Approach A — module-per-component, mirroring the L2 pattern exactly.

```
python/aruna/l3_nowcasting/
  models.py         # NowcastConfig (all thresholds)
  baseline.py       # RollingP85Baseline
  changepoint.py    # ChangepointDetector (ruptures PELT)
  goes_class.py     # GOESClassifier (calibration factor → W/m²)
  consensus.py      # ConsensusEngine (4-state FSM + overlap detection)
  catalog.py        # CatalogService (SQLite dev / PostgreSQL prod)
  orchestrator.py   # NowcastOrchestrator (Kafka consumer, wires all)
  __init__.py       # public re-exports

python/tests/
  test_l3_nowcasting.py   # replaces all xfail stubs
  conftest.py             # adds L3 fixtures (synthetic_flare_stream)
```

### Data Flow

```
PhysicsFrame (from Kafka or in-memory in tests)
  → NowcastOrchestrator
      → RollingP85Baseline[SoLEXS].update(flux, t)  → (p85, threshold)
      → RollingP85Baseline[HEL1OS].update(flux, t)  → (p85, threshold)
      → ChangepointDetector[SoLEXS].update(flux)    → changepoint index | None
      → ChangepointDetector[HEL1OS].update(flux)    → changepoint index | None
      → Derive: sol_triggered, hel_triggered, hope_flag from PhysicsFrame
      → ConsensusEngine.ingest(...)                 → NowcastEvent | None
          (on state transition: HOPE_PRECURSOR / FLARE_ONSET / FLARE_PEAK / FLARE_END)
      → GOESClassifier.classify(peak_flux)          → (letter, subclass)
      → CatalogService.write(event, peak_metrics)   → master_flare_catalog row
  → Kafka publish: aruna.nowcast.events
```

---

## 3. Module Interfaces

### 3.1 `models.py` — NowcastConfig

```python
@dataclass
class NowcastConfig:
    # Baseline
    baseline_window_sec: int = 10800       # 3-hour P85 window
    baseline_percentile: float = 85.0
    baseline_recompute_interval: int = 60
    trigger_sigma_multiplier: float = 3.0

    # PELT
    pelt_penalty: float = 1.5
    pelt_min_size: int = 10
    pelt_model: str = "rbf"
    pelt_flux_window: int = 600            # last 600s fed to PELT

    # Consensus / state machine
    dual_trigger_window_sec: int = 180     # both instruments within 3 min
    peak_decay_threshold: float = 0.5     # flux < 50% of peak → FLARE_END
    long_duration_lock_sec: int = 21600   # 6-hour max baseline lock

    # GOES calibration
    solexs_to_wm2_factor: float = 1.0e-8  # counts/s/keV → W/m²

    # Escalation
    x3_auto_critical: bool = True
    sep_kp_threshold: int = 6

    # Kafka
    kafka_topic_input: str = "aruna.physics.frames"
    kafka_topic_output: str = "aruna.nowcast.events"

    # Database
    db_url: str = "sqlite:///aruna_catalog.db"
    pipeline_version: str = "2.0.0"
```

### 3.2 `baseline.py` — RollingP85Baseline

```python
baseline.update(flux: float, t_sec: int) -> Tuple[float, float]
# Returns (p85_baseline, trigger_threshold)
# threshold = p85 + sigma_multiplier * std(first_3600_samples)
# Baseline is None until window has at least 1 sample.

baseline.freeze()    # locks baseline value — called at FLARE_ONSET
baseline.unfreeze()  # resumes updates — called at FLARE_END
baseline.is_frozen() -> bool
```

- Uses `deque(maxlen=baseline_window_sec)` — O(1) append
- P85 recomputed every `recompute_interval` seconds (60s default)
- σ estimated from the first 3600 samples (first-hour quiet-Sun estimate per PRD §7.4)
- `freeze()` prevents contamination from long-duration flares

### 3.3 `changepoint.py` — ChangepointDetector

```python
detector.update(flux: float) -> Optional[int]
# Appends flux to deque(maxlen=pelt_flux_window=600).
# Returns index of most recent PELT breakpoint once buffer fills.
# Returns None during warm-up or when no changepoint found.
```

- Wraps `ruptures.Pelt(model='rbf', min_size=10).fit(flux_array).predict(pen=1.5)`
- Buffer must be full before PELT runs (warm-up guard)
- Returns `breakpoints[-2]` (most recent internal breakpoint) per PRD §7.5

### 3.4 `goes_class.py` — GOESClassifier

```python
classifier.classify(peak_flux_counts: float) -> Tuple[str, float]
# Applies solexs_to_wm2_factor, then maps W/m² to GOES letter + subclass.
# e.g. 12500 counts/s/keV → 1.25e-4 W/m² → ('X', 1.25 / 1e-4 * 1e-4) = ('X', 1.3)
# Returns ('A', subclass) for sub-B events.

classifier.is_major_xclass(goes_class: str, subclass: float) -> bool
# True for X3+ (goes_class='X' and subclass >= 3.0)
```

GOES thresholds (1–8 Å equivalent W/m²):
- X: ≥ 1e-4
- M: ≥ 1e-5
- C: ≥ 1e-6
- B: ≥ 1e-7
- A: below B

### 3.5 `consensus.py` — ConsensusEngine (4-state FSM)

**States:** `QUIET → HOPE → ONSET → PEAK → END → QUIET`

```python
engine.ingest(
    sol_triggered: bool,         # SoLEXS: flux > threshold AND PELT change-point
    hel_triggered: bool,         # HEL1OS: flux > threshold AND SHEI hardening
    hope_flag: bool,             # from PhysicsFrame.hope_flag
    hope_reason: Optional[str],
    physics_frame: PhysicsFrame,
    flux_solexs: float,          # raw band flux for peak tracking
    flux_hel1os: float,
    kp_index: Optional[float] = None,
) -> Optional[NowcastEvent]
# Returns a NowcastEvent on each state transition, else None.
```

**Transition rules:**

| From | To | Condition | Event emitted |
|------|----|-----------|---------------|
| QUIET | HOPE | `hope_flag=True` | `HOPE_PRECURSOR` |
| QUIET/HOPE | ONSET | `sol_triggered AND hel_triggered` within 3-min window | `FLARE_ONSET` |
| ONSET | PEAK | flux stops rising (dflux/dt < 0 for 3 consecutive frames) | `FLARE_PEAK` |
| PEAK | END | flux falls below 50% of peak value | `FLARE_END` |
| PEAK | END | duration > 6 hours (long-duration guard) | `FLARE_END` |

**Overlapping flare detection:**
- New changepoint during DECAY (PEAK state, flux still above baseline) → new `event_id`, `flare_type='OVERLAPPING'`, both events active simultaneously

**Single-instrument degraded mode:**
- `sol_triggered=True, hel_triggered=False` → emits `FLARE_ONSET` with `detection_method='SOLEXS_ONLY'`, `confidence` capped at 0.5

**SEP event classification:**
- `kp_index > sep_kp_threshold AND NOT hope_flag` → `detection_method='SEP_EVENT'`

**X3+ escalation:**
- When GOES class is X3+, `confidence` forced to 1.0 and alert flagged CRITICAL in the NowcastEvent notes field

### 3.6 `catalog.py` — CatalogService

```python
catalog = CatalogService(db_url="sqlite:///aruna_catalog.db")

catalog.write(
    event: NowcastEvent,
    peak_T_MK: float = 0.0,
    peak_EM_cm3: float = 0.0,
    peak_flux_solexs: float = 0.0,
    peak_flux_hel1os: float = 0.0,
    peak_DNI: float = 0.0,
) -> None
# Upserts by event_id. Safe to call for FLARE_ONSET, FLARE_PEAK, FLARE_END
# (each call fills in more columns). < 100ms guaranteed.
```

- Schema: `master_flare_catalog` as defined in PRD §7.8
- Uses raw SQLite `sqlite3` (dev) or `psycopg2` (prod) — no ORM overhead
- Exception caught and logged; Kafka publish is not blocked by DB failure

### 3.7 `orchestrator.py` — NowcastOrchestrator

```python
orchestrator = NowcastOrchestrator(
    config: NowcastConfig,
    kafka_consumer=None,   # None → in-memory mode for tests
    kafka_producer=None,
)
orchestrator.process(physics_frame: PhysicsFrame) -> Optional[NowcastEvent]
# Single-frame entry point — used directly in tests and replay.
# In production, called from Kafka consumer loop.
```

- Owns: 2× `RollingP85Baseline` (SoLEXS + HEL1OS), 2× `ChangepointDetector`, 1× `ConsensusEngine`, 1× `CatalogService`
- Derives `sol_triggered` = flux > sol_threshold AND sol changepoint detected
- Derives `hel_triggered` = flux > hel_threshold AND SHEI hardening (`frame.dSHEI_dt > 0 AND frame.SHEI > -1.0`)
- Not thread-safe — one instance per Kafka partition

---

## 4. Testing Strategy

### 4.1 Unit tests (per module)

| Class | Key tests |
|-------|-----------|
| `TestRollingP85Baseline` | correct P85 value, threshold = P85 + 3σ, freeze holds value fixed, unfreeze resumes |
| `TestChangepointDetector` | flat signal → None, step change → returns index, warm-up guard |
| `TestGOESClassifier` | X6.3 maps to ('X', ~6), M/C/B ranges, sub-A, X3+ major flag |
| `TestConsensusEngine` | SoLEXS-only no confirmed onset, dual-trigger within window fires onset, outside window no onset, HOPE emits precursor, full 4-state transition chain, overlapping flare distinct event_ids, SEP classification on high Kp |
| `TestCatalogService` | write < 100ms, upsert idempotent, provenance hash stored |

### 4.2 Integration tests (orchestrator)

| Test | PRD criterion |
|------|--------------|
| `test_x63_detected_within_30s` | X6.3 (2024-02-22) detected ≤ 30s of GOES onset |
| `test_hope_fires_90s_before_peak` | HOPE precursor ≥ 90s before X6.3 peak |
| `test_tpr_far_tss_synthetic_injection` | TPR > 0.80, FAR < 0.05, TSS > 0.65 on synthetic M/X set |
| `test_all_events_have_provenance_hash` | Every NowcastEvent has non-empty provenance_hash |
| `test_x3_escalated_to_critical` | X3+ confidence forced to 1.0 |
| `test_solexs_only_no_confirmed_onset` | Single-instrument never sets detection_method='DUAL_TRIGGER' |
| `test_overlapping_flares_distinct_ids` | Overlapping flare → 2 distinct event_ids |
| `test_catalog_write_under_100ms` | CatalogService write latency |

### 4.3 Synthetic flare fixture

`conftest.py` gets `synthetic_flare_stream(goes_class, subclass, rise_sec, peak_sec, decay_sec)` — generates a list of `PhysicsFrame` objects with injected flux profile (Gaussian rise + exponential decay), `hope_flag=True` during precursor, valid SHEI hardening during impulsive phase. Used for TPR/FAR/TSS tests.

### 4.4 xfail stubs retained for

- `test_tpr_over_0_80_on_m_x_validation_set` (PRADAN real data) — xfail until catalog populated
- `test_far_under_0_05_on_100h_quiet_sun` (PRADAN real data) — xfail until catalog populated

---

## 5. Error Handling & Failure Modes

| Scenario | Handling |
|----------|---------|
| SoLEXS saturated (X10+) | `data_quality_flag == SATURATED` → still fires, confidence lower-bounded at 0.7, notes flag |
| HEL1OS offline | `hel_triggered=False` always → `SOLEXS_ONLY` detection method, confidence ≤ 0.5 |
| Long-duration event > 6hr | Force `FLARE_END` after `long_duration_lock_sec`, unfreeze baseline |
| SAA spike (Kp > 6, no HOPE) | `detection_method='SEP_EVENT'`, NowcastEvent still published |
| Overlapping flare | New changepoint during DECAY → new UUID event_id, `flare_type='OVERLAPPING'` |
| DB write failure | Exception caught and logged; Kafka publish proceeds unblocked |
| PELT warm-up (< 600 frames) | `changepoint` treated as `None`; threshold-only trigger still possible |
| NaN flux | Treated as 0.0; triggers are False; no state transition |

---

## 6. Global Constraints

- All thresholds in `NowcastConfig` — never hardcoded in module files
- `NowcastOrchestrator` not thread-safe — one instance per Kafka partition
- All tests must pass offline: no Kafka, no PRADAN, no PostgreSQL
- SQLite in dev, PostgreSQL in prod — toggled via `db_url`
- Pipeline version: `"2.0.0"`, schema version: `"1.0"`
- Every `NowcastEvent` carries `provenance_hash` linking to the source `PhysicsFrame.frame_id`

---

## 7. Dependencies

```bash
pip install ruptures psycopg2-binary
# sqlite3 is stdlib — no install needed
# ruptures: PELT change-point detection
# psycopg2-binary: PostgreSQL prod driver
```

---

## 8. Acceptance Criteria (PRD §7.11)

```
✓ X6.3 event (2024-02-22) detected within 30 seconds of GOES onset time
✓ HOPE precursor fires ≥ 90 seconds before X6.3 peak
✓ TPR > 0.80 on M+X class events (synthetic injection)
✓ FAR < 0.05 on quiet-Sun synthetic windows
✓ TSS > 0.65 on M+X class events
✓ SoLEXS-only trigger never produces detection_method='DUAL_TRIGGER'
✓ Overlapping flares logged with distinct event_ids
✓ Master Catalog write < 100ms per event
✓ All detections have provenance_hash linking to source PhysicsFrame
✓ X3+ events: confidence=1.0 regardless of input confidence score
```

---

## 9. Build Order

1. `models.py` — NowcastConfig
2. `baseline.py` — RollingP85Baseline
3. `changepoint.py` — ChangepointDetector
4. `goes_class.py` — GOESClassifier
5. `consensus.py` — ConsensusEngine (4-state FSM)
6. `catalog.py` — CatalogService
7. `orchestrator.py` — NowcastOrchestrator (wires all)
8. `test_l3_nowcasting.py` — full test suite + integration tests
