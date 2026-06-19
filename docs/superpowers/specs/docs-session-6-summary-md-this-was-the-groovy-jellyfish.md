# Session 7 Plan — L5 Analogs + L6 Threat + L7 LLM Briefing

## Context

Sessions 1–6 completed L0–L4 and XL-1/2/3. As of Session 6: 230 tests passing, 37 xfailed, 0 failed.
The remaining 20 xfails in scope are:
- L5 Analogs: 6 xfail (`test_l5_analogs.py`)
- L6 Threat Intelligence: 7 xfail (`test_l6_threat.py`)
- L7 LLM Briefing: 7 xfail (`test_l7_llm.py`)

These three layers complete the downstream pipeline: nowcast → analogs → threat → briefing. Together they are the BAH 2026 jury demo centrepiece. XL-1 wiring is deliberately excluded from this session to keep blast radius tight.

**Build order: L6 → L5 → L7** (dependency order: L6 and L5 are independent; L7 consumes both).

---

## Design

### Architecture

Three new packages, each with a pure orchestrator + separate Kafka adapter:

```
python/aruna/
  l6_threat/
    __init__.py       ← ThreatOrchestrator (façade)
    models.py         ← ThreatOutput, SafingEntry dataclasses
    rules.py          ← all lookup tables as typed constants
    navic.py          ← enriched NavIC risk computation
    bulletin.py       ← SpaceWx bulletin formatter
    kafka.py          ← Kafka consumer/producer wrapper

  l5_analogs/
    __init__.py       ← AnalogOrchestrator (façade)
    models.py         ← AnalogMatch, NO_ANALOG sentinel
    embedder.py       ← FlareEmbedder protocol + PhysicsFeatureEmbedder
    index.py          ← FAISS IndexFlatL2 builder/searcher
    dtw.py            ← DTW refinement (top-10 → top-3)
    kafka.py          ← Kafka consumer/producer wrapper

  l7_llm/
    __init__.py       ← BriefingOrchestrator (façade)
    models.py         ← AlertBriefing, BriefingContext dataclasses
    client.py         ← BriefingLLM protocol + AzureOpenAIClient + MockLLMClient + OllamaClient
    guard.py          ← hallucination guard (schema + regex)
    template.py       ← deterministic fallback formatter
    kafka.py          ← Kafka consumer/producer wrapper
```

### Kafka Topics

| Producer | Topic |
|----------|-------|
| L6 ThreatOrchestrator | `aruna.threat.output` |
| L5 AnalogOrchestrator | `aruna.analogs.results` |
| L7 BriefingOrchestrator | `aruna.alerts.briefings` |

L7 consumes: `aruna.nowcast.events`, `aruna.forecast.frames`, `aruna.threat.output`, `aruna.analogs.results`.

All orchestrators expose `process(...)` for direct synchronous test calls without Kafka.

### Dataclasses

```python
# l5_analogs/models.py
@dataclass
class AnalogMatch:
    event_id: str
    goes_class: str
    dtw_distance: float
    similarity: float         # 1 / (1 + dtw_distance)
    feature_vector: np.ndarray

NO_ANALOG = AnalogMatch(event_id="NO_ANALOG", goes_class="UNKNOWN",
                        dtw_distance=float("inf"), similarity=0.0,
                        feature_vector=np.zeros(17))

# l6_threat/models.py
@dataclass
class SafingEntry:
    name: str
    orbit: str
    vulnerability: dict[str, float]   # {"X": 1.0, "M5": 0.85}

@dataclass
class ThreatOutput:
    navic_risk: dict[str, str]        # region → level string
    navic_prob: dict[str, float]      # region → 0.0–1.0
    hf_risk: str
    hf_duration_min: int
    radiation_risk: str
    gaganyaan_advisory: str
    safing_queue: list[SafingEntry]   # ranked by vulnerability for this goes_class
    dos: list[str]
    donts: list[str]
    bulletin_text: str
    source_event_id: str

# l7_llm/models.py
@dataclass
class BriefingContext:
    nowcast: NowcastEvent
    forecast: ForecastFrame
    threat: ThreatOutput
    analogs: list[AnalogMatch]        # top-3, or [NO_ANALOG]

@dataclass
class AlertBriefing:
    briefing_text: str
    briefing_source: str              # "azure_gpt4.1-mini" | "azure_gpt4.1" | "ollama" | "deterministic"
    hallucination_violations: list[str]
    context_used: BriefingContext
    created_at: str                   # ISO 8601 UTC
```

---

## L6 — Threat Intelligence

### Files to create

**`rules.py`** — all lookup tables as module-level typed constants:
- `NAVIC_BASE_RISK: dict[str, tuple[str, float]]` — maps GOES band → (level, probability)
  - B→NEGLIGIBLE/0.02, C→LOW/0.15, M1→MODERATE/0.30, M5→HIGH/0.55, X1→VERY_HIGH/0.67, X5→EXTREME/0.90
- `HF_BLACKOUT: dict[str, tuple[str, int]]` — maps band → (label, duration_min)
  - B→NONE/0, C→MINOR/15, M1→MODERATE/45, M5→STRONG/120, X1→SEVERE/240, X5→COMPLETE/480
- `GAGANYAAN: dict[str, str]` — maps band → advisory string
- `SAFING_REGISTRY: list[SafingEntry]` — 6 ISRO satellites, typed, ranked by default priority
  - Chandrayaan-3 (LUNAR_POLAR), Aditya-L1 (L1_HALO), RISAT-2BR1 (LEO_SAR), GSAT-30 (GEO), IRNSS-1I (MEO_NAVIC), EOS-06 (LEO_OCEAN)
- `EIA_AMPLIFIED_REGIONS: frozenset` = `{"south_india", "island_chains"}`

**`navic.py`** — `compute_navic_risk(event: NowcastEvent) -> dict[str, tuple[str, float]]`:
1. `parse_goes_band(goes_class)` → explicit banding: M1–M4→"M1", M5–M9→"M5", X1–X4→"X1", X5+→"X5", B/C pass through directly
2. Base lookup from `NAVIC_BASE_RISK`
3. Modifiers (applied in order, each ±1 level capped at EXTREME/NEGLIGIBLE):
   - `confidence > 0.90` → +1 level
   - `confidence < 0.50` → −1 level
   - `flare_phase == "IMPULSIVE"` → +1 for EIA_AMPLIFIED_REGIONS only
   - `peak_flux > class_midpoint × 3` → +1 level
4. Safety clamp (final, after all modifiers): X-class events → floor at VERY_HIGH, never downgrade below

**`bulletin.py`** — `format_bulletin(event, threat) -> str` using SWMC template structure.

**`__init__.py`** — `ThreatOrchestrator.process(event: NowcastEvent) -> ThreatOutput`:
- Calls `compute_navic_risk`, HF lookup, Gaganyaan lookup, safing sort, dos/donts generation, bulletin format
- Pure Python, no I/O

**`kafka.py`** — wraps orchestrator, consumes `aruna.nowcast.events`, publishes `aruna.threat.output`.

### Test targets (7 xfail → pass)
- `test_threat_flags_within_2_seconds` — pure lookup, trivially < 2s
- `test_navic_risk_matches_kintner_model_within_1_level` — calibrated lookup satisfies ±1 tolerance
- `test_chandrayaan3_is_priority_1_for_x_class` — first entry in sorted safing_queue for X
- `test_spacewx_bulletin_validates_against_swmc_template` — bulletin has required sections
- `test_dos_donts_generated_for_m5_plus_and_x_class` — non-empty dos/donts for M5+
- `test_south_india_highest_eia_amplification` — south_india risk ≥ north_india for same event
- `test_x_class_impact_never_understated` — safety clamp: X-class always ≥ VERY_HIGH

---

## L5 — Historical Analog Retrieval

### Files to create

**`embedder.py`**:
```python
class FlareEmbedder(Protocol):
    def encode(self, event: NowcastEvent) -> np.ndarray: ...

class PhysicsFeatureEmbedder:
    FEATURES = [
        # Thermal (4)
        "peak_T_MK", "peak_EM_cm3", "dT_dt_peak", "dEM_dt_peak",
        # Nonthermal (3)
        "peak_SHEI", "peak_dSHEI_dt", "peak_HEL_flux",
        # Dynamics (7)
        "peak_DNI", "peak_TNPSV_mag", "peak_TNPSV_angle",
        "hope_lead_time_s", "flare_duration_s", "rise_time_s", "decay_time_s",
        # Classification (3)
        "goes_class_numeric",   # B=1, C=2, M=3, X=4
        "goes_subclass",        # e.g. 3.8
        "confidence",
    ]  # 17 total
    # encode(): extract → np.array → StandardScaler.transform → L2-normalize
```

Scaler is fitted at index-build time and serialized alongside the FAISS index (pickle). Every query vector goes through the same scaler + L2-normalize.

**`index.py`**:
- `build(events: list[NowcastEvent], embedder: FlareEmbedder) -> AnalogIndex` — encodes all events, fits scaler, builds `faiss.IndexFlatL2(17)`, computes NO_ANALOG threshold = mean pairwise L2 + 3σ
- `search(query: np.ndarray, k: int = 10) -> list[tuple[str, float]]` — returns (event_id, l2_dist) top-10
- `AnalogIndex` persisted to disk as `{faiss_index, scaler, event_ids, no_analog_threshold}` via pickle
- Auto-rebuild triggered when `CatalogService` signals a new entry (listener pattern)
- Cold-start: if catalog < 3 events → return `[NO_ANALOG]` immediately

**`dtw.py`**:
- `refine(candidates: list[tuple[str, float]], query_event: NowcastEvent, catalog: CatalogService) -> list[AnalogMatch]`
- Fetches timeseries for top-10 candidates from catalog
- Runs `dtaidistance.dtw_fast` on the feature timeseries
- Returns sorted top-3 as `AnalogMatch` objects with `similarity = 1 / (1 + dtw_distance)`

**`__init__.py`** — `AnalogOrchestrator.process(event: NowcastEvent) -> list[AnalogMatch]`

**`kafka.py`** — consumes `aruna.nowcast.events`, publishes `aruna.analogs.results`.

### Reused utilities
- `CatalogService` from `python/aruna/l3_nowcasting/catalog.py` — `fetch_event(event_id)`, upsert
- `PhysicsFrame` fields from `python/aruna/l2_physics/models.py` for feature extraction

### Test targets (6 xfail → pass)
- `test_retrieval_precision_over_0_75` — validated against X6.3 fixture
- `test_faiss_search_under_100ms_for_10k_events` — IndexFlatL2 at 17-dim is microseconds
- `test_no_analog_fires_when_dtw_over_3_sigma` — inject event far from library mean
- `test_x63_retrieves_at_least_one_x_class_analog` — X6.3 fixture retrieves X-class neighbor
- `test_faiss_index_auto_rebuilds_after_new_catalog_entry` — listener triggers rebuild
- `test_cold_start_empty_catalog_no_crash` — returns `[NO_ANALOG]`, no exception

---

## L7 — LLM Briefing

### Files to create

**`client.py`**:
```python
class BriefingLLM(Protocol):
    def generate(self, system_prompt: str, context: dict) -> str: ...

class AzureOpenAIClient:
    # AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY from env
    # model from config.yaml: gpt-4.1-mini (default) or gpt-4.1 (premium)
    # 3 retries with exponential backoff → raises BriefingLLMError on exhaustion

class MockLLMClient:
    # Returns canned valid JSON briefing_text — no network, used in all tests

class OllamaClient:
    # Optional offline backend, same BriefingLLM interface
```

Config (`config.yaml`):
```yaml
llm:
  provider: azure         # azure | ollama | mock
  model: gpt-4.1-mini     # or gpt-4.1
  fallback: deterministic
  max_tokens: 400
  temperature: 0.1
```

**`guard.py`** — `validate_briefing(text: str, ctx: BriefingContext) -> tuple[bool, list[str]]`:
1. Schema check: JSON parses, `briefing_text` key present, required fields appear verbatim (goes_class, navic_risk string)
2. Numeric constraint: extract all numeric tokens from briefing_text; each must appear in the allowed set `{lead_time_min, confidence×100, goes_subclass, hf_duration_min, analog_similarity×100}`; any extra number → violation
3. Returns `(True, [])` if clean; `(False, [violation_strings])` if not

**`template.py`** — `format_deterministic(ctx: BriefingContext) -> str`:
```
ARUNA SOLAR FLARE ALERT — {created_at} UTC
{goes_class} | Confidence: {confidence_pct}% | Lead Time: {lead_time_min} min

NavIC: {navic_risk} ({affected_regions})
HF Comms: {hf_risk} blackout ~{hf_duration_min} min | Radiation: {radiation_risk}

DO: {dos_formatted}
AVOID: {donts_formatted}

Most similar past event: {analog_goes_class} ({similarity_pct}% match)
```

**`__init__.py`** — `BriefingOrchestrator.process(ctx: BriefingContext) -> AlertBriefing`:
1. Assemble context dict from all `BriefingContext` fields
2. Call `BriefingLLM.generate(SYSTEM_PROMPT, context_dict)`
3. Run `validate_briefing()`; on any violation → use `format_deterministic(ctx)`, set `briefing_source="deterministic"`
4. Return `AlertBriefing`

LLM prompt rule: context dict contains **only validated upstream outputs** — LLM narrates, never computes physics.

**`kafka.py`** — consumes `aruna.nowcast.events` + `aruna.forecast.frames` + `aruna.threat.output` + `aruna.analogs.results`, assembles `BriefingContext`, publishes `aruna.alerts.briefings`.

### Test targets (7 xfail → pass)
- `test_hallucination_guard_passes_on_all_test_alerts` — MockLLMClient returns valid canned response
- `test_briefing_generated_within_15_seconds` — mocked, trivially fast
- `test_fallback_activates_when_guard_fails` — inject a bad LLM response, assert `briefing_source=="deterministic"`
- `test_every_briefing_has_required_fields` — check briefing_text contains goes_class, confidence, lead time, navic_risk
- `test_no_novel_numbers_in_briefing` — inject canned response with extra number, assert guard catches it
- `test_llm_runs_fully_offline` — MockLLMClient, no network (monkeypatch socket)
- `test_deterministic_fallback_valid_without_ollama` — call template.py directly, assert output non-empty and contains required fields

---

## Files Modified

- `python/aruna/l6_threat/__init__.py` — replace stub
- `python/aruna/l5_analogs/__init__.py` — replace stub
- `python/aruna/l7_llm/__init__.py` — replace stub
- `python/tests/test_l5_analogs.py` — replace xfail stubs with real assertions
- `python/tests/test_l6_threat.py` — replace xfail stubs with real assertions
- `python/tests/test_l7_llm.py` — replace xfail stubs with real assertions
- `python/tests/conftest.py` — add fixtures: `sample_nowcast_event`, `sample_threat_output`, `sample_analog_matches`, `mock_llm_client`
- `config/config.yaml` (or `config.example.yaml`) — add `llm:` section

## Files Created (new)

```
python/aruna/l6_threat/models.py
python/aruna/l6_threat/rules.py
python/aruna/l6_threat/navic.py
python/aruna/l6_threat/bulletin.py
python/aruna/l6_threat/kafka.py

python/aruna/l5_analogs/models.py
python/aruna/l5_analogs/embedder.py
python/aruna/l5_analogs/index.py
python/aruna/l5_analogs/dtw.py
python/aruna/l5_analogs/kafka.py

python/aruna/l7_llm/models.py
python/aruna/l7_llm/client.py
python/aruna/l7_llm/guard.py
python/aruna/l7_llm/template.py
python/aruna/l7_llm/kafka.py
```

Also: `docs/superpowers/specs/2026-06-19-session7-l5-l6-l7-design.md` (written and committed at start of implementation).

---

## Verification

```bash
# Run the three new test suites — all 20 xfails must become passing
pytest python/tests/test_l6_threat.py -v        # 7 pass
pytest python/tests/test_l5_analogs.py -v       # 6 pass
pytest python/tests/test_l7_llm.py -v           # 7 pass

# Full suite — must stay 0 failures, prior 230 still passing
pytest python/ -v

# Quick smoke: end-to-end synchronous call chain
python -c "
from aruna.l3_nowcasting.models import NowcastEvent
from aruna.l6_threat import ThreatOrchestrator
from aruna.l5_analogs import AnalogOrchestrator
from aruna.l7_llm import BriefingOrchestrator
from aruna.l7_llm.models import BriefingContext
from aruna.l7_llm.client import MockLLMClient
from aruna.l4_forecasting.models import ForecastFrame

event = NowcastEvent(...)          # minimal X-class fixture
threat = ThreatOrchestrator().process(event)
analogs = AnalogOrchestrator().process(event)
ctx = BriefingContext(nowcast=event, forecast=ForecastFrame(...), threat=threat, analogs=analogs)
briefing = BriefingOrchestrator(llm=MockLLMClient()).process(ctx)
print(briefing.briefing_text[:200])
"
```

Target: **250 passed, 17 xfailed, 0 failed** (the 17 remaining xfails are from earlier session xfail registry, not L5/L6/L7).
