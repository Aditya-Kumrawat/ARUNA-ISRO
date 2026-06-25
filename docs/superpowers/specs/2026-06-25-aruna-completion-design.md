# ARUNA Completion Design Spec
**Date:** 2026-06-25  
**Scope:** All remaining work to bring ARUNA to BAH 2026 demo-ready state  
**Author:** Engineering session  

---

## Guiding Principle

> Every page answers one operational question.

| Page | Question |
|------|----------|
| Mission Control | What is happening right now? |
| Live Physics Stream | Why is it happening? |
| Nowcasting | Has the event begun? |
| Forecast Center | What will happen next, and how certain are we? |
| Threat Intelligence | Who or what is affected? |
| Historical Replay | Has this happened before? |
| Master Flare Catalog | What do we know from past events? |
| Validation & Performance | How trustworthy is the system? |
| Gaganyaan Advisor | What operational action should be taken? |
| Pipeline Health | Can we trust the platform itself? |

---

## Cross-Cutting Architecture

### Mission Event Bus

All data in ARUNA flows as typed events. Every producer (Socket.IO, REST, replay, tests) emits identical events. Every consumer (stores, pages, logs, analytics) receives identical events.

**Canonical event types:**
```
PHYSICS_FRAME
HOPE_DETECTED
FLARE_ONSET
FORECAST_UPDATED
THREAT_UPDATED
NAVIC_IMPACT_UPDATED
PIPELINE_WARNING
PIPELINE_ANOMALY
MODEL_SWITCHED
REPLAY_STARTED
REPLAY_COMPLETE
MISSION_COMPLETE
CONFIG_CHANGED
```

### FrameHeader (universal payload prefix)

Every message — Socket.IO, REST response, Kafka, replay — begins with:
```typescript
interface FrameHeader {
  timestamp: string;     // ISO 8601 UTC
  frame_id: string;      // UUID4
  source: string;        // 'LIVE' | 'REPLAY' | 'SIMULATION'
  version: string;       // pipeline_version
  quality: string;       // 'GOOD' | 'DEGRADED' | 'STALE'
  latency_ms: number;    // ms since frame was produced
}
```

All existing `PhysicsFrame`, `ForecastFrame`, `AlertEvent`, `ThreatOutput` schemas gain this header.

### Mission State Manager

Single authoritative state object consumed by all frontend stores:
```typescript
interface MissionState {
  physics:        PhysicsState;
  forecast:       ForecastState;
  threat:         ThreatState;
  decision:       DecisionState;
  infrastructure: InfrastructureState;
}
```

Every page renders one sub-state. No page constructs its own data model.

### Connection State Machine

```
LIVE        — Socket.IO connected, real frames arriving
REPLAY      — Replay mode active (POST /replay/event)
SIMULATION  — Socket.IO disconnected, synthetic fallback running
OFFLINE     — No API, no replay, degraded
```

Header color coding: green (LIVE), blue (REPLAY), purple (SIMULATION), grey (OFFLINE).

---

## Workstream 1 — Test Fix

**File:** `python/aruna/l6_threat/rules.py:28`  
**Change:** `"HIGH - SHELTER ADVISORY"` → `"HIGH — SHELTER ADVISORY"` (em-dash) on line 28.  
Also fix line 29: `"CRITICAL - SHELTER REQUIRED"` → `"CRITICAL — SHELTER REQUIRED"` for consistency (not yet tested, but same pattern).  
**Do not modify the test.** The test defines the correct output.

---

## Workstream 2 — Frontend Wiring

### Architecture

```
Socket.IO gateway (localhost:3001)
        ↓
  useSocket.ts           (connection state machine, raw event buffer)
        ↓
  Mission Stores         (Zustand)
  ├── PhysicsStore       (physics frames, TNPSV history, spectrograms)
  ├── ForecastStore      (5-class probs, MAPIE intervals, analog results)
  ├── ThreatStore        (NavIC risk, HF blackout, Gaganyaan advisory)
  └── MissionStore       (connection mode, active alert, pipeline health)
        ↓
  Page-level hooks       (e.g. useLivePhysics, useForecast, useThreat)
        ↓
  Pages                  (read from hooks only — zero direct store access)
```

When `connectionMode === 'SIMULATION'`, stores are fed by the existing synthetic generator in `useMissionData.ts` (renamed to `useSyntheticPhysics.ts`).

### useSocket.ts

- Wraps `socket.io-client`
- Tracks `connectionMode: ConnectionMode`
- Transitions: connect → LIVE, disconnect → SIMULATION, replay active → REPLAY
- Emits all incoming events onto the Mission Event Bus

### Stores (Zustand)

**PhysicsStore** — updated every `physics:frame` socket event. Keeps a 60-min rolling buffer of T/EM/DNI/SHEI/HFD history, TNPSV trail (200 pts), current spectrograms.

**ForecastStore** — updated on `forecast:frame`. Holds 5-class probability vector, MAPIE set, top-3 analogs, sub-model weights, horizon.

**ThreatStore** — updated on `nowcast:event` and polled from `/api/v1/threat/current` every 30s. Holds NavIC risk per region, HF blackout, Gaganyaan advisory, satellite safing queue, SpaceWx bulletin.

**MissionStore** — meta-state: connectionMode, activeAlert (from `/alerts/latest`), pipelineHealth (from `/health`), lastLLMBriefing (from `/bulletin/latest`).

### REST layer (useRestQuery.ts)

TanStack Query wrapper, `baseURL = http://localhost:3001/api/v1`. **API routes are unchanged** — the domain grouping below is a frontend query-key organization, not a route rename. Pages import from domain modules rather than calling fetch directly.

| Domain module | Maps to existing API routes |
|---------------|----------------------------|
| `physics` | `/physics/current`, `/physics/history?minutes=N` |
| `forecast` | `/forecast/current`, `/forecast/history` |
| `threat` | `/threat/current`, `/threat/navic` |
| `catalog` | `/catalog/events`, `/catalog/events/:id` |
| `system` | `/health`, `/status/sun`, `/bulletin/latest`, `/alerts/latest` |
| `mission` | `/gaganyaan/window`, `/validation/metrics`, `/replay/status` |

### Per-page wiring

| Page | Store(s) | REST queries |
|------|----------|--------------|
| Mission Control | PhysicsStore, MissionStore | `/system/alerts/latest`, `/system/bulletin/latest` |
| Live Physics Stream | PhysicsStore | none (socket-only) |
| Nowcasting | PhysicsStore, MissionStore | `/system/alerts/latest` |
| Forecast Center | ForecastStore | `/forecast/history`, `/catalog/events/:id` (analogs) |
| Threat Intelligence | ThreatStore | `/threat/current`, `/threat/navic`, `/system/bulletin/latest` |
| Historical Replay | MissionStore | `/mission/replay/status` (poll while replaying) |
| Master Catalog | — | `/catalog/events` (paginated + filtered) |
| Validation/Performance | — | `/mission/validation/metrics` |
| Gaganyaan Advisor | ThreatStore | `/mission/gaganyaan/window?days=7` |
| Pipeline Health | MissionStore | `/system/health` (polled every 5s) |

---

## Workstream 3 — XL-4: Confidence Engine

**Location:** `python/aruna/xl/confidence_engine.py`

Replaces the ad-hoc "MPCI modifier" with a composable confidence pipeline:

```
ForecastFrame (from L4)
      ↓
ConfidenceEngine
  ├── MPCIContributor      (XL-2: ±20%/−15% based on payload coherence)
  ├── PhysicsGateContributor (temperature/DNI consistency check)
  └── HistoricalBaseRateContributor (prior P(M+X) for this GOES subclass)
      ↓
StackedForecastFrame
  ├── adjusted_probabilities  (float[5])
  ├── decision_confidence     (0–1 scalar)
  ├── contributor_breakdown   (dict: name → delta)
  └── header: FrameHeader
```

MPCI is one contributor. Future pluggable contributors: Bayesian calibration, MAPIE ensemble, external model scores.

**Output topic:** `aruna.confidence.frames`  
**Tests:** `python/tests/test_xl_confidence.py` — verify MPCI boost/penalty propagates correctly, verify contributors are logged in breakdown.

---

## Workstream 4 — XL-5: Mission Impact Layer

**Location:** `python/aruna/xl/mission_impact.py`

Maps GOES flux to operational mission impact using the Kintner ionospheric model.

### NavIC Degradation

Kintner model: X-ray flux → TEC perturbation → S4 scintillation index per region.  
Regions: North India, Central India, South India (EIA crest — worst affected), East India, West India, Island Chains.

### Mission Impact Output

```python
@dataclass
class MissionImpactFrame:
    header: FrameHeader
    navic_s4_per_region: Dict[str, float]    # 0–1 S4 scintillation index
    navic_risk_per_region: Dict[str, str]    # NEGLIGIBLE→EXTREME
    satellite_risk: str          # LOW / MEDIUM / HIGH / CRITICAL
    navigation_risk: str
    communications_risk: str
    astronaut_eva_recommendation: str        # PROCEED / CAUTION / DELAY / ABORT
    launch_recommendation: str              # PROCEED / HOLD / SCRUB
    hf_blackout_duration_min: int
    impact_summary: str                     # one-line plain English
```

**Output topic:** `aruna.mission.impact`  
**Tests:** `python/tests/test_xl_mission_impact.py` — verify X6.3 flux gives EXTREME South India S4, X-class → ABORT EVA.

---

## Workstream 5 — XL-6: Mission Integrity Monitor

**Rename from "Self-Correction Agent."** This system **observes, diagnoses, flags, and explains**. It never modifies forecasts or pipeline outputs.

**Location:** `python/aruna/xl/integrity_monitor.py`

### LangGraph Workflow (3 nodes)

```
OBSERVE
  Watches PhysicsFrame stream for out-of-bounds values
  Compares T/EM/DNI against 3σ historical distribution
  Detects instrument dropouts (SDD1→SDD2 switch, CZT saturation)
      ↓
DIAGNOSE
  Checks instrument-level coherence (SoLEXS vs HEL1OS timing)
  Checks L4 forecast vs L2 physics consistency
  Checks LLM briefing numbers against PhysicsFrame values
      ↓
FLAG + EXPLAIN
  Emits PIPELINE_ANOMALY event with:
    anomaly_type: str
    severity: 'INFO' | 'WARNING' | 'CRITICAL'
    affected_stage: str
    explanation: str   (plain English, one sentence)
    recommended_action: str
```

**Invariant:** The monitor never writes to any pipeline topic. It only emits to `aruna.integrity.flags`.  
**Tests:** `python/tests/test_xl_integrity_monitor.py` — inject a physically impossible T=500MK frame, assert CRITICAL flag with correct explanation.

---

## Workstream 6 — E2E Replay Integration Test

**File:** `python/tests/test_integration.py` (extend existing)

### Test sequence

```
1. Run aruna.replay (X6.3, speed=50)
2. Poll aruna_catalog.db every 500ms, timeout 30s
3. Assert catalog event written: goes_class='X', peak_temp ≈ 23.5 MK ±1σ
4. Assert forecast record written: P(X) > 0.5
5. Assert threat record written: gaganyaan_advisory contains 'SHELTER'
6. Assert validation metrics updated: at least 1 TP recorded
7. Assert integrity monitor emits no CRITICAL flags for valid X6.3 data
```

Replay must be deterministic: same FITS fixtures → same catalog row every run.

---

## Workstream 7 — Configuration Page

**Route:** `/config`  
**API:** New endpoints `GET /api/v1/config` and `POST /api/v1/config` in Fastify gateway. Reads/writes `config/config.yaml` via `js-yaml` (add `js-yaml` + `@types/js-yaml` to `api/package.json`).

### Configuration Profiles

```
Research      — loose thresholds, all logging, physics debug output on
Operations    — PRD defaults, M+X alert only, minimal logging
Demo          — X6.3 reference values, synthetic data on, speed=10x
Development   — all features on, strict assertions, verbose logs
```

Selecting a profile writes a preset to config.yaml. Individual fields remain editable after profile selection.

### Editable fields (8 key thresholds)

| Field | Config key | Description |
|-------|-----------|-------------|
| Alert threshold | `forecasting.alert_policy.threshold` | P(M)+P(X) trigger |
| Persistence windows | `forecasting.alert_policy.persistence` | consecutive windows |
| P85 window (min) | `nowcasting.baseline_window_min` | rolling baseline |
| Dual-trigger timeout (s) | `nowcasting.consensus_timeout_sec` | SoLEXS+HEL1OS window |
| HOPE DNI threshold | `physics.hope.dni_threshold` | default 1.5 |
| HOPE TNPSV angle min | `physics.hope.tnpsv_angle_min` | default 60° |
| HOPE TNPSV angle max | `physics.hope.tnpsv_angle_max` | default 120° |
| Dead-time SDD1 sat. | `physics.deadtime.sdd1_saturation` | default 5e4 cts/s |

---

## Workstream 8 — Logs / Mission Timeline

**Route:** `/logs`

### Two-panel layout

**Top panel — Mission Timeline** (primary view)  
Event-based story view. Each entry is a meaningful mission event:

```
18:24:12  HOPE DETECTED        T rising 3.2 MK/s, DNI=1.62, EM flat
18:24:58  FORECAST UPDATED     P(M+X) = 78%, lead time 22 min
18:25:31  THREAT ESCALATED     NavIC South India → HIGH, Gaganyaan → CAUTION
18:26:44  NAVIC WARNING        S4 index 0.72 — degraded positioning likely
18:28:09  LLM BRIEFING         Generated in 12.3s, hallucination check PASSED
18:29:02  FLARE ONSET          X1.2 confirmed, dual-trigger consensus
```

Powered by the Mission Event Bus. Filter by event type.

**Bottom panel — Raw Log Stream** (secondary, collapsed by default)  
Structured log stream via Socket.IO `pipeline:log` event.  
Columns: timestamp, level (DEBUG/INFO/WARN/ERROR), stage (L1–L7/XL), message.  
Filter by level and stage.

### New API endpoint

`GET /api/v1/logs/timeline?limit=100` — returns last N mission timeline events from SQLite.  
Socket.IO event `mission:event` — pushed on every bus event for real-time timeline updates.

---

## Acceptance Criteria Summary

| Workstream | Done when |
|------------|-----------|
| Test fix | `pytest python/tests/test_l6_threat.py` — 0 failures |
| Frontend wiring | All 10 pages show real data when API is running; show synthetic data when API is offline |
| XL-4 Confidence Engine | `test_xl_confidence.py` passes; MPCI boost visible in ForecastFrame |
| XL-5 Mission Impact | `test_xl_mission_impact.py` passes; X6.3 flux → EXTREME South India, EVA DELAY |
| XL-6 Integrity Monitor | `test_xl_integrity_monitor.py` passes; T=500MK injection → CRITICAL flag |
| E2E Replay | `test_integration.py::test_x63_replay_end_to_end` passes, 0 CRITICAL flags |
| Config page | Profile switching changes config.yaml; individual field edits persist |
| Logs/Timeline | Mission timeline shows real events when connected; Socket.IO `mission:event` streams live |
