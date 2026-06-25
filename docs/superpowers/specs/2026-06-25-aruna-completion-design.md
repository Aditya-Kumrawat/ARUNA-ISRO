# ARUNA Completion Design Spec v2
**Date:** 2026-06-25  
**Scope:** All remaining work to bring ARUNA to BAH 2026 demo-ready state  
**Status:** Approved with refinements — proceed to implementation plan  

---

## Guiding Principle

> Every page answers one operational question. Every event drives one decision.

ARUNA is a **Solar Threat Intelligence Platform**, not a dashboard collection. The pipeline flows:

```
Physics → Forecast → Mission Impact → Decision Engine → Operator
```

Operators don't consume threats. They consume decisions.

---

## Brand Navigation

The sidebar groups pages under the brand philosophy:

| Group | Pages | Brand meaning |
|-------|-------|---------------|
| **OBSERVE** | Mission Control | Situational awareness |
| **UNDERSTAND** | Live Physics Stream | The why behind the signal |
| **FORESEE** | Nowcasting, Forecast Center, Explainability Center | Prediction with trust |
| **PROTECT** | Threat Intelligence, Gaganyaan Advisor | Operational safeguarding |
| **LEARN** | Historical Replay, Master Catalog, Validation & Performance | Knowledge from history |
| **OPERATE** | Pipeline Health, Configuration, Mission Chronicle | Platform governance |

---

## Mission Banner (global, top of every page)

A persistent strip across the top of every page below the header:

```
[ ARUNA WATCH ]  HOPE DETECTED   Forecast Confidence 82%   Mission State: FORESEE   Next Decision: SATELLITE SAFING IN 18 MIN
```

Fields: current watch level (QUIET/WATCH/WARNING/ALERT/EMERGENCY), active event label, forecast confidence, connection mode badge, next recommended action with countdown. Populated from MissionStore. Color tracks connection mode (green LIVE / blue REPLAY / purple SIMULATION / grey OFFLINE).

---

## Cross-Cutting Architecture

### Mission Event Bus

All data in ARUNA flows as typed events. Every producer (Socket.IO, REST, replay, tests) emits identical events. Every consumer (stores, pages, Mission Chronicle, analytics) receives identical events.

**Canonical event types:**
```
PHYSICS_FRAME
HOPE_DETECTED
FLARE_ONSET
FORECAST_UPDATED
CONFIDENCE_UPDATED
THREAT_UPDATED
NAVIC_IMPACT_UPDATED
DECISION_UPDATED
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
  decision:       DecisionState;       // richest state — see below
  infrastructure: InfrastructureState;
}
```

### DecisionState (richest object in the system)

```typescript
interface DecisionState {
  recommendation: string;           // 'PROCEED' | 'DELAY' | 'SAFE_MODE' | 'INCREASE_MONITORING' | 'NOTIFY_GROUND' | 'ABORT_EVA'
  confidence: number;               // 0–1
  actions: OperationalAction[];     // ranked list of recommended actions
  affectedAssets: AssetImpact[];    // satellites, NavIC, Gaganyaan, HF comms
  riskTimeline: RiskTimelinePoint[];// how risk evolves over next 60 min
  missionBrief: string;            // one-sentence plain-English summary
  reasoning: ExplainabilityTrace;  // WHY — physics chain that led to this decision
  validUntil: string;              // ISO UTC — when to recompute
}
```

**Every page that shows operational guidance consumes DecisionState** — Mission Control (next action), Threat Intelligence (asset impacts), Gaganyaan Advisor (EVA/launch recommendation). Pages don't independently compute decisions.

### Connection State Machine

```
LIVE        — Socket.IO connected, real frames arriving          (header: green)
REPLAY      — Replay mode active (POST /replay/event)            (header: blue)
SIMULATION  — Socket.IO disconnected, synthetic fallback active  (header: purple)
OFFLINE     — No API, no replay, degraded                        (header: grey)
```

### Pipeline Layer Model

```
L0  Source
L1  Ingestion
L2  Physics
L3  Nowcasting
L4  Forecasting
L5  Analogs
L6  Threat
L7  LLM Briefing
XL2 MPCI Coherence
XL3 Physics Tokens
XL4 Confidence Engine  ← new
XL5 Mission Impact     ← new
XL6 Integrity Monitor  ← new
DX  Decision Engine    ← new (Decision eXecutive)
```

---

## Workstream 1 — Test Fix

**File:** `python/aruna/l6_threat/rules.py`  
**Line 28:** `"HIGH - SHELTER ADVISORY"` → `"HIGH — SHELTER ADVISORY"` (em-dash)  
**Line 29:** `"CRITICAL - SHELTER REQUIRED"` → `"CRITICAL — SHELTER REQUIRED"` (consistency)  
**Do not modify the test.** The test defines the correct output.

---

## Workstream 2 — Frontend Wiring

### Store Architecture

```
Socket.IO (localhost:3001)
        ↓
  useSocket.ts           (connection state machine, raw event buffer)
        ↓
  Mission Stores (Zustand)
  ├── PhysicsStore       (physics frames, TNPSV history, spectrograms)
  ├── ForecastStore      (5-class probs, MAPIE intervals, analog results)
  ├── ThreatStore        (NavIC risk, HF blackout, Gaganyaan advisory)
  ├── DecisionStore      (full DecisionState — richest store)
  └── MissionStore       (connection mode, active alert, pipeline health)
        ↓
  Page-level hooks       (useLivePhysics, useForecast, useThreat, useDecision)
        ↓
  Pages + Mission Banner (read from hooks only — zero direct store access)
```

When `connectionMode === 'SIMULATION'`, stores are fed by the existing synthetic generator (renamed `useSyntheticPhysics.ts`). All 10 existing pages keep the same component interface — no page rewrites needed, only hook swap.

### useSocket.ts

Wraps `socket.io-client`. Tracks `connectionMode`. Transitions: connect → LIVE, disconnect → SIMULATION, replay active → REPLAY. Emits all incoming events onto the Mission Event Bus.

### REST Layer (useRestQuery.ts)

TanStack Query wrapper, `baseURL = http://localhost:3001/api/v1`. **API routes are unchanged** — domain grouping is a frontend query-key organization, not route renaming.

| Domain module | Maps to existing API routes |
|---------------|----------------------------|
| `physics` | `/physics/current`, `/physics/history?minutes=N` |
| `forecast` | `/forecast/current`, `/forecast/history` |
| `threat` | `/threat/current`, `/threat/navic` |
| `catalog` | `/catalog/events`, `/catalog/events/:id` |
| `system` | `/health`, `/status/sun`, `/bulletin/latest`, `/alerts/latest` |
| `mission` | `/gaganyaan/window`, `/validation/metrics`, `/replay/status`, `/config` |

### Per-page wiring

| Page | Store(s) | REST queries |
|------|----------|--------------|
| Mission Control | PhysicsStore, MissionStore, DecisionStore | `/system/alerts/latest`, `/system/bulletin/latest` |
| Live Physics Stream | PhysicsStore | none (socket-only) |
| Nowcasting | PhysicsStore, MissionStore | `/system/alerts/latest` |
| Forecast Center | ForecastStore | `/forecast/history`, `/catalog/events/:id` |
| Explainability Center | ForecastStore, DecisionStore | `/forecast/current`, `/system/alerts/latest` |
| Threat Intelligence | ThreatStore, DecisionStore | `/threat/current`, `/threat/navic`, `/system/bulletin/latest` |
| Historical Replay | MissionStore | `/mission/replay/status` |
| Master Catalog | — | `/catalog/events` (paginated + filtered) |
| Validation/Performance | — | `/mission/validation/metrics` |
| Gaganyaan Advisor | ThreatStore, DecisionStore | `/mission/gaganyaan/window?days=7` |
| Pipeline Health | MissionStore | `/system/health` (polled 5s) |
| Configuration | — | `/mission/config` |
| Mission Chronicle | MissionStore | `/mission/events/timeline` |

---

## Workstream 3 — Decision Engine (DX layer)

**Location:** `python/aruna/decision/engine.py`

New pipeline layer sitting above XL5. Consumes `MissionImpactFrame` and `StackedForecastFrame`, produces `DecisionFrame`.

### Decision Logic

```python
@dataclass
class DecisionFrame:
    header: FrameHeader
    recommendation: str          # PROCEED | DELAY | SAFE_MODE | INCREASE_MONITORING | NOTIFY_GROUND | ABORT_EVA
    confidence: float            # 0–1
    actions: List[OperationalAction]  # ranked
    affected_assets: List[AssetImpact]
    risk_timeline: List[RiskTimelinePoint]  # next 60 min
    mission_brief: str           # one sentence
    reasoning: ExplainabilityTrace
    valid_until: datetime
```

### Decision Rules (explicit, auditable, not ML)

| Condition | Recommendation |
|-----------|---------------|
| P(M+X) < 0.10, no HOPE | PROCEED |
| P(M+X) 0.10–0.30, HOPE inactive | INCREASE_MONITORING |
| P(M+X) 0.30–0.60, HOPE active | NOTIFY_GROUND |
| P(M+X) > 0.60, HOPE active | SAFE_MODE |
| FLARE_ONSET confirmed, M-class | DELAY (EVA 30 min) |
| FLARE_ONSET confirmed, X-class | ABORT_EVA |

### ExplainabilityTrace

The reasoning chain that produced the decision:
```python
@dataclass
class ExplainabilityTrace:
    physics_evidence: List[str]   # e.g. "T rising 3.2 MK/s for 8s"
    trigger_chain: List[str]      # e.g. "HOPE → DUAL_TRIGGER → FLARE_ONSET"
    confidence_contributors: Dict[str, float]  # from XL4
    analog_match: str             # "87% similar to X6.3 (2024-02-22)"
    physics_consistency: str      # "SoLEXS + HEL1OS coherent (MPCI=0.91)"
    mission_impact_summary: str   # from XL5
```

**Output topic:** `aruna.decision.frames`  
**Tests:** `python/tests/test_decision_engine.py`

---

## Workstream 4 — XL-4: Confidence Engine

**Location:** `python/aruna/xl/confidence_engine.py`

```
ForecastFrame (L4)
      ↓
ConfidenceEngine
  ├── MPCIContributor      (XL2: ±20%/−15%)
  ├── PhysicsGateContributor (T/EM/DNI internal consistency)
  └── HistoricalBaseRateContributor (prior P(M+X) for GOES subclass)
      ↓
StackedForecastFrame
  ├── adjusted_probabilities  float[5]
  ├── decision_confidence     0–1
  ├── contributor_breakdown   dict: name → delta
  └── header: FrameHeader
```

Pluggable — future contributors (Bayesian, MAPIE ensemble) added without changing consumers.

**Output topic:** `aruna.confidence.frames`  
**Tests:** `python/tests/test_xl_confidence.py`

---

## Workstream 5 — XL-5: Mission Impact Layer

**Location:** `python/aruna/xl/mission_impact.py`

Maps GOES flux → TEC perturbation (Kintner model) → operational mission impact.

```python
@dataclass
class MissionImpactFrame:
    header: FrameHeader
    navic_s4_per_region: Dict[str, float]     # S4 scintillation 0–1
    navic_risk_per_region: Dict[str, str]     # NEGLIGIBLE→EXTREME
    satellite_risk: str          # LOW / MEDIUM / HIGH / CRITICAL
    navigation_risk: str
    communications_risk: str
    astronaut_eva_recommendation: str         # PROCEED / CAUTION / DELAY / ABORT
    launch_recommendation: str               # PROCEED / HOLD / SCRUB
    hf_blackout_duration_min: int
    impact_summary: str                      # one-line plain English
```

South India + Island Chains are worst-affected (EIA equatorial ionization anomaly crest zone).

**Output topic:** `aruna.mission.impact`  
**Tests:** `python/tests/test_xl_mission_impact.py` — X6.3 flux → EXTREME South India, EVA DELAY

---

## Workstream 6 — XL-6: Mission Integrity Monitor

**Location:** `python/aruna/xl/integrity_monitor.py`

Observes, diagnoses, flags, and explains. **Never modifies pipeline outputs.**

### LangGraph Workflow

```
OBSERVE  → watch PhysicsFrame for out-of-bounds, instrument dropouts
DIAGNOSE → check SoLEXS/HEL1OS timing coherence; L4 vs L2 consistency; LLM numbers vs physics
FLAG     → emit PIPELINE_ANOMALY: {anomaly_type, severity, affected_stage, explanation, recommended_action}
```

`PIPELINE_ANOMALY` goes to `aruna.integrity.flags` only — never to forecast or threat topics.

**Tests:** `python/tests/test_xl_integrity_monitor.py` — T=500MK injection → CRITICAL flag with explanation

---

## Workstream 7 — E2E Replay Integration Test

**File:** `python/tests/test_integration.py`

```
1. Run aruna.replay X6.3 (speed=50, deterministic seed)
2. Poll aruna_catalog.db, timeout 30s
3. Assert: goes_class='X', peak_temp ≈ 23.5 MK ±1σ
4. Assert: forecast record — P(X) > 0.5
5. Assert: threat record — gaganyaan_advisory contains '—'
6. Assert: decision record — recommendation in {'ABORT_EVA', 'NOTIFY_GROUND', 'SAFE_MODE'}
7. Assert: validation metrics — ≥1 TP recorded
8. Assert: integrity monitor — 0 CRITICAL flags for valid X6.3 data
```

---

## Workstream 8 — Configuration Page

**Route:** `/config`  
**API:** `GET /api/v1/config` and `POST /api/v1/config`. Reads/writes `config/config.yaml` via `js-yaml` (add `js-yaml` + `@types/js-yaml` to `api/package.json`).

### Configuration Profiles

| Profile | Description |
|---------|-------------|
| Research | Loose thresholds, all logging, physics debug on |
| Operations | PRD defaults, M+X alert only, minimal logging |
| Demo | X6.3 reference values, synthetic data, speed=10x |
| Development | All features on, strict assertions, verbose logs |
| Competition | Locks replay speed, thresholds, demo data, and UI — frozen for hackathon judging |

Selecting a profile writes a preset to config.yaml. Individual fields remain editable after profile selection (Competition profile locks edits).

### Editable thresholds (8 fields)

| Field | Config key |
|-------|-----------|
| Alert threshold | `forecasting.alert_policy.threshold` |
| Persistence windows | `forecasting.alert_policy.persistence` |
| P85 window (min) | `nowcasting.baseline_window_min` |
| Dual-trigger timeout (s) | `nowcasting.consensus_timeout_sec` |
| HOPE DNI threshold | `physics.hope.dni_threshold` |
| HOPE TNPSV angle min/max | `physics.hope.tnpsv_angle_min/max` |
| SDD1 saturation | `physics.deadtime.sdd1_saturation` |

---

## Workstream 9 — Mission Chronicle (formerly Mission Timeline)

**Route:** `/chronicle`

### Two-panel layout

**Top panel — Mission Chronicle** (aerospace logbook view)

```
18:24:12  ◉ HOPE DETECTED        T rising 3.2 MK/s · DNI=1.62 · EM flat
18:24:58  ◎ FORECAST UPDATED     P(M+X) = 78% · lead time 22 min · TCN dominant
18:25:31  ▲ THREAT ESCALATED     NavIC South India → HIGH · Gaganyaan → CAUTION
18:26:09  ⬡ DECISION             SAFE MODE recommended · confidence 74%
18:26:44  ◉ NAVIC WARNING        S4 = 0.72 · degraded positioning likely
18:28:09  ✓ LLM BRIEFING         Generated 12.3s · hallucination check PASSED
18:29:02  ◉ FLARE ONSET          X1.2 confirmed · dual-trigger consensus
```

Each event is clickable — opens an event detail panel with full ExplainabilityTrace.

**Bottom panel — Raw Log Stream** (collapsed by default)

Structured logs via Socket.IO `pipeline:log`. Columns: timestamp, level, stage, message. Filter by level (DEBUG/INFO/WARN/ERROR) and stage (L1–L7, XL, DX).

### New API

`GET /api/v1/events/timeline?limit=100` — last N mission chronicle events from SQLite.  
Socket.IO `mission:event` — pushed on every bus event for real-time updates.

---

## Workstream 10 — Explainability Center

**Route:** `/explain`  
**Brand group:** FORESEE  
**Operational question:** *Why is ARUNA forecasting this, and should I trust it?*

### Layout

**Left panel — Current Decision**
- Recommendation badge (large): SAFE MODE
- Confidence: 74%
- Valid until: 18:47:12 UTC

**Center panel — Reasoning Chain** (the trust engine)
```
PHYSICS EVIDENCE
  ✓ Temperature rising 3.2 MK/s for 8 consecutive seconds
  ✓ Emission Measure stable (HOPE precursor signature)
  ✓ DNI = 1.62 (crossed threshold 1.5)
  ✓ TNPSV angle 88° (within HOPE quadrant 60°–120°)

TRIGGER CHAIN
  HOPE DETECTED → DUAL-TRIGGER → FLARE ONSET → DECISION

CONFIDENCE CONTRIBUTORS
  TCN model:          +45%
  MPCI (coherence):   +18%  (MPCI = 0.91 — SUIT + SoLEXS + HEL1OS agree)
  Historical base:    +11%
  Physics gate:        −0%  (no penalty — T and EM consistent)
  ─────────────────────────
  Decision confidence: 74%

HISTORICAL ANALOG
  87% similar to X6.3 (2024-02-22 18:46 UTC)
  That event peaked at X6.3 · T = 23.5 MK · 22 min lead time

PHYSICS CONSISTENCY
  SoLEXS + HEL1OS coherent (MPCI = 0.91)
  No instrument dropout, no SAA contamination
```

**Right panel — Mission Impact**
```
SATELLITE RISK     HIGH
NAVIGATION (NavIC) EXTREME (South India)
COMMUNICATIONS     STRONG BLACKOUT
EVA RECOMMENDATION DELAY
LAUNCH             HOLD
```

The Explainability Center is the **trust page** — it transforms a number (82%) into a narrative a mission director can defend to stakeholders.

---

## Workstream 11 — Live Solar View (signature feature)

**Route:** `/solar`  
**Brand group:** UNDERSTAND

A layered view of the Sun with Google Maps-style toggleable overlays. Rendered using MapLibre GL or SVG — no EUV imagery (ARUNA has no imager). Layers are data-driven from pipeline outputs.

### Toggleable layers

| Layer | Data source | Visual |
|-------|-------------|--------|
| HOPE | PhysicsStore | Pulsing ring at solar disk center when HOPE active |
| Forecast Risk | ForecastStore | Disk color gradient by P(M+X) |
| Dual-trigger | MissionStore | SoLEXS/HEL1OS status badges |
| Mission State | DecisionStore | Colored border (green/yellow/orange/red) |
| NavIC Impact | ThreatStore | India overlay with regional risk colors |

This is not a real solar image viewer — it is a **physics-state visualization** projected onto the solar disk concept, making abstract numbers spatially intuitive. Nobody else is doing this.

---

## Pipeline Health — Domain View

**Route:** `/health`

Expanded from layer-by-layer to domain view:

| Domain | Stages | Shows |
|--------|--------|-------|
| Data | L0, L1 | PRADAN status, FITS freshness, dead-time fraction, gap count |
| Physics | L2, XL2, XL3 | T/EM fit quality, MPCI score, frame rate, χ² |
| Forecast | L4, XL4 | TCN inference latency, MAPIE coverage, Confidence Engine output |
| Threat | L6 | NavIC model latency, bulletin generation time |
| Decision | XL5, DX | Mission Impact latency, Decision Engine latency, recommendation age |
| UI | Socket.IO, API | Frame delivery latency, client connection count, REST response times |

Each domain card shows: status badge, latency (current vs target), version, frame rate, quality score.

---

## Acceptance Criteria

| Workstream | Done when |
|------------|-----------|
| Test fix | `pytest python/tests/test_l6_threat.py` — 0 failures |
| Frontend wiring | All pages show real data when connected; synthetic when offline; Mission Banner visible everywhere |
| Decision Engine | `test_decision_engine.py` passes; X6.3 path produces ABORT_EVA or SAFE_MODE |
| XL-4 Confidence Engine | `test_xl_confidence.py` passes; MPCI boost visible in contributor breakdown |
| XL-5 Mission Impact | `test_xl_mission_impact.py` passes; X6.3 → EXTREME South India, EVA DELAY |
| XL-6 Integrity Monitor | `test_xl_integrity_monitor.py` passes; T=500MK → CRITICAL flag |
| E2E Replay | `test_integration.py::test_x63_replay_end_to_end` passes; 0 CRITICAL integrity flags |
| Config page | Profile switching changes config.yaml; Competition profile locks UI |
| Mission Chronicle | Timeline shows real events; clickable events open ExplainabilityTrace panel |
| Explainability Center | Reasoning chain renders; all 5 sections populated from live stores |
| Live Solar View | All 5 layers toggle; HOPE layer pulses when HOPE active |
| Pipeline Health | Domain view shows all 6 domains with latency, version, quality |
| Brand navigation | Sidebar shows OBSERVE/UNDERSTAND/FORESEE/PROTECT/LEARN/OPERATE groupings |
| Mission Banner | Persists across all pages; tracks connectionMode color |
