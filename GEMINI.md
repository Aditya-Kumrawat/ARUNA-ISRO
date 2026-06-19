# GEMINI.md

This file provides guidance to Gemini (Antigravity CLI / `agy`) when working in this repository.
The canonical project spec is `ARUNA_Pipeline_PRD.md`. Implementation is for BAH 2026 Problem Statement 15.

---

## What This Is

**ARUNA** (Aditya-L1 Real-time Unified Nowcasting & Alert System) — a physics-grounded solar flare
detection and forecasting pipeline for ISRO's Aditya-L1 satellite instruments:
- **SoLEXS** — soft X-ray spectrometer (thermal flare signal)
- **HEL1OS** — hard X-ray spectrometer (non-thermal impulsive signal)

---

## Current Pipeline State — Read This First

Before suggesting any implementation, check what is already built:

| Phase | Layer | Status | Tests |
|-------|-------|--------|-------|
| 1 | L0 Source | ✅ Built | passing |
| 2 | L1 Ingestion | ✅ Built | passing |
| 3 | L2 Physics Engine | ✅ Built | passing |
| 4 | L3 Nowcasting Engine | ✅ Built | 41 passed, 2 xfailed |
| 5 | L4 Forecasting | ⏳ Next | 10 xfail stubs |
| 6 | L5 Analogs | ⏳ Pending | 6 xfail stubs |
| 7 | L6 Threat | ⏳ Pending | 7 xfail stubs |
| 8 | L7 LLM Briefing | ⏳ Pending | 7 xfail stubs |

**Do not re-implement completed layers.** Run `pytest python/ -q` to see current state.
Full xfail catalogue: `docs/xfail_registry.md`
Session history: `docs/SESSION_4_SUMMARY.md`, `docs/SESSION_5_SUMMARY.md`

---

## Architecture

```
React + TypeScript + Vite (web/)         ← mission operations UI
          │
   WebSocket / REST
          │
Node.js Fastify Gateway (api/)           ← auth, routing, caching, real-time push
          │
  Internal REST / gRPC
          │
Python FastAPI microservices (python/)   ← all scientific computation
(L1 Ingestion → L2 Physics → L3 Nowcasting → L4 Forecasting
 → L5 Analogs → L6 Threat → L7 LLM)
          │
        Kafka
          │
       PostgreSQL
```

**Streamlit is NOT the production frontend.** It lives in `streamlit/` for scientist debug only.
Frontend is React + TypeScript + Vite in `web/` — no Next.js.

---

## Tool Mapping (Antigravity-specific)

Gemini on Antigravity uses different tool names than Claude Code. Map actions as follows:

| What to do | Tool to use |
|------------|------------|
| Read a file | `view_file` |
| Create a new file | `write_to_file` |
| Edit a file | `replace_file_content` or `multi_replace_file_content` |
| Run shell / tests | `run_command` |
| Search file contents | `grep_search` |
| Find files / list directory | `list_dir` (no glob — combine with `grep_search`) |
| Fetch a URL | `read_url_content` |
| Web search | `search_web` |
| Ask the user a question | `ask_question` |
| Dispatch a subagent | `invoke_subagent` with `TypeName: "self"` (full) or `"research"` (read-only) |

### Loading Skills

Antigravity has **no `activate_skill` tool**. To use a skill, read its `SKILL.md`:
```
view_file(".../plugins/superpowers/skills/<skill-name>/SKILL.md", IsSkillFile: true)
```
Set `IsSkillFile: true` — this is how Antigravity activates a skill's instructions.

### Task Tracking

Antigravity has no `TodoWrite`. Track tasks as a **markdown artifact**:
```
write_to_file("task-list.md", IsArtifact: true, ArtifactMetadata.ArtifactType: "task")
```
Edit it with `replace_file_content` / `multi_replace_file_content` as steps complete.

### Subagents

Two built-in TypeNames — no `define_subagent` needed for standard work:
- `TypeName: "self"` — full clone (reads, writes, runs commands). Use for implementing.
- `TypeName: "research"` — read-only (no file writes, no commands). Use for reviewing.

Multiple independent subagents: put them in one `invoke_subagent` call's `Subagents` array.

---

## Workflow Rules

### Before Writing Any Code

1. Check `docs/SESSION_5_SUMMARY.md` to understand what was last built.
2. Run `pytest python/ -q --tb=no` to see current test state.
3. Read `docs/xfail_registry.md` to understand which stubs exist for your target layer.
4. Read the relevant PRD section (`ARUNA_Pipeline_PRD.md`) before designing anything.

### Development Workflow (TDD)

Always: **write failing test → run → implement → run until green → commit**

```bash
# Run full suite
pytest python/

# Run one layer's tests
pytest python/tests/test_l4_forecasting.py -v

# Run single test
pytest python/tests/test_l4_forecasting.py::test_tss_over_0_65 -v
```

Never skip the failing-first step. Never commit without tests passing.

### Branching

- All implementation work on `master` (no separate feature branches unless explicitly requested).
- Commit after every completed module, not at the end of a session.
- Commit format: `feat(l4): add ForecastConfig` / `fix(l3): bounded deque maxlen`

### Configuration

All thresholds live in `config/config.yaml` and the layer's `Config` dataclass.
**Never hardcode a threshold** in a module file. If a number appears in two places it belongs in config.

---

## Critical Implementation Decisions

These are non-negotiable. Do not deviate without flagging to the user.

- **Dual-trigger consensus:** Both SoLEXS AND HEL1OS must trigger within 3 minutes for a confirmed `FLARE_ONSET`. SoLEXS-only produces a `SOLEXS_ONLY` advisory at `confidence=0.5` — never a confirmed alert.
- **Asymmetric loss α=0.2:** Under-predicting M/X flares penalised 4× more than over-predicting. **Standard MSE must not be substituted** for this loss function.
- **Temporal split only:** Train 2023-09-02→2024-06-30 | Val 2024-07-01→2024-12-31 | Test 2025-01-01→2025-06-30. **No random shuffling** — temporal leakage invalidates all results.
- **Rolling P85 baseline:** 3-hour 85th percentile, recomputed every 60s. No static thresholds.
- **LLM is narrator only:** L7 never makes operational decisions. Every briefing validated by hallucination guard; deterministic template is always the fallback.
- **SQLite in dev, PostgreSQL in prod.** Toggled via `db_url` config. No ORM in Python layers — raw sqlite3 / psycopg2.
- **No Streamlit** in any production path.
- **X3+ flares:** `confidence` forced to `1.0` regardless of model output. Never escalate on `SOLEXS_ONLY` advisories.

---

## Pipeline Kafka Topics

| Layer | Consumes | Produces |
|-------|----------|---------|
| L0 | PRADAN HTTP | `aruna.source.health` |
| L1 | FITS files | `aruna.observations.{solexs,hel1os}` |
| L2 | `aruna.observations.*` | `aruna.physics.frames` |
| L3 | `aruna.physics.frames` | `aruna.nowcast.events` |
| L4 | `aruna.nowcast.events` | `aruna.forecast.frames` |
| L5 | `aruna.forecast.frames` | `aruna.analogs.results` |
| L6 | `aruna.analogs.results` | `aruna.threat.output` |
| L7 | `aruna.threat.output` | `aruna.alerts.briefings` |

---

## Core Data Schemas (already defined — do not redefine)

| Class | File | Used by |
|-------|------|---------|
| `ObservationFrame` | `python/aruna/models/observation.py` | L1 → L2 |
| `PhysicsFrame` | `python/aruna/l2_physics/models.py` | L2 → L3 |
| `NowcastEvent` | `python/aruna/models/nowcast.py` | L3 → L4 |
| `ForecastFrame` | `python/aruna/models/forecast.py` | L4 → L5 |
| `AlertEvent` | `python/aruna/models/alert.py` | L6 → L7 |

All timestamps: ISO 8601 UTC. All messages include `schema_version`, `pipeline_version`, `created_at`.

---

## Key Algorithms Already Implemented

### L2 Physics (complete — do not modify unless fixing a bug)
- Isothermal fit: `sunkit-spex f_vth`, 3–20 keV, CHIANTI abundances
- DNI: `∫F_HEL1OS dt / F_SoLEXS`, HOPE threshold > 1.5
- TNPSV: normalised T-EM phase-space velocity, precursor quadrant 60°–120°
- SHEI: `log10[F_HEL(40–70 keV) / F_HEL(10–25 keV)]`, impulsive onset > −0.8

### L3 Nowcasting (complete — do not modify unless fixing a bug)
- Rolling P85 baseline with σ threshold, freeze on ONSET, clear+unfreeze on END
- PELT changepoint detection (ruptures, rbf model, 600-frame buffer)
- 4-state FSM: QUIET → HOPE → ONSET → PEAK → QUIET
- GOES classification: counts × 1e-8 → W/m² → letter + subclass
- Master Flare Catalog: SQLite ON CONFLICT upsert, <100ms

---

## Validation Reference Event

**X6.3 flare 2024-02-22** is the primary anchor for every module's acceptance test.
- Test fixtures: `python/tests/fixtures/x63/`
- Historical Replay must always work offline using these fixtures
- Reference: Sarwade et al. arXiv:2502.19795 — X6.3 peak T ≈ 23.5 MK

---

## Setup

```bash
# Python
conda create -n aruna python=3.11
conda activate aruna
pip install -r python/requirements.txt
cp config/config.example.yaml config/config.yaml
# Set PRADAN_USERNAME, PRADAN_PASSWORD in environment

# Node API
cd api/ && npm install

# Frontend
cd web/ && npm install
```

## Running

```bash
docker compose up -d                          # full stack
python -m aruna.pipeline.run                  # Python only
cd api/ && npm run dev                        # Node only (http://localhost:3001)
cd web/ && npm run dev                        # React only (http://localhost:5173)

# Always-offline demo
python -m aruna.replay --event-id SOL2024-02-22-X6.3 --fits-dir tests/fixtures/x63/ --speed 10
```

## Environment Variables

`PRADAN_USERNAME`, `PRADAN_PASSWORD`, `DATABASE_URL`, `KAFKA_BOOTSTRAP_SERVERS` (default: `localhost:9092`),
`ARUNA_CONFIG`, `ARUNA_FORCE_NOAA_FALLBACK`, `REDIS_URL`, `MLFLOW_TRACKING_URI`, `OLLAMA_HOST`

---

## Next Phase: L4 Forecasting Engine (PRD §8)

**Entry points:**
- Stubs: `python/tests/test_l4_forecasting.py` (10 xfail tests to replace)
- Placeholder: `python/aruna/l4_forecasting/__init__.py`
- Model artifact: `models/tcn_v1.pt`, `models/mapie_calibration.pkl`

**Modules to build:**
```
python/aruna/l4_forecasting/
  models.py        # ForecastConfig
  features.py      # PhysicsFrame → model input tensor
  tcn_model.py     # DualStreamTCN loader (PyTorch, models/tcn_v1.pt)
  moirai_model.py  # uni2ts Moirai zero-shot wrapper
  ensemble.py      # weighted ensemble + asymmetric loss (α=0.2)
  calibrator.py    # MAPIE conformal prediction
  forecaster.py    # ForecastOrchestrator
  __init__.py
```

**Key constraints:**
- Asymmetric loss α=0.2: `loss = α * (y - ŷ) if ŷ < y else (1-α) * (ŷ - y)` — never MSE
- Moirai fallback: if context > 512 steps → up-weight TCN, down-weight Moirai
- ForecastFrame already defined in `aruna/models/forecast.py` — import it, don't redefine
- Temporal split: train/val/test by date, never shuffled

**Kick-off prompt for this project:**
> "we have completed phase 4 now lets start with phase 5 that is the forecasting engine, check the aruna prd and brainstorm"
