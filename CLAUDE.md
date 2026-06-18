# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

ARUNA (Aditya-L1 Real-time Unified Nowcasting & Alert System) is a physics-grounded solar flare detection and forecasting pipeline built for ISRO's Aditya-L1 satellite instruments (SoLEXS soft X-ray + HEL1OS hard X-ray). The full specification is in `ARUNA_Pipeline_PRD.md`. Implementation is for BAH 2026 Problem Statement 15.

## Three-Tier Architecture

```
React + TypeScript (Next.js)          ← mission operations UI
          │
   WebSocket / REST
          │
Node.js API Gateway (Fastify)         ← auth, routing, caching, real-time push
          │
  Internal REST / gRPC
          │
Python FastAPI microservices          ← all scientific computation
(L1 Ingestion → L2 Physics → L3 Nowcasting → L4 Forecasting
 → L5 Analog → L6 Threat → L7 LLM)
          │
        Kafka
          │
       PostgreSQL
```

Streamlit is **not** the production frontend. It lives in `streamlit/` as an internal research/debug tool for scientists only.

## Setup

### Python scientific layer
```bash
conda create -n aruna python=3.11
conda activate aruna
pip install -r python/requirements.txt

cp config/config.example.yaml config/config.yaml
# Set PRADAN_USERNAME, PRADAN_PASSWORD in environment or config.yaml
```

### Node.js API gateway
```bash
cd api/
npm install
```

### React frontend
```bash
cd web/
npm install
```

## Running

```bash
# Full stack
docker compose up -d

# Verify API health
curl http://localhost:3001/health

# Python pipeline only (no Docker)
python -m aruna.pipeline.run

# Node API only
cd api/ && npm run dev        # http://localhost:3001

# React frontend only
cd web/ && npm run dev        # http://localhost:5173

# X6.3 event replay (primary demo — always works offline)
python -m aruna.replay --event-id SOL2024-02-22-X6.3 --fits-dir tests/fixtures/x63/ --speed 10

# May 2024 superstorm counterfactual report
python -m aruna.counterfactual --start 2024-05-09T00:00:00Z --end 2024-05-11T23:59:59Z
```

## Testing

```bash
# Python tests
pytest python/

# Single test
pytest python/tests/test_physics.py::test_temperature_x63_event -v

# Physics validation (most critical — X6.3 T/EM vs Sarwade et al.)
pytest python/tests/test_physics.py -v

# Node API tests
cd api/ && npm test

# Frontend tests
cd web/ && npm test
```

## Directory Structure

```
web/                  # React + TypeScript + Vite frontend (no Next.js)
  src/
    pages/            # one file per dashboard page
    components/       # shadcn/ui + custom components
    stores/           # Zustand state stores
    hooks/            # TanStack Query + Socket.IO hooks
    lib/              # utilities, chart helpers

api/                  # Node.js Fastify gateway
  routes/             # REST endpoints
  ws/                 # Socket.IO real-time handlers
  prisma/             # Prisma schema + migrations

python/               # All scientific computation
  aruna/
    l0_source/
    l1_ingestion/
    l2_physics/
    l3_nowcasting/
    l4_forecasting/
    l5_analogs/
    l6_threat/
    l7_llm/
    xl/               # XL extension modules
  tests/
    fixtures/x63/     # X6.3 FITS test data

notebooks/            # Jupyter — scientist physics inspection, model validation

config/
  config.yaml         # All tunable thresholds — do not hardcode in source
models/
  tcn_v1.pt
  mapie_calibration.pkl
data/
  cache/              # FITS files, LRU 48h / 50 GB max
```

## Tech Stack

### Frontend (`web/`)
- **React + TypeScript + Vite** — app framework (no Next.js)
- **Tailwind CSS + shadcn/ui** — design system
- **Plotly.js** — scientific time-series charts
- **React Flow** — pipeline visualization
- **Zustand** — global state management
- **TanStack Query** — server state / data fetching
- **Socket.IO client** — real-time physics frame streaming
- **Framer Motion** — UI animations
- **MapLibre GL** — India NavIC regional risk map

### API Gateway (`api/`)
- **Node.js + Fastify** — HTTP server
- **Socket.IO** — WebSocket for 1-second physics stream to frontend
- **Prisma ORM + PostgreSQL** — relational data
- **Redis** — caching, session store
- **Kafka** (kafka-node / kafkajs) — consumes Python pipeline events

### Python Scientific Layer (`python/`)
- **FastAPI** — internal HTTP/gRPC interface to Node gateway
- **SunPy + sunkit-spex** — solar physics, isothermal fitting
- **Astropy** — FITS parsing
- **Sherpa** — spectral fitting / background subtraction
- **PyTorch** — DualStreamTCN model
- **NumPy + SciPy** — signal processing, physics computations
- **scikit-learn** — preprocessing, logistic regression sub-models
- **MAPIE** — conformal prediction uncertainty
- **FAISS** — analog vector search
- **ruptures** — PELT change-point detection
- **dtaidistance** — DTW similarity
- **uni2ts (Moirai)** — zero-shot time-series foundation model
- **structlog** — structured logging
- **LangGraph** (optional) — XL-6 agent workflow

### Infrastructure
- **Docker + Docker Compose** — development
- **Kafka + Zookeeper** — message bus between Python layers
- **PostgreSQL** — event store, model registry, validation results
- **Redis** — hot caching for Node gateway
- **MinIO** — FITS object storage (production)
- **Prometheus + Grafana** — metrics and dashboards
- **Loki** — log aggregation
- **Nginx** — reverse proxy
- **MLflow** — model experiment tracking and artifact store
- **Ollama (Mistral 7B)** — offline LLM, air-gapped capable

## Frontend Pages (12–15)

| Page | Purpose |
|------|---------|
| Mission Control | System status, active alert count, sun status indicator |
| Live Solar View | 1-second SoLEXS + HEL1OS light curves updating in real time |
| Physics Diagnostics | T(t), EM(t), DNI gauge, TNPSV phase space, HOPE indicator |
| Nowcasting | Active flare detection, dual-trigger status, flare phase label |
| Forecast Center | 5-class probability, horizon selector, MAPIE confidence intervals |
| Threat Intelligence | NavIC regional risk map, HF blackout, Gaganyaan advisory, SpaceWx bulletin |
| Historical Replay | X6.3 always-on replay; arbitrary date range replay |
| Master Flare Catalog | Searchable event table, drill-down, CSV export |
| Pipeline Health | Per-stage latency, Kafka lag, PRADAN status, data quality |
| Validation | Rolling confusion matrix, TSS, Kappa, ECE trends |
| Model Performance | Bootstrap CI metrics, calibration reliability diagram |
| Gaganyaan Advisor | 7-day EVA/launch window calendar |
| Configuration | Threshold tuning (maps to `config.yaml`) |
| Logs | Structured log stream from all pipeline stages |

## Pipeline Architecture (Python layer)

Nine sequential processing layers connected by Kafka:

| Layer | Module | Kafka output topic | Purpose |
|-------|--------|--------------------|---------|
| L0 | `l0_source/` | `aruna.source.health` | PRADAN FITS access, source health |
| L1 | `l1_ingestion/` | `aruna.observations.{solexs,hel1os}` | FITS read, dead-time correction, time alignment, GOES label join |
| L2 | `l2_physics/` | `aruna.physics.frames` | T/EM isothermal fit, DNI, TNPSV, SHEI, HFD, QPP, HOPE flag |
| L3 | `l3_nowcasting/` | `aruna.nowcast.events` | PELT + P85 dual-trigger, GOES classification, Master Catalog write |
| L4 | `l4_forecasting/` | `aruna.forecast.frames` | DualStreamTCN + Moirai ensemble, MAPIE conformal uncertainty |
| L5 | `l5_analogs/` | `aruna.analogs.results` | DTW + FAISS top-3 historical analog retrieval |
| L6 | `l6_threat/` | `aruna.threat.output` | NavIC/HF/Gaganyaan risk, satellite safing queue, SpaceWx bulletin |
| L7 | `l7_llm/` | `aruna.alerts.briefings` | Ollama/Mistral offline briefing with hallucination guard |

ARUNA-XL modules (`xl/`) sit above L0–L7 consuming outputs: XL-1 AR Registry, XL-2 MPCI coherence, XL-3 physics tokens, XL-4 stacked forecast ensemble, XL-5 Earth/NavIC impact, XL-6 self-correction agent.

## Core Data Schemas

All Python layers exchange typed dataclasses serialised as JSON on Kafka. Node gateway deserialises and forwards relevant subsets to the frontend over Socket.IO.

- `ObservationFrame` — dead-time-corrected spectrum from one instrument at one timestamp
- `PhysicsFrame` — T(t), EM(t), DNI, TNPSV, SHEI, HFD, HOPE flag — 1-second cadence from L2
- `NowcastEvent` — confirmed flare alert with GOES class, confidence, lead time
- `ForecastFrame` — 5-class probability vector with MAPIE conformal intervals
- `AlertEvent` — unified alert with threat flags, provenance hash

All timestamps: ISO 8601 UTC. All messages include `schema_version`, `pipeline_version`, `created_at`.

## Key Algorithms (L2 Physics Engine)

These are the scientific core. Implementations must match the PRD exactly:

- **Isothermal fit:** `sunkit-spex f_vth` model, energy range 3–20 keV, coronal abundances via CHIANTI. Validation target: X6.3 peak T ≈ 23.5 MK within 1σ (Sarwade et al. arXiv:2502.19795).
- **DNI (Dynamic Neupert Integral):** `∫F_HEL1OS dt / F_SoLEXS`. HOPE threshold: DNI > 1.5.
- **TNPSV:** Normalised velocity in T-EM phase space. HOPE quadrant: 60°–120° (T rising, EM flat).
- **SHEI:** `log10[F_HEL(40–70 keV) / F_HEL(10–25 keV)]`. Impulsive onset: SHEI > −0.8 and hardening.
- **HOPE precursor:** Fires when T rising ≥ 3s AND EM flat AND HEL1OS quiet AND TNPSV in precursor quadrant. Must fire ≥ 90s before X6.3 peak.

Dead-time constants (non-paralysable model, Sarwade et al. 2025): SDD1/SDD2 = 4.5e-6 s, CdTe = 2.1e-6 s, CZT = 1.8e-6 s. SDD1 saturation at 5e4 counts/s → auto-switch to SDD2.

## Critical Implementation Decisions

- **Dual-trigger consensus required:** Both SoLEXS AND HEL1OS must trigger within 3 minutes for a confirmed `FLARE_ONSET`. SoLEXS-only never produces a confirmed alert.
- **Asymmetric loss (alpha=0.2):** Under-predicting M/X flares penalised 4× more than over-predicting. Standard MSE must not be substituted.
- **Temporal split only:** Train: 2023-09-02 to 2024-06-30. Validate: 2024-07-01 to 2024-12-31. Test: 2025-01-01 to 2025-06-30. No random shuffling — temporal leakage invalidates results.
- **Rolling P85 baseline:** 3-hour 85th percentile, recomputed every 60 seconds. Static thresholds must not be used.
- **LLM is narrator only:** L7 never makes decisions. Every briefing validated by hallucination guard; fallback is always the deterministic template.
- **No Streamlit.** React frontend for operators, Jupyter notebooks for scientist inspection.
- **SQLite in dev, PostgreSQL in prod.** Config toggles via `database.type`. Prisma handles PostgreSQL from Node; SQLAlchemy handles it from Python.

## Validation Reference Event

The **X6.3 flare on 2024-02-22** is the primary validation anchor. Every physics and nowcasting module has an acceptance criterion tied to this event. Test fixtures live in `python/tests/fixtures/x63/`. The Historical Replay page is hardcoded to this event and must always work even without live PRADAN data.

## Environment Variables

`PRADAN_USERNAME`, `PRADAN_PASSWORD`, `DATABASE_URL` (PostgreSQL), `KAFKA_BOOTSTRAP_SERVERS` (default: `localhost:9092`), `ARUNA_CONFIG` (path to config.yaml), `ARUNA_FORCE_NOAA_FALLBACK`, `REDIS_URL`, `MLFLOW_TRACKING_URI`, `OLLAMA_HOST`.
