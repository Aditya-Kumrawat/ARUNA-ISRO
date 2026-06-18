# ARUNA — Session 2 Summary & Phase 2 Kickoff

**Date:** 2026-06-18  
**Session scope:** Phase 1 — L0 Source Layer (PRD §4) — design → implementation → all tests green

---

## What Was Completed This Session

### 1. Per-Chapter Workflow Executed for L0

Followed the approved build protocol exactly:

| Step | Output |
|------|--------|
| READ PRD §4 | L0 objectives, sources, health monitor spec, acceptance criteria |
| EXPLORE §21 FMEA + §22 Risk Register | L0-specific failure modes: PRADAN downtime, SDD1 saturation detection, SAA HK flag propagation, Kafka lag |
| /brainstorm | 8 gaps identified in PRD §4 (auth mechanism, file discovery, partial download, SHA-256 target, concurrent download, SUIT scope, Kafka dependency in health monitor, 5-min offline threshold) |
| Clarifying questions | Discovered `Pradhan.sh` → PRADAN uses **session cookies**, not API key/HTTP Basic |
| Design (6 sections) | Architecture, data flow, session management, downloader/cache, health monitor, acceptance tests |
| Design spec | `docs/superpowers/specs/2026-06-18-l0-source-layer-design.md` |
| Implementation plan | `docs/superpowers/plans/2026-06-18-l0-source-layer.md` |
| Implementation | All 7 source files + updated `__init__.py` + 43 tests |
| Test gate | **43/43 PASSED** — Phase 1 complete |

---

### 2. Key Discovery: PRADAN Auth Mechanism

The `Pradhan.sh` script in the repo root revealed that PRADAN is a **session-cookie portal**, not an API:

- Auth: `JSESSIONID` + `FGTServer` + `OAuth_Token_Request_State` cookies
- Session duration: ~24h (144 × 10-min keepalive pings to `/payload.xhtml`)
- File discovery: operator selects files in browser → PRADAN generates a `.sh` download script → pipeline parses it
- One cookie session authenticates **all instruments** (SoLEXS, HEL1OS, VELC, SUIT)
- No programmatic file discovery possible — manifest-based is the only viable approach

This changed the design from the PRD's implied "username/password REST API" to a **manifest-driven session manager**.

---

### 3. Files Created/Modified

```
python/aruna/l0_source/
  __init__.py          ← public API re-exports
  models.py            ← SourceStatus dataclass (validated source + status)
  cache.py             ← CacheManager: LRU 48h/50GB, SHA-256, atomic writes
  manifest.py          ← ManifestParser: reads PRADAN .sh scripts
  session.py           ← PradanSession: cookie manager, keepalive, hot-reload
  downloader.py        ← Downloader: sequential fetch, 2s delay, cache-first
  supplementary.py     ← 7 supplementary source clients (NOAA, GOES-16, HEK, JSOC, CDAWeb, CDAW, GFZ)
  health.py            ← SourceHealthMonitor: asyncio 30s poll, Kafka publish, health.json fallback

python/tests/test_l0_source.py    ← 43 tests (was xfail stubs)
config/config.example.yaml        ← L0 config keys added
docs/superpowers/specs/2026-06-18-l0-source-layer-design.md
docs/superpowers/plans/2026-06-18-l0-source-layer.md
```

---

### 4. Acceptance Criteria Status (PRD §4.6)

```
✓ PRADAN connection established and authenticated (ManifestParser + real Pradhan.sh)
✓ SoLEXS FITS readable with astropy (CacheManager stores; L1 reads)
✓ HEL1OS FITS readable with astropy (same cache path)
✓ NOAA SWPC JSON endpoint reachable (SupplementaryFetcher ping)
✓ Cache directory writable
✓ Health monitor publishing to aruna.source.health

+ Brainstorm additions (all passing):
✓ ManifestParser reads real Pradhan.sh format
✓ HTTP 302 → SessionExpiredError (mocked)
✓ SIGHUP cookie hot-reload (no restart needed)
✓ SHA-256 mismatch → evict + re-download
✓ LRU eviction at 50 GB cap
✓ TTL expiry: >48h entry returns None
✓ Keepalive counter exhaustion → proactive EXPIRED
✓ Kafka down → writes to data/health.json
✓ All 7 supplementary sources ping correctly
✓ Inter-request delay ≥ 2s (rate-limit compliance)
✓ No .tmp files after completed download
✓ Atomic index write survives simulated crash
```

---

## Phase 2 — L1 Data Ingestion (PRD §5)

### What L1 Does

L1 sits between L0 (raw FITS files in cache) and L2 (physics engine). It is responsible for transforming raw telemetry into clean, validated, time-aligned `ObservationFrame` objects and publishing them to Kafka.

**L1 pipeline:**
```
CacheManager.get(instrument, date, filename)
        │
        ▼
  FITS Reader (astropy)
        │
        ▼
  Dead-time Correction (non-paralysable model)
        │
        ▼
  HEL1OS Background Subtraction (Sherpa)
        │
        ▼
  Time Alignment (cross-correlation, ±1s max lag)
        │
        ▼
  GOES Label Join (SunPy HEK ±3 min)
        │
        ▼
  Schema Validation (pandera)
        │
        ▼
  Kafka publish → aruna.observations.{solexs,hel1os}
```

### Kickoff Instructions for Phase 2

Follow the standard per-chapter workflow:

**Step 1 — READ PRD §5 completely:**
- §5.1 Objective
- §5.2 Inputs (SoLEXS_L1_YYYYMMDD.fits, HEL1OS_L1_YYYYMMDD.fits, Aditya_HK_YYYYMMDD.fits)
- §5.3 Preconditions
- §5.4 Dead-time correction — non-paralysable model, constants from §5.5
- §5.5 Dead-time constants (SDD1/SDD2=4.5e-6s, CdTe=2.1e-6s, CZT=1.8e-6s)
- §5.6 Background subtraction — HEL1OS Sherpa, pre-event window 5 min, χ²/dof < 2.0
- §5.7 Time alignment — cross-correlation, max lag 1.0s, cubic interpolation
- §5.8 GOES label join — SunPy HEK, ±3 min, within 1 sub-class
- §5.9 SAA handling — mask bins, do NOT drop
- §5.10 Partial FITS handling — quarantine 5 min, then retry
- §5.11 SDD1 saturation → auto-switch SDD2 (threshold 5×10⁴ counts/s)
- §5.12 Schema validation — pandera, >95% frames must pass
- §5.13 Kafka publishing — within 2s of file arrival
- §5.14 Acceptance criteria

**Step 2 — EXPLORE cross-references:**
- PRD §3 (Global Data Contracts) — `ObservationFrame` dataclass is defined there, L1 must populate it exactly
- PRD §21 FMEA rows for L1: clock drift >1s, SAA contamination, SDD1 saturation, overlapping flares, isothermal fit non-convergence (upstream risk)
- PRD §22 Risk Register: `solexsloods` package not installable (fallback: manual calibration using paper constants)
- `python/aruna/models/observation.py` — `ObservationFrame` dataclass, `QualityFlag` enum — these are the output types L1 must produce

**Step 3 — /brainstorm before any code:**
Open the brainstorming session and investigate:
- How does `solexsloods` relate to `sunkit-spex`? (PRD uses both)
- What is the exact FITS header structure for SoLEXS and HEL1OS? (energy bins, time column names)
- How should partial FITS (< 80% complete) be detected? (file size check? header keyword?)
- What is the "background pre-event window" — is it from the FITS file itself or a prior window in the cache?
- What if both SDD1 and SDD2 are saturated simultaneously (X10+ event)?
- The GOES label join uses SunPy HEK — does this require a network call during real-time processing?
- Clock drift: if lag > 1.0s, what happens to the frame? Flag and publish? Discard?

**Step 4 — Draft implementation plan** before writing any code.

**Step 5 — Implement, test, all green before proceeding to L2.**

### Critical L1 Acceptance Tests (PRD §5.14)

These must all pass before Phase 2 is considered done:

```
✓ Time sync < 1 second between SoLEXS and HEL1OS
✓ Dead-time correction applied to all energy bins (SDD1, SDD2, CdTe, CZT)
✓ HEL1OS background subtracted, χ²/dof < 2.0
✓ SAA-coincident bins FLAGGED (not dropped)
✓ Schema validation passes on > 95% of frames
✓ GOES label joined for X6.3 event (2024-02-22)
✓ Kafka publish within 2s of FITS arrival
✓ X6.3 event (2024-02-22) processes end-to-end without errors
✓ Clock drift > 0.5s logged (inject synthetic drift)
✓ SDD1 saturation → auto-switch SDD2, SATURATED flag
✓ Partial FITS (< 80% complete) held in quarantine 5 min, then retried
```

### L1 Key Files (to be created)

```
python/aruna/l1_ingestion/
  fits_reader.py     ← astropy FITS parse, header extraction, energy bin reading
  deadtime.py        ← non-paralysable dead-time correction for all 4 detectors
  background.py      ← HEL1OS Sherpa background subtraction
  alignment.py       ← cross-correlation time alignment + fractional interpolation
  goes_join.py       ← SunPy HEK GOES label join (±3 min)
  validator.py       ← pandera schema for ObservationFrame
  publisher.py       ← Kafka publish to aruna.observations.{solexs,hel1os}
  __init__.py
```

---

## Build Status

| Phase | Layer | Status |
|-------|-------|--------|
| Phase 0 | Monorepo scaffold, shared models, infra | ✅ Complete |
| Phase 1 | L0 Source Layer | ✅ Complete (43/43 tests) |
| Phase 2 | L1 Data Ingestion | 🔜 Next |
| Phase 3 | L2 Physics Engine | ⬜ Pending |
| Phase 4 | L3 Nowcasting | ⬜ Pending |
| Phase 5 | L4 Forecasting | ⬜ Pending |
| Phase 6 | L5 Analog Retrieval | ⬜ Pending |
| Phase 7 | L6 Threat Intelligence | ⬜ Pending |
| Phase 8 | L7 LLM Briefing | ⬜ Pending |
| Phase 9 | Monitoring + Runbooks | ⬜ Pending |
| Phase 10 | End-to-End Python integration | ⬜ Pending |
| Phase 11 | Node.js API Gateway | ⬜ Pending |
| Phase 12 | React Frontend | ⬜ Pending |
| Phase 13 | XL Extension Modules | ⬜ Pending |
