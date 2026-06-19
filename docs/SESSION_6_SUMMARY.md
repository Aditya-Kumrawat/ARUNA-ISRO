# ARUNA — Session 6 Summary
## L4 Forecasting Engine + Multi-Payload Expansion + XL Upgrades

**Date:** 2026-06-19  
**Pipeline Version:** `2.0.0`  
**Test Status:** ✅ 230 passed, 37 xfailed, 0 failed

---

## 1. Who Built What

Two agents collaborated on this session:

| Agent | Responsibility |
|-------|----------------|
| **Gemini** | L4 Forecasting Engine core — DualStreamTCN, Moirai ensemble, asymmetric focal loss, MAPIE calibration, feature extractor (initial 24 features), ForecastOrchestrator |
| **Claude** | Multi-payload expansion (SUIT + VELC ingestion), L2 TNCI field, L4 feature vector 24→27, XL-2 MPCI, XL-3 Physics Tokens, XL-1 Magnetogram Integration (JSOC SHARP) |

---

## 2. L4 Forecasting Engine (Gemini)

### Files Built

| File | Description |
|------|-------------|
| `python/aruna/l4_forecasting/tcn_model.py` | `DualStreamTCN` — two-branch TCN (thermal + nonthermal) merged via shared FC head |
| `python/aruna/l4_forecasting/moirai_model.py` | Moirai zero-shot wrapper with context-length clipping and TCN fallback |
| `python/aruna/l4_forecasting/ensemble.py` | Weighted ensemble (TCN 50% + Moirai 50%), fallback to 70/30 when Moirai context clipped |
| `python/aruna/l4_forecasting/features.py` | Feature extractor from `PhysicsFrame` (thermal + nonthermal + shared streams) |
| `python/aruna/l4_forecasting/calibrator.py` | MAPIE conformal prediction wrapper, 90% coverage guarantee |
| `python/aruna/l4_forecasting/forecaster.py` | `ForecastOrchestrator` — sliding 600s window, Kafka input/output |
| `python/aruna/l4_forecasting/models.py` | `ForecastConfig` dataclass, `FORECAST_HORIZONS = (5, 10, 15, 30, 60)` minutes |

### Key Design Decisions

**Asymmetric focal loss (α = 0.2)**  
Under-predicting M/X flares is penalised 4× more than over-predicting. Standard MSE/cross-entropy is not used. Enforced in `ForecastConfig.alpha`.

**Dual-stream architecture**
```
Thermal stream     [T_MK, EM_cm3, DNI, TNPSV_mag, TNPSV_angle, dT/dt, dEM/dt, d²T/dt²]
Nonthermal stream  [SHEI, dSHEI/dt, FeKα, HEL bands ×3, QPP_period, QPP_amp]
Shared context     [HFD, SoLEXS bands ×2, flare_phase one-hot ×5]
```

**Temporal split — strictly enforced**  
Train: 2023-09-02 → 2024-06-30 | Val: 2024-07-01 → 2024-12-31 | Test: 2025-01-01 → 2025-06-30  
`ForecastConfig.__post_init__` validates ordering — random shuffling is rejected at construction time.

**No trained weights yet.** `models/tcn_v1.pt` does not exist. The model runs on random initialisation until trained on the PRADAN FITS archive (2023–2025). Estimated training time on a 4060 GPU: ~8–12 hours on the X6.3 anchor dataset.

---

## 3. Multi-Payload Expansion (Claude)

### Why This Matters — Warning Timeline

| Signal | How far ahead | New in this session |
|--------|--------------|---------------------|
| R_VALUE (magnetogram) | 24h | ✅ XL-1 SHARP |
| dUSFLUX/dt (flux emergence) | 4–6h | ✅ XL-1 SHARP |
| SUIT UV brightening | 2–10 min | ✅ L1 + L2 + L4 |
| TNCI (thermal-nonthermal coupling) | 90s–5 min | ✅ L2 + L4 |
| HOPE flag (SoLEXS + HEL1OS) | 90s–3 min | L2 — prior session |
| HEL1OS onset | 0–30s | L3 — prior session |

### L1 Ingestion — SUIT + VELC Readers

**`python/aruna/l1_ingestion/fits_reader.py`**
- `read_suit(path)` — reads UV imager FITS (tries LIGHTCURVE/SPECTRUM/EVENTS extensions), returns 1-bin `FitsData` with `instrument='SUIT_FD'`, `dead_time_fraction=0.0`
- `read_velc(path)` — same pattern for VELC Doppler proxy

**`python/aruna/l1_ingestion/alignment.py`**
- `resample_suit_to_xray(suit, xray_timestamps)` — forward-fill from 60s SUIT cadence onto 1s X-ray grid via `scipy.interpolate.interp1d(kind='previous')`. Physically correct: UV brightness is constant across the 60s integration window.

**`python/aruna/l1_ingestion/pipeline.py`**
- `ingest()` now accepts optional `suit_path`, `velc_path` — failure of either never stops the core pipeline (wrapped in `try/except` with `logger.warning`)

**`python/aruna/models/observation.py`**
- Added `SUIT_FD`, `VELC_DP` to `INSTRUMENT_IDS`
- Added `is_suit()`, `is_velc()` helper methods

### L2 Physics — Three New Fields

**`python/aruna/l2_physics/models.py`** — all `float = 0.0` (backward-compatible):

```python
suit_uv_flux: float = 0.0               # integrated UV brightness [counts/s]
suit_uv_brightening_rate: float = 0.0   # dUV/dt [counts/s per second]
tnci: float = 0.0                       # Thermal-Nonthermal Coupling Index (PRD §13.4)
```

**TNCI formula (PRD §13.4)**
```
TNCI = d/dt[T × EM] / F_HEL1OS
     = (dT/dt × EM + T × dEM/dt) / (hel_total_flux + 1e-30)
```
When T rises rapidly and hard X-ray flux spikes simultaneously, TNCI spikes — an early signal of the impulsive phase onset before peak GOES flux registers.

### L4 Feature Vector — 24 → 27

```python
N_THERMAL    = 9   # was 8 — added tnci
N_NONTHERMAL = 8   # unchanged
N_SHARED     = 10  # was 8 — added suit_uv_flux, suit_uv_brightening_rate
N_FEATURES   = 27  # was 24
```

`DualStreamTCN` defaults updated to `n_thermal=9, n_shared=10`. Tests updated to import constants instead of hardcoding shapes.

---

## 4. XL-2 — Multi-Payload Coherence Index (MPCI)

**`python/aruna/xl/mpci.py`**

Exact PRD §13.3 formula — weighted average of per-payload normalised scores, re-normalised when any payload is absent:

```
MPCI = Σ(wᵢ × scoreᵢ) / Σ(wᵢ for available payloads)

Weights: SUIT=0.20, SoLEXS=0.35, HEL1OS=0.30, VELC=0.15
```

| Payload | Score derived from |
|---------|-------------------|
| SoLEXS | T rising + EM flat (TNPSV precursor quadrant check) |
| HEL1OS | SHEI > −0.8 threshold |
| SUIT | Sigmoid on UV flux level |
| VELC | Doppler proxy scalar, NaN-tolerant |

`MPCIResult.confidence_modifier`: +0.20 when MPCI > 0.8 (strong multi-payload coherence), −0.15 when MPCI < 0.3 (payload discord).

---

## 5. XL-3 — Physics Tokens

**`python/aruna/xl/physics_tokens.py`**

21-token `PhysicsTokens` dataclass encoding the full physics state snapshot:

```python
TOKEN_NAMES = [
    'T_MK', 'EM_cm3', 'DNI', 'TNPSV_mag', 'TNPSV_angle',
    'dT_dt', 'dEM_dt', 'd2T_dt2',
    'SHEI', 'dSHEI_dt', 'FeKa_flux',
    'hel_10_25', 'hel_25_50', 'hel_50_70',
    'QPP_period', 'QPP_amp', 'HFD',
    'solexs_3_6', 'solexs_6_10',
    'suit_uv_flux', 'tnci',
]
```

`extract_tokens(frame, mpci)` converts one `PhysicsFrame` → `PhysicsTokens`. All None/NaN values sanitised to 0.0 via `_safe(v)`. Tokens are the canonical input for XL-4 logistic ensemble (M3) and future attention-based models.

---

## 6. XL-1 — Magnetogram Integration (SDO/HMI SHARP)

The most physically impactful addition of this session. Magnetograms give ARUNA a **24-hour predictive horizon** — something no X-ray instrument can provide because the magnetic free energy is stored before the flare is triggered.

### Module Structure

```
python/aruna/xl1_magnetograms/
  models.py          — SHARPRecord, MagneticContext, SHARPConfig dataclasses
  downloader.py      — JSOCDownloader: drms library primary → JSOC HTTP API fallback
  parser.py          — Raw JSOC rows → typed SHARPRecord (handles DRMS_MISSING sentinels)
  sharp_features.py  — P(M+X, 24h) + P(M+X, 6h) heuristics from threshold bands
  cache.py           — File-based SHA-256-keyed pickle cache, 2h TTL (air-gap safe)
  fusion.py          — SHARPFusionLayer: soft probability shift on ForecastFrames
  __init__.py        — MagnetogramLayer high-level façade
```

### 24-Hour Probability Heuristic (Bobra & Couvidat 2015 approximation)

| Parameter | Threshold | Contribution |
|-----------|-----------|-------------|
| R_VALUE (Wang-Sheffield) | ≥ 15 | +0.40 (primary predictor) |
| R_VALUE | ≥ 10 | +0.25 |
| R_VALUE | ≥ 6 | +0.12 |
| TOTPOT (free energy proxy) | ≥ 4×10²³ erg/cm | +0.20 |
| TOTUSJH (current helicity) | ≥ 10²⁰ G²/m | +0.22 |
| MEANSHR (shear angle) | ≥ 60° | +0.18 |

### 6-Hour Probability (Flux Emergence Rate)

dUSFLUX/dt computed over last 5 records (~1 hour at 12-min SHARP cadence):

| Emergence rate | Contribution |
|----------------|-------------|
| > 4×10²⁰ Mx/hr | +0.35 |
| > 2×10²⁰ Mx/hr | +0.20 |
| > 1×10²⁰ Mx/hr | +0.10 |
| Elevated MEANSHR ≥ 60° | ×1.4 multiplier |
| Elevated MEANSHR ≥ 45° | ×1.2 multiplier |

### Fusion — Non-Destructive Soft Prior

`SHARPFusionLayer.apply()` modifies `ForecastFrame` probability vectors without touching the core SoLEXS+HEL1OS pipeline:

```
SHARP P(24h) ≥ 0.75  →  +0.20 shift toward M+X classes
SHARP P(24h) ≥ 0.50  →  +0.10 shift toward M+X classes
SHARP P(6h)  ≥ 0.60  →  additional +0.10 (rapid flux emergence)
SHARP P(24h) < 0.10  →  −0.08 dampening (magnetically quiet sun)
No SHARP data         →  0.00 (frames returned unchanged)
```

Mass is taken proportionally from QUIET/B/C classes (capped at 80% of available) and redistributed equally to M and X. Re-normalised to sum=1 after adjustment.

### Resilience Design

- `drms` library is a soft dependency — if not installed, falls back to JSOC HTTP API
- HTTP fallback returns `[]` on network failure (no exception raised)
- `MagnetogramLayer.get_context()` wraps all of the above in `try/except` → returns `None`
- `None` → `SHARPFusionLayer.apply()` returns frames unchanged
- File cache works without Redis — suitable for air-gapped deployments

### Usage
```python
from aruna.xl1_magnetograms import MagnetogramLayer

layer = MagnetogramLayer()
ctx = layer.get_context(harpnum=4977)   # HARP for 2024-02-22 X6.3 AR
frames = layer.fuse(forecast_frames, ctx)
```

---

## 7. Current Test Matrix

| Suite | Passed | xFailed |
|-------|--------|---------|
| `test_l1_ingestion.py` | ✅ | — |
| `test_l2_physics.py` | ✅ | — |
| `test_l3_nowcasting.py` | ✅ | — |
| `test_l4_forecasting.py` | ✅ | — |
| `test_l5_analogs.py` | — | 6 xfail |
| `test_l6_threat.py` | — | 7 xfail |
| `test_l7_llm.py` | — | 7 xfail |
| **TOTAL** | **230** | **37** |

0 failures.

---

## 8. Architecture State After This Session

```
L1 Ingestion     ✅  SoLEXS + HEL1OS + SUIT (60s→1s resample) + VELC
L2 Physics       ✅  T/EM fit + DNI + TNPSV + SHEI + HOPE + TNCI + SUIT UV
L3 Nowcasting    ✅  PELT + P85 dual-trigger + GOES class + Master Catalog
L4 Forecasting   ✅  DualStreamTCN + Moirai + MAPIE + N_FEATURES=27
L5 Analogs       ❌  6 xfail — FAISS retrieval not built
L6 Threat        ❌  7 xfail — NavIC/HF/Gaganyaan not built
L7 LLM           ❌  7 xfail — Ollama/Mistral not built

XL-1 Magnetograms  ✅  JSOC SHARP → MagneticContext → SHARPFusionLayer
XL-2 MPCI          ✅  PRD §13.3 weighted coherence index, VELC-optional
XL-3 Tokens        ✅  21-token PhysicsTokens for downstream ML
XL-4               ❌  LightGBM M1 + logistic M3 + Moirai M4 + isotonic M5
XL-5 Earth impact  ❌  CME propagation, Kp proxy, NavIC risk (feeds L6)
XL-6 Self-correct  ❌  LangGraph agent — lowest priority

React Frontend   ❌  0 of 12–15 pages built
Node API Gateway ❌  Not started
```

---

## 9. Next Phase — What to Build

The remaining 37 xfailed tests represent the three unbuilt downstream layers. All three are high-visibility for the BAH 2026 jury demo.

### Priority 1 — L6 Threat Intelligence (7 xfail)

The "so what" layer — converts physics alerts into operator-facing threat assessments. This is the centrepiece of the demo jury presentation.

| Module | What it builds |
|--------|---------------|
| NavIC regional risk | India ionospheric grid, Kp proxy → EIA amplification by region |
| HF blackout predictor | SoLEXS flux → D-layer absorption model |
| Gaganyaan EVA advisor | 7-day window scoring (flare probability + trajectory) |
| Satellite safing queue | X3+ → auto-safing command generation |
| SpaceWx bulletin | Structured operator report per confirmed event |

### Priority 2 — L5 Analogs (6 xfail)

FAISS-based historical event retrieval. Given a current `PhysicsFrame` sequence, finds the 3 most similar archived flares using DTW distance. Feeds the Forecast Center and Historical Replay pages.

### Priority 3 — L7 LLM Narrator (7 xfail)

Offline Ollama/Mistral 7B briefing generator with hallucination guard — every number in the briefing must appear verbatim in the source `NowcastEvent`/`ForecastFrame`. Deterministic template is the fallback. L7 is always optional.

### Priority 4 — Wire XL-1 into ForecastOrchestrator

`MagnetogramLayer` is built but not yet called from `forecaster.py`. Requires:
1. `ForecastOrchestrator.__init__` accepts optional `MagnetogramLayer`
2. `process()` calls `layer.get_context(active_harpnum)` each cycle
3. `layer.fuse(frames, ctx)` applied before Kafka publish
4. `active_harpnum` tracked from `aruna.xl.ar_state` Kafka subscription

### Priority 5 — React Frontend

12–15 dashboard pages, zero built. Minimum for a functional demo: Physics Diagnostics, Nowcasting, Forecast Center, Threat Intelligence (the visual centrepiece).
