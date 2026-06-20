# TCN Training Pipeline — Design Spec

**Date:** 2026-06-20  
**Phase:** Training infrastructure  
**Status:** Approved — ready for implementation  
**PRD reference:** §8 (L4 — Forecasting Engine), §8.5 (Loss), §8.9 (Temporal split)  
**Prerequisite:** L1 ingestion + L2 physics engine complete, 745 SoLEXS + 1750 HEL1OS FITS downloaded

---

## 1. Objective

Train `DualStreamTCN` on real Aditya-L1 observations (SoLEXS + HEL1OS) with NOAA GOES flare
event catalog as ground-truth labels, producing:

- `models/tcn_v1.pt` — trained TCN weights
- `models/mapie_calibration.pkl` — MAPIE conformal calibrator fitted on validation set
- `models/feature_scaler.pkl` — StandardScaler fitted on training set only

The model must classify the next-30-minute flare risk into 5 classes: QUIET, B, C, M, X,
with under-prediction of M/X penalised 4× via asymmetric focal loss (α=0.2).

---

## 2. Input Data

### 2.1 Aditya-L1 FITS (training inputs)

| Instrument | Files | Coverage | Format |
|------------|-------|----------|--------|
| SoLEXS | 745 files | ~2 years (2024–2026) | `AL1_SLX_L1_YYYYMMDD_v1.0` — 1 file/day |
| HEL1OS | 1750 files | ~2 years | `AL1_HXS_L1_*` — ~2–3 files/day |

Each SoLEXS file = one calendar day at 1-second cadence = ~86,400 rows.  
Each HEL1OS file = one orbit (~10 hours) at 1-second cadence.

ISRO Level 1 products: dead-time correction and gain calibration already applied by PRADAN.
The ARUNA L1 module reads these and extracts spectral bands; it does **not** re-apply dead-time.

**MAG and VELC are NOT training inputs.** They appear only in the XL-module MPCI score.

### 2.2 NOAA GOES Flare Event Catalog (labels)

Source: NOAA Space Weather Prediction Center — GOES XRS event list.  
Covers same 2-year window as the FITS data.  
Downloaded once via SunPy HEK query; cached locally as `data/goes_events.parquet`.

Each event record provides: `start_time`, `peak_time`, `end_time`, `goes_class` (e.g. `X6.3`).

**Why GOES for labels:** GOES XRS is the international reference standard for flare classification.
The scientist-marked peak events from ISRO cross-reference against GOES. The GOES catalog *is*
those labels in standardised downloadable form. ARUNA never depends on GOES at inference time —
only at training label-join time.

---

## 3. Data Pipeline

```
SoLEXS FITS (745 files)   HEL1OS FITS (1750 files)
         │                          │
   FitsReader.read_solexs()   FitsReader.read_hel1os()
         │                          │
         └──────────┬───────────────┘
                    │
             L1Processor (align, gap-fill)
                    │
             L2PhysicsEngine (T/EM/DNI/TNPSV/SHEI/HFD/QPP/HOPE)
                    │
             PhysicsFrame (27 features, 1-second cadence)
                    │
          Cache: data/physics_cache/YYYY-MM-DD.parquet
                    │
                    ├── GOES event catalog join
                    │
          600-frame sliding windows (stride = 60s)
                    │
          Label: max GOES class with peak in [T, T+30min]
                 → QUIET if no event
                    │
          Temporal split (hard date boundaries, no shuffling)
                    │
          TrainingDataset (PyTorch Dataset)
```

### 3.1 Parquet Cache

Running L2 physics on every training epoch is prohibitively expensive (isothermal fitting is
~0.1s per timestep on CPU). Process all FITS once, write per-day Parquet files:

```
data/physics_cache/
  2024-02-22.parquet   # 86,400 rows × 27 columns
  2024-02-23.parquet
  ...
```

Estimated size: float32 × 27 cols × 86,400 rows/day × 745 days ≈ **7.8 GB total**.  
L2 processing time estimate: ~2–4 days on CPU (parallelisable across days).

### 3.2 Gap Detection

SoLEXS and HEL1OS must both be present for a valid training window (dual-stream requirement).
Days where one instrument is missing → those windows are excluded from training, not zero-filled.
A window straddling a date boundary with >10% missing rows is also excluded.

---

## 4. Label Construction

### 4.1 GOES Label Join

For each window ending at timestamp T:

```python
# Look for any GOES event with peak time in [T, T + 30 minutes]
events_in_horizon = goes_catalog[
    (goes_catalog.peak_time >= T) &
    (goes_catalog.peak_time < T + timedelta(minutes=30))
]

if events_in_horizon.empty:
    label = "QUIET"
else:
    # Take the highest class in the horizon
    label = max_goes_class(events_in_horizon.goes_class)
    # e.g. if both M2.8 and X6.3 are in range → label = X
```

### 4.2 Class Mapping

```
QUIET → 0
B     → 1
C     → 2
M     → 3
X     → 4
```

Soft labels (one-hot with ε=0.1 label smoothing) fed to asymmetric focal loss.

### 4.3 Expected Class Distribution

SC25 rising phase statistics (approximate):

| Class | % of windows | Notes |
|-------|-------------|-------|
| QUIET | ~94% | Background + B-class pre-events |
| B | ~3% | Low-level activity |
| C | ~2% | Common, moderate events |
| M | ~0.8% | Training-critical |
| X | ~0.2% | Rare — 4× penalty if missed |

**Class imbalance handling:**
- `WeightedRandomSampler` in DataLoader: over-samples M/X by 10×
- Asymmetric focal loss (α=0.2): under-prediction penalty 4× over-prediction
- Both mechanisms active simultaneously

---

## 5. Dataset Construction

### 5.1 Window Parameters

```python
SEQUENCE_LENGTH = 600    # 10 minutes at 1-second cadence
WINDOW_STRIDE   = 60     # slide by 60 seconds between windows
LABEL_HORIZON   = 30     # minutes after window end
```

Approximate window count from 745 days:
- Raw: 745 × 86,400 / 60 ≈ 1,073,600 windows
- After gap exclusion and dual-instrument alignment: estimate ~700,000 usable windows

### 5.2 Feature Normalisation

`StandardScaler` fitted on training set only (no validation/test leakage).
Saved to `models/feature_scaler.pkl` and loaded by `FeatureExtractor.from_file()`.
EM values (~1e47–1e49) overflow float32 without scaling — scaler is mandatory.

---

## 6. Temporal Split

Per PRD §8.9 — hard date boundaries, no random shuffling:

```
TRAIN    2023-09-02 → 2024-06-30   (SC25 rising phase)
VALIDATE 2024-07-01 → 2024-12-31   (high-activity period — includes X6.3)
TEST     2025-01-01 → 2025-06-30   (held out — not touched until final eval)
```

**X6.3 (2024-02-22) falls in the TRAINING set.** It is a training anchor event, not a test event.
The test set covers 2025 events which are genuinely unseen.

---

## 7. Training Configuration

```python
OPTIMIZER   = AdamW(lr=1e-3, weight_decay=1e-4)
SCHEDULER   = CosineAnnealingLR(T_max=50, eta_min=1e-5)
EPOCHS      = 100 (early stopping: patience=10 on validation TSS)
BATCH_SIZE  = 64
LOSS        = asymmetric_focal_loss(alpha=0.2, gamma=2.0)   # already in tcn_model.py
```

Early stopping metric: **TSS (True Skill Statistic)** on validation set, not loss.
TSS is the operational metric — loss minimisation alone can still miss M/X events.

```python
TSS = POD - POFD
    = (TP / (TP + FN)) - (FP / (FP + TN))
```

Best validation TSS checkpoint saved as `models/tcn_v1.pt`.

---

## 8. MAPIE Calibration

After training, calibrate ensemble probabilities on the validation set:

```python
from mapie.classification import MapieClassifier

# val_probs: (N_val, 5) — ensemble output (TCN + Moirai weighted)
mapie = MapieClassifier(estimator=None, method="score", cv="prefit")
mapie.fit(val_probs, val_labels)
pickle.dump(mapie, open("models/mapie_calibration.pkl", "wb"))
```

Target: Expected Calibration Error (ECE) < 0.05 after calibration (PRD §8.10).

---

## 9. Scripts to Write

All 5 scripts go in `python/scripts/`. They call existing ARUNA modules — no new model code.

| Script | Purpose |
|--------|---------|
| `audit_pradan_fits.py` | Rewrite existing stub: date coverage table, SoLEXS/HEL1OS alignment gaps, missing day report |
| `download_goes_labels.py` | SunPy HEK query → `data/goes_events.parquet` |
| `batch_l1_l2.py` | For each FITS pair: L1 → L2 → PhysicsFrames → daily Parquet in `data/physics_cache/` |
| `build_dataset.py` | Load Parquet cache → 600-frame windows → GOES label join → train/val/test split → `data/dataset/` |
| `train_tcn.py` | Full training loop → best checkpoint → MAPIE calibration → save all 3 artifacts |

---

## 10. Acceptance Criteria (from PRD §8.10)

- TSS > 0.60 on held-out test set (2025 events)
- ECE < 0.05 after MAPIE calibration
- Under-prediction rate < 5% for M5+ events (asymmetric loss validation)
- Ensemble (TCN + Moirai) beats single TCN by TSS ≥ 0.03
- No temporal leakage (no training events post-dating validation by < 24h)
- Inference latency < 10 seconds per forecast frame

---

## 11. What the Friend Does NOT Need

- MAG (magnetometer) — not a training feature
- VELC — XL-module MPCI only, not training
- SUIT UV — zero-filled gracefully if absent (`suit_uv_flux` → 0.0 in FeatureExtractor)
- GOES XRS flux time series — only the event catalog (not the raw flux) is needed for labels
- Any changes to `tcn_model.py` or `features.py` — both are complete and correct
