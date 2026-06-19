# ARUNA — ML Training Guide for Team Members

This guide is written for team members who have access to Claude and need to create training data, train the ML models, and get the forecasting pipeline fully operational.

**You do not need to be a machine learning expert.** Claude will write the code for you. Your job is to understand what each step does, run the commands, and validate the outputs.

---

## Overview: What Models Exist

ARUNA has four ML components. Two need training, one needs downloading, one builds itself from data.

| Model | Purpose | Status | Output file |
|-------|---------|--------|-------------|
| **DualStreamTCN** | Predicts flare class (B/C/M/X) from 1-second physics data | Architecture built, **NO weights** | `models/tcn_v1.pt` |
| **MAPIE Calibrator** | Adds uncertainty intervals to TCN predictions | Code built, **needs fitting** | `models/mapie_calibration.pkl` |
| **Moirai** | Zero-shot time-series foundation model (ensemble member) | **Just download** | Auto-cached by `uni2ts` |
| **FAISS Analog Index** | Finds similar historical flares for context | Code built, **needs data** | `models/analog_index.faiss` |

The most important one is **DualStreamTCN**. Start there.

---

## Part 1 — Data You Need

### Option A: Real PRADAN data (best)

PRADAN is ISRO's data portal for Aditya-L1. You need `PRADAN_USERNAME` and `PRADAN_PASSWORD` set in your environment (get these from the team lead).

```bash
export PRADAN_USERNAME=your_username
export PRADAN_PASSWORD=your_password
```

The L0 source layer will download FITS files automatically when you run the pipeline. Training data period:
- **Train:** 2023-09-02 to 2024-06-30
- **Validate:** 2024-07-01 to 2024-12-31
- **Test:** 2025-01-01 to 2025-06-30

### Option B: Synthetic data (if no PRADAN access)

If you do not have PRADAN credentials, use Claude to generate synthetic FITS files that mimic real SoLEXS/HEL1OS data. This is good enough for initial training.

**Claude prompt to use:**

> I am working on the ARUNA solar flare forecasting project. Read the file `python/aruna/l1_ingestion/fits_reader.py` to understand the FITS format expected by the pipeline. Then write a script `python/scripts/generate_synthetic_fits.py` that generates realistic synthetic SoLEXS and HEL1OS FITS files covering 2023-09-02 to 2024-06-30. The SoLEXS data should include X-ray flux in 3-10 keV and 10-20 keV bands at 1-second cadence. The HEL1OS data should include 10-25 keV, 25-50 keV, and 50-70 keV bands. Include realistic background, noise, and at least 50 synthetic flare events at random times with realistic GOES classes (80% C-class, 15% M-class, 5% X-class). Use `astropy.io.fits` to write the files.

### GOES Labels (ground truth)

GOES flare labels are free from NOAA SWPC. Download the event list:

```bash
# Downloads all GOES flare events 2023-2025
curl -o data/goes_events.csv "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_2023.txt"
```

Or ask Claude to write a downloader:

> Write `python/scripts/download_goes_labels.py` that downloads GOES X-ray flare event lists from NOAA SWPC for 2023-2025, parses them into a pandas DataFrame with columns [start_time, peak_time, end_time, goes_class, goes_subclass], and saves to `data/goes_events.parquet`.

---

## Part 2 — Training DualStreamTCN

### What it is

`DualStreamTCN` is a two-branch Temporal Convolutional Network. It reads a 600-second (10-minute) sliding window of 27 physics features and outputs a 5-class probability vector:

```
Classes: B | C | M1-M4 | M5-M9 | X
```

The architecture is already written in `python/aruna/l4_forecasting/tcn_model.py`. You just need to train it.

### Step 1 — Generate PhysicsFrames (training features)

Run the L1→L2 pipeline on your FITS files to produce `PhysicsFrame` objects:

```bash
python python/scripts/generate_training_frames.py \
  --fits-dir data/fits/ \
  --output data/physics_frames_train.parquet \
  --start 2023-09-02 \
  --end 2024-06-30
```

**If this script doesn't exist yet, ask Claude:**

> I am working on the ARUNA project. Read `python/aruna/l1_ingestion/pipeline.py` and `python/aruna/l2_physics/engine.py` to understand how PhysicsFrames are produced. Then write `python/scripts/generate_training_frames.py` that:
> 1. Walks a directory of SoLEXS and HEL1OS FITS files
> 2. Runs the L1 ingestion pipeline on each pair
> 3. Runs the L2 physics engine to produce PhysicsFrame objects
> 4. Saves them as a parquet file with columns matching the PhysicsFrame dataclass fields
> 5. Accepts --fits-dir, --output, --start, --end arguments

### Step 2 — Join with GOES labels

Match each PhysicsFrame to its GOES class label:

**Claude prompt:**

> I have `data/physics_frames_train.parquet` (PhysicsFrame rows with timestamps) and `data/goes_events.parquet` (GOES flare events). Write `python/scripts/join_labels.py` that:
> 1. For each PhysicsFrame row, looks up whether its timestamp falls within a GOES flare event window (start_time to end_time + 30 min)
> 2. If yes: assigns the GOES class as the label (B=0, C=1, M1-M4=2, M5-M9=3, X=4)
> 3. If no: assigns label 0 (background/B-class)
> 4. Saves the labeled dataset to `data/training_dataset.parquet`

### Step 3 — Write the training loop

**Claude prompt:**

> I am working on the ARUNA project. Read these files:
> - `python/aruna/l4_forecasting/tcn_model.py` (the DualStreamTCN architecture)
> - `python/aruna/l4_forecasting/features.py` (the 27-feature extractor, N_FEATURES=27)
> - `python/aruna/l4_forecasting/models.py` (ForecastConfig with alpha=0.2)
>
> Then write `python/scripts/train_tcn.py` that:
> 1. Loads `data/training_dataset.parquet`
> 2. Creates sliding 600-second windows (600 rows per sample, stride 60)
> 3. Splits: train 2023-09-02 to 2024-06-30, val 2024-07-01 to 2024-12-31 — NO RANDOM SHUFFLING (temporal split only)
> 4. Trains DualStreamTCN with **asymmetric focal loss** where under-predicting M/X events is penalised 4x more than over-predicting (alpha=0.2 from ForecastConfig)
> 5. Uses AdamW optimizer, lr=1e-3, cosine annealing schedule, batch_size=32
> 6. Saves the best validation-loss checkpoint to `models/tcn_v1.pt`
> 7. Prints train/val loss and per-class accuracy every epoch
> 8. Runs for 50 epochs by default

### Step 4 — Run training

```bash
conda activate aruna
python python/scripts/train_tcn.py --epochs 50 --batch-size 32
```

Expected training time:
- GPU (RTX 4060): ~8–12 hours on full dataset
- CPU only: ~48–72 hours (not recommended)
- Synthetic dataset (small): ~30 minutes on CPU

### Step 5 — Verify the trained model

```bash
python -c "
import torch
from aruna.l4_forecasting.tcn_model import DualStreamTCN
from aruna.l4_forecasting.models import ForecastConfig

config = ForecastConfig()
model = DualStreamTCN(config)
model.load_state_dict(torch.load('models/tcn_v1.pt'))
model.eval()
print('Model loaded successfully')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

---

## Part 3 — Fitting the MAPIE Calibrator

After TCN is trained, fit the uncertainty calibrator on the validation set.

**Claude prompt:**

> I am working on the ARUNA project. Read `python/aruna/l4_forecasting/calibrator.py` to understand how the MAPIE conformal calibrator works. Then write `python/scripts/calibrate_mapie.py` that:
> 1. Loads the trained model from `models/tcn_v1.pt`
> 2. Runs it on the validation set (2024-07-01 to 2024-12-31 from `data/training_dataset.parquet`)
> 3. Fits the MAPIE MapieClassifier on the validation predictions (90% coverage target)
> 4. Saves the fitted calibrator to `models/mapie_calibration.pkl`
> 5. Prints the achieved coverage and average interval width

```bash
python python/scripts/calibrate_mapie.py
```

---

## Part 4 — Downloading Moirai

Moirai is a pre-trained zero-shot model from Salesforce. No training needed — just download.

```bash
pip install uni2ts
python -c "from uni2ts.model.moirai import MoiraiForecast; print('Moirai ready')"
```

The first run will download the weights (~2 GB) automatically. After that it works offline.

---

## Part 5 — Building the FAISS Analog Index

The FAISS index is built automatically from the Master Flare Catalog once events are in it. Run the pipeline on any historical FITS data to populate the catalog, then build the index:

```bash
python -c "
from aruna.l5_analogs import AnalogOrchestrator
orch = AnalogOrchestrator()
orch.build_index()
print('FAISS index built')
"
```

The index is saved to `models/analog_index.faiss` automatically.

---

## Part 6 — Validation Targets

After training, verify your model meets these targets from the ARUNA PRD:

### X6.3 event (2024-02-22) — primary anchor

```bash
python -m aruna.replay --event-id SOL2024-02-22-X6.3 --fits-dir python/tests/fixtures/x63/ --speed 1
```

Expected outputs:
| Metric | Target | Tolerance |
|--------|--------|-----------|
| Peak temperature T | 23.5 MK | ±1σ (Sarwade et al. 2025) |
| DNI at peak | > 1.5 | — |
| HOPE fires before peak | ≥ 90 seconds | — |
| GOES classification | X6 | Must not be misclassified as M |

### Forecasting metrics (test set 2025-01-01 to 2025-06-30)

| Metric | Minimum target |
|--------|---------------|
| TSS (True Skill Statistic) | > 0.70 |
| X-class recall | > 0.85 |
| ECE (calibration error) | < 0.05 after MAPIE |
| False alarm rate | < 0.30 |

---

## Part 7 — Files to Create (Summary)

These are all the files your team needs to produce. Use the Claude prompts above, or adapt them.

| File to create | What it does | Claude prompt section |
|---------------|-------------|----------------------|
| `python/scripts/generate_synthetic_fits.py` | Creates synthetic FITS training data | Part 1 Option B |
| `python/scripts/download_goes_labels.py` | Downloads GOES flare event labels | Part 1 |
| `python/scripts/generate_training_frames.py` | Runs L1→L2 pipeline to extract features | Part 2 Step 1 |
| `python/scripts/join_labels.py` | Matches physics frames to GOES classes | Part 2 Step 2 |
| `python/scripts/train_tcn.py` | Trains DualStreamTCN with asymmetric loss | Part 2 Step 3 |
| `python/scripts/calibrate_mapie.py` | Fits MAPIE conformal calibrator | Part 3 |
| `models/tcn_v1.pt` | Trained TCN weights | Output of training |
| `models/mapie_calibration.pkl` | Fitted calibrator | Output of calibration |
| `models/analog_index.faiss` | FAISS retrieval index | Auto-built by L5 |

---

## Quick Start (Everything in Order)

```bash
# 1. Setup environment
conda create -n aruna python=3.11
conda activate aruna
pip install -r python/requirements.txt

# 2. Configure credentials
cp config/config.example.yaml config/config.yaml
# Edit config.yaml: set PRADAN_USERNAME, PRADAN_PASSWORD (or use synthetic data)

# 3. Generate synthetic training data (if no PRADAN access)
#    Ask Claude to write generate_synthetic_fits.py first, then:
python python/scripts/generate_synthetic_fits.py --output-dir data/fits/ --n-events 500

# 4. Download GOES labels
python python/scripts/download_goes_labels.py

# 5. Generate PhysicsFrame features
python python/scripts/generate_training_frames.py \
  --fits-dir data/fits/ \
  --output data/physics_frames_train.parquet \
  --start 2023-09-02 --end 2024-12-31

# 6. Join with GOES labels
python python/scripts/join_labels.py

# 7. Train TCN
python python/scripts/train_tcn.py --epochs 50

# 8. Calibrate MAPIE
python python/scripts/calibrate_mapie.py

# 9. Download Moirai
python -c "from uni2ts.model.moirai import MoiraiForecast"

# 10. Run validation replay
python -m aruna.replay --event-id SOL2024-02-22-X6.3 \
  --fits-dir python/tests/fixtures/x63/ --speed 1

# 11. Run test suite
pytest python/ -v
```

---

## Tips for Using Claude

When working with Claude on this project, always start your prompt with:

> "I am working on the ARUNA solar flare forecasting pipeline (BAH 2026 Problem Statement 15). Read [filename] before writing any code."

This gives Claude the context it needs to write code that matches the existing architecture.

If Claude writes code that breaks existing tests, paste the error and say:

> "This broke existing tests. Read the failing test and fix only the specific issue without changing anything else."

The existing tests in `python/tests/` are the ground truth. Do not change them to make them pass — fix the code instead.

---

## Architecture Reference

The full pipeline processes data in this order:

```
FITS files (SoLEXS + HEL1OS)
      ↓
L1 Ingestion  [dead-time correction, time alignment]
      ↓
L2 Physics    [T/EM fit, DNI, SHEI, HOPE flag]     ← features extracted here
      ↓
L3 Nowcasting [PELT detection, GOES classification]  ← labels come from here
      ↓
L4 Forecasting [DualStreamTCN + Moirai + MAPIE]     ← what you are training
      ↓
L5 Analogs    [FAISS + DTW retrieval]
      ↓
L6 Threat     [NavIC/HF/Gaganyaan risk]
      ↓
L7 LLM        [Azure GPT-4.1-mini operator briefing]
```

Key source files to read before writing any training code:
- `python/aruna/l4_forecasting/tcn_model.py` — the model architecture
- `python/aruna/l4_forecasting/features.py` — the 27-feature extractor
- `python/aruna/l4_forecasting/models.py` — ForecastConfig (α=0.2, temporal split dates)
- `python/aruna/l2_physics/models.py` — PhysicsFrame (the feature source)
- `CLAUDE.md` — full project spec and constraints
