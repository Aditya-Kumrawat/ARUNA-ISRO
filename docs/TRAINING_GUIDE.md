# ARUNA TCN Training Guide

Train `DualStreamTCN` on real Aditya-L1 PRADAN data with NOAA GOES labels.
Follow every step in order.

---

## Project layout (relevant paths)

```
e:\ARUNA\
  python/
    aruna/
      l1_ingestion/
        fits_reader.py        -- reads PRADAN FITS files
        deadtime.py           -- dead-time correction (Sarwade et al. 2025)
        stix_provider.py      -- Solar Orbiter STIX fallback for missing HEL1OS
        hel1os_resolver.py    -- decision tree: PRADAN / STIX / fixtures / skip
      l2_physics/
        gpu_isothermal.py     -- GPU-accelerated isothermal fitter (340 SoLEXS ch)
      l4_forecasting/
        tcn_model.py          -- DualStreamTCN architecture (~1.4M params)
    scripts/
      audit_pradan_fits.py    -- Step 1: check data coverage
      download_goes_labels.py -- Step 2: download GOES flare catalog (labels)
      batch_l1_l2.py          -- Step 3a: CPU physics cache (safe, slower)
      batch_l1_l2_gpu.py      -- Step 3b: GPU physics cache (recommended)
      build_dataset.py        -- Step 4: sliding windows + GOES join + scaler
      train_tcn.py            -- Step 5: train TCN + MAPIE calibration
  data/
    solexs/                   -- PRADAN SoLEXS FITS (AL1_SLX_L1_YYYYMMDD_v1.0)
    hel1os/                   -- PRADAN HEL1OS FITS (AL1_HXS_L1_YYYYMMDD_v1.0)
    physics_cache/            -- output of Step 3 (one Parquet per day)
    dataset/                  -- output of Step 4 (train/val/test Parquet)
    goes_events.parquet       -- output of Step 2
  models/
    tcn_v1.pt                 -- output of Step 5 (TCN weights)
    mapie_calibration.pkl     -- output of Step 5 (conformal calibrator)
    feature_scaler.pkl        -- output of Step 4 (StandardScaler)
    training_log.csv          -- output of Step 5 (per-epoch metrics)
```

---

## Key numbers

| Thing | Value |
|-------|-------|
| SoLEXS channels | 340 (1.5-15 keV, PRADAN Level 1) |
| HEL1OS channels | 256 CdTe (10-100 keV) |
| STIX channels (fallback) | 32 (4-150 keV) → mapped to 4 HEL1OS-equivalent bins |
| TCN sequence length | 600 frames = 10 minutes at 1-second cadence |
| TCN inputs | thermal(9) + nonthermal(8) + shared(10) per timestep |
| TCN output | 5-class probability (QUIET/B/C/M/X) |
| Temporal split | Train: 2023-09-02 to 2024-06-30 / Val: 2024-07-01 to 2024-12-31 / Test: 2025-01-01 to 2025-06-30 |
| Loss function | Asymmetric focal loss alpha=0.2 gamma=2.0 (M/X penalized 4x) |
| Target TSS | > 0.60 on validation set |
| Target ECE | < 0.05 after MAPIE calibration |

---

## Prerequisites

```bash
conda activate aruna
pip install mapie sunpy stixdcpy astropy torch pandas pyarrow scikit-learn
```

Verify:
```bash
python -c "import torch, mapie, stixdcpy, sunpy, astropy; print('All OK')"
```

Place downloaded PRADAN files in:
- `data/solexs/`  -- filenames like `AL1_SLX_L1_20240222_v1.0` (one folder per day)
- `data/hel1os/`  -- filenames like `AL1_HXS_L1_20240222_v1.0`

---

## Step 1 -- Audit data coverage (5 minutes)

```bash
python python/scripts/audit_pradan_fits.py \
  --solexs-dir data/solexs/ \
  --hel1os-dir data/hel1os/
```

Expected output:
```
DATE          SoLEXS    HEL1OS    STATUS
2024-02-22    OK        MISSING   WARN -- HEL1OS missing (STIX fallback available)
2024-02-23    OK        OK        OK -- dual-instrument

Total dates found: 800
  Dual-instrument : 720  (90.0%)  <- usable for training
  SoLEXS only     :  25
  HEL1OS only     :  55

PRD temporal split alignment:
  TRAIN (2023-09-02 to 2024-06-30): 312 days
  VAL   (2024-07-01 to 2024-12-31): 184 days
  TEST  (2025-01-01 to 2025-06-30): 145 days
```

**If TRAIN < 200 days:** download more from PRADAN before continuing.

---

## Step 2 -- Download GOES flare labels (5 minutes)

```bash
python python/scripts/download_goes_labels.py \
  --start 2023-09-01 \
  --end   2026-06-20 \
  --out   data/goes_events.parquet
```

This downloads the NOAA GOES XRS flare event catalog. These are the **labels only** --
ARUNA never uses GOES at inference. If HEK is down, the script falls back to NCEI CSV.

Expected: `Saved 847 events -> data/goes_events.parquet`

---

## Step 3 -- Build physics cache (20-60 min GPU / 2-5 days CPU)

This runs L1 ingestion + L2 physics on every FITS file and caches to Parquet.
**Run only once.** If interrupted, re-run -- already-cached days are skipped.

### With GPU (recommended):

```bash
python python/scripts/batch_l1_l2_gpu.py \
  --solexs-dir data/solexs/ \
  --hel1os-dir data/hel1os/ \
  --output-dir data/physics_cache/ \
  --device     cuda
```

### CPU fallback (no GPU):

```bash
python python/scripts/batch_l1_l2.py \
  --solexs-dir data/solexs/ \
  --hel1os-dir data/hel1os/ \
  --output-dir data/physics_cache/ \
  --workers    8
```

### Missing HEL1OS handling

The GPU script applies this decision tree automatically:

```
HEL1OS file found on PRADAN?
  YES -> use it (normal path)
  NO  -> Is GOES class M or X on this date?
           YES -> Try STIX (Solar Orbiter) fallback via stixdcpy
                    STIX works -> use it (tagged STIX:YYYY-MM-DD)
                    STIX fails -> is it 2024-02-22 (X6.3)?
                                    YES -> use offline test fixtures
                                    NO  -> skip date (log WARNING)
           NO  (quiet/B/C) -> skip date (quiet days are abundant)
```

The X6.3 2024-02-22 case is validated: STIX returned 20,517 frames, SHEI
range -4.14 to +0.64, covering the full impulsive phase.

Expected output:
```
2024-02-22 -- 20515 frames written  [source: STIX:2024-02-22]
2024-02-23 -- 86400 frames written
...
Done. Success=698  Failed=22  Skipped=0
```

Output: `data/physics_cache/YYYY-MM-DD.parquet` (~7-8 GB total for 2 years)

---

## Step 4 -- Build windowed training dataset (30 minutes)

```bash
python python/scripts/build_dataset.py \
  --cache-dir   data/physics_cache/ \
  --goes-file   data/goes_events.parquet \
  --output-dir  data/dataset/ \
  --scaler-out  models/feature_scaler.pkl \
  --horizon-min 30 \
  --stride-sec  60
```

This slides 600-frame (10-minute) windows over the physics cache, labels each
window with the highest GOES class in the next 30 minutes, applies the PRD
temporal split, and fits a StandardScaler on train windows only.

Expected output:
```
TRAIN (312,450 windows):
  QUIET (0): 294,103  (94.0%)
  B     (1):   9,373  ( 3.0%)
  C     (2):   6,249  ( 2.0%)
  M     (3):   2,186  ( 0.7%)
  X     (4):     539  ( 0.2%)
VAL   (121,200 windows) ...
TEST  ( 89,700 windows) ...
Scaler saved -> models/feature_scaler.pkl
```

---

## Step 5 -- Train the TCN (6-12 hours GPU / 3-7 days CPU)

```bash
# GPU (recommended)
python python/scripts/train_tcn.py \
  --dataset-dir data/dataset/ \
  --scaler-path models/feature_scaler.pkl \
  --tcn-out     models/tcn_v1.pt \
  --mapie-out   models/mapie_calibration.pkl \
  --epochs      100 \
  --batch-size  64 \
  --lr          1e-3 \
  --device      cuda

# CPU
python python/scripts/train_tcn.py \
  --device cpu --batch-size 32
```

Per-epoch log:
```
Epoch   1/100  loss=1.4231  val_loss=1.3892  TSS=0.1203  ECE=0.2841  lr=1.00e-03  142s
Epoch   2/100  loss=1.2104  val_loss=1.1876  TSS=0.2451  ECE=0.2103  lr=9.99e-04  139s
...
Epoch  47/100  loss=0.3421  val_loss=0.4102  TSS=0.6834  ECE=0.0412  lr=4.12e-04  141s
  -> New best TSS=0.6834 -- saved models/tcn_v1.pt
...
Early stopping: no TSS improvement for 10 epochs

TRAINING COMPLETE
  Best validation TSS : 0.6834  (target > 0.60)
  TCN weights         : models/tcn_v1.pt
  MAPIE calibrator    : models/mapie_calibration.pkl
  Training log        : models/training_log.csv
```

The script saves `models/tcn_v1.pt` every time TSS improves. If training
crashes, re-run -- it resumes from the last saved checkpoint automatically.

---

## Step 6 -- Verify with X6.3 replay

```bash
python -m aruna.pipeline.run --fits-dir python/tests/fixtures/x63/
```

If the pipeline outputs `ForecastFrame` with `prob_X > 0.5` at ~22:34 UTC
on 2024-02-22, training succeeded.

---

## Acceptance criteria (PRD §8.10)

| Metric | Target | Where to check |
|--------|--------|----------------|
| Validation TSS | > 0.60 | `models/training_log.csv` last best row |
| ECE | < 0.05 | printed after MAPIE step |
| Under-prediction M5+ | < 5% | `training_log.csv` |
| Inference latency | < 10s | `ForecastOrchestrator` logs elapsed time |

---

## Troubleshooting

**`KeyError: EBOUNDS` in Step 3**
The FITS file is missing its energy calibration extension. The day is skipped
automatically. Normal for ~3% of files.

**`TSS stuck below 0.2` after 20 epochs**
Class imbalance winning. Add `--batch-size 32` (smaller batches see M/X
windows more frequently per weight update).

**CUDA out of memory**
Lower `--batch-size` to 32 or 16 in Step 5. The TCN is small but 600-frame
sequences use ~2 GB at batch-size 64.

**`STIX fetch failed` in Step 3**
The stixdcpy API is temporarily down. The day is skipped with a WARNING.
Re-run the script the next day -- it will retry skipped days (not in cache).

**MAPIE calibration skipped**
Run `pip install mapie scikit-learn` and re-run `train_tcn.py`. The TCN
weights are already saved; MAPIE is a post-hoc calibration step.

**`No module named aruna`**
Run scripts from the repo root (`e:\ARUNA`), not from inside `python/scripts/`.

---

## Output files summary

| File | Purpose | Required for inference |
|------|---------|----------------------|
| `models/tcn_v1.pt` | TCN weights | YES |
| `models/mapie_calibration.pkl` | Conformal calibrator | Recommended |
| `models/feature_scaler.pkl` | StandardScaler | YES |
| `models/training_log.csv` | Per-epoch metrics | No (debug only) |
| `data/physics_cache/` | Daily Parquet (keep!) | No (re-training only) |
| `data/goes_events.parquet` | GOES labels (keep!) | No (re-training only) |

Once `tcn_v1.pt` and `feature_scaler.pkl` exist, the live pipeline
(`python -m aruna.pipeline.run`) loads them automatically.
