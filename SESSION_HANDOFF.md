# ARUNA — Session Handoff

Context dump for continuing in a new (Colab terminal) session. Read this first.

## What ARUNA is
Aditya-L1 Real-time Unified Nowcasting & Alert System. ISRO BAH 2026 **Problem
Statement 15**: use combined **SoLEXS (soft X-ray)** + **HEL1OS (hard X-ray)**
time series to **nowcast** (real-time detect + classify flares, independently
per instrument → combined master catalogue) and **forecast** (precursor-based
prediction with quantifiable lead time). Eval criteria: detect **low- and
high-class** flares, **high TPR / low FAR**, **lead time** before peak. Also: a
UI showing light curves + visual alerts.

## Repo / environment
- GitHub: `github.com/Aditya-Kumrawat/ARUNA-ISRO`, branch `master`.
- Colab clone path: `/content/ARUNA-ISRO`. Pull latest before doing anything:
  `git -C /content/ARUNA-ISRO pull`
- Three-tier: `python/` (science, FastAPI), `api/` (Node/Fastify gateway), `web/` (React+Vite).
- **L3 nowcasting is pure algorithm — NO GPU.** Runs locally or in the Colab terminal directly.
- **L4 (TCN) needs the Colab GPU.**
- `models/Aruna.ipynb` is the Colab notebook.
- Install: `pip install -r /content/ARUNA-ISRO/python/requirements.txt` in Colab

## Data (critical)
- `data/physics_cache/` — **165 daily L2 PhysicsFrame parquet files** (2024-02 → 2026-06).
  **MUST be transferred from Drive before running validate_l3.py or aruna_loop.py:**
  ```
  # In Colab:
  from google.colab import drive; drive.mount('/drive')
  !cp /drive/MyDrive/physics_cache.zip /content/ARUNA-ISRO/data/
  !unzip /content/ARUNA-ISRO/data/physics_cache.zip -d /content/ARUNA-ISRO/data/
  ```
- `data/goes_events.parquet` — GOES flare catalogue = ground truth.
  Regenerate if missing:
  ```
  python python/scripts/download_goes_labels.py \
      --start 2023-09-01 --end 2026-06-20 \
      --out data/goes_events.parquet
  ```
- `data/dataset/{train,val,test}.parquet` — event-level windowed dataset (600×19 features).
  **NEVER modify val.parquet or test.parquet.**
- `train_balanced_augmented.parquet` — rebalanced train.

## What's DONE (as of this session)
- Python pipeline L0–L7 implemented (261 tests pass). API gateway done. React Mission Control stub.
- **L4 TCN forecaster: trained + validated.**
  - Test (event-level, n=12 M+X events): **58% event recall (7/12), 0.17 FA/day, 23.7 min mean lead.**
  - Window-level TSS ≈ 0.39, ECE ≈ 0.008–0.02.
  - Operating policy frozen in `config/config.yaml`.
- **B1–B4 L3 blockers: ALL FIXED in this session:**
  - **B1**: `goes_class.py` now has `calibrate_from_matched_events()` class method. Pass
    `--calibrate` to `validate_l3.py` to fit the `solexs_to_wm2_factor` from data.
  - **B2**: `consensus.py` now has symmetric `HEL1OS_ONLY` degraded mode (mirrors `SOLEXS_ONLY`).
    `DETECTION_METHODS` in `models/nowcast.py` updated to include `"HEL1OS_ONLY"`.
  - **B3**: `orchestrator.py` now uses `hel_cp = self._hel_cp.update(flux_hel)` as a true
    independent HEL trigger (with SHEI proxy as fallback). Previously hel_cp result was discarded.
  - **B4**: `orchestrator.py` now uses real frame timestamps for baseline timing
    (`t_sec = int((frame.timestamp - self._epoch_time).total_seconds()) + 1`). `_epoch_time`
    set on first frame.

## Two new scripts built this session

### `python/scripts/validate_l3.py` — L3 end-to-end validation harness
```bash
# Full validation (requires physics_cache + goes_events in place)
cd /content/ARUNA-ISRO
python python/scripts/validate_l3.py \
    --physics-cache data/physics_cache \
    --goes-events data/goes_events.parquet \
    --out experiments/l3_baseline

# With B1 calibration
python python/scripts/validate_l3.py ... --calibrate

# Test B2/B3 fixes (instrument dropout)
python python/scripts/validate_l3.py ... --zero-hel1os
python python/scripts/validate_l3.py ... --zero-solexs

# All three modes at once + comparison table
python python/scripts/validate_l3.py ... --all-modes
```
**Outputs per run:** `metrics.json`, `report.txt`, `confusion.txt`, `false_alarms.csv`, `missed_events.csv`

### `python/scripts/aruna_loop.py` — Autonomous experiment loop
```bash
# Step 1: Freeze baseline (run ONCE, do not re-run without intent to reset)
cd /content/ARUNA-ISRO
python python/scripts/aruna_loop.py freeze-baseline \
    --physics-cache data/physics_cache \
    --goes-events data/goes_events.parquet

# Step 2: Run experiments (resumes from checkpoint after Colab restart)
python python/scripts/aruna_loop.py run \
    --physics-cache data/physics_cache \
    --goes-events data/goes_events.parquet \
    --max-experiments 3   # omit to run all pending

# Check status
python python/scripts/aruna_loop.py status

# Add a custom experiment
python python/scripts/aruna_loop.py add \
    --name "lower PELT min_size" \
    --hypothesis "Smaller min_size detects shorter-duration B events" \
    --config-patch '{"pelt_min_size": 5}'
```
**State is checkpointed to `experiments/loop_state.json` after every experiment.**
**Colab restart safe: just re-run `run` and it picks up where it stopped.**

## Experiment backlog (8 pre-loaded)
```
exp_001  □ B1 SoLEXS calibration          → metric: goes_class_accuracy
exp_002  □ B2 HEL1OS-only degraded mode   → metric: recall_total (zero-solexs mode)
exp_003  □ B3 HEL changepoint trigger     → metric: recall_high_class_MX
exp_004  □ Wider dual-trigger window      → NowcastConfig.dual_trigger_window_sec=300
exp_005  □ Lower sigma multiplier (3→2)   → NowcastConfig.trigger_sigma_multiplier=2.0
exp_006  □ Narrower PELT penalty (1.5→0.8)→ NowcastConfig.pelt_penalty=0.8
exp_007  □ Wider GOES match window        → MATCH_WINDOW_MIN=15
exp_008  □ Peak decay relaxation (0.5→0.3)→ NowcastConfig.peak_decay_threshold=0.3
```

## Scoring function (loop objective)
```
PRIMARY:   Maximise recall_high_class_MX  (M+X recall)
HARD CONSTRAINTS:
  false_alarms_per_day < 1.0
  goes_class_accuracy > 0.50
SECONDARY (small weights):
  + 0.05 × recall_total
  + 0.02 × (1 - lead_time_mean_min / 60)
```
An experiment is KEPT only if its score exceeds the current baseline AND constraints pass.

## AGREED PLAN — 3 stages (do in order)
**Stage 1 (PRIORITY): Validate L3 end-to-end.** ← WE ARE HERE
1. Transfer physics_cache.zip from Drive.
2. Run `validate_l3.py --all-modes --calibrate`.
3. Read the GOES class accuracy and recall numbers.
4. Then freeze baseline with `aruna_loop.py freeze-baseline`.
5. Run `aruna_loop.py run --max-experiments 3` to step through exp_001–exp_003 (the B-fix experiments).
6. Commit results to git.

**Stage 2: Integrate L3→L4** — measure end-to-end latency + robustness.

**Stage 3: Improve L4 TCN** — only after Stages 1 & 2. Already at 58% recall; bounded gains.

## L3 architecture (`python/aruna/l3_nowcasting/`)
- `changepoint.py` — PELT changepoint detection (per instrument)
- `baseline.py` — rolling P85 baseline → trigger threshold
- `consensus.py` — dual-trigger FSM (QUIET→HOPE→ONSET→PEAK→QUIET); now with HEL1OS_ONLY
- `goes_class.py` — flux → GOES class; now has `calibrate_from_matched_events()`
- `catalog.py` — master catalogue store (SQLite)
- `orchestrator.py` — wires all of it; B3+B4 fixed; `process(PhysicsFrame)` one frame at a time

## User's goals (verbatim intent)
Very precise detection of any flare class from SoLEXS+HEL1OS, and vice versa if
one instrument's data is missing; minimize **both** false positives (false
alarms) AND false negatives (missed flares); test the system end-to-end. Use the
Colab GPU for train/debug/test of the model + weights.

## Workflow constraints
- Edit `.py` scripts in repo → commit/push → `git pull` in Colab → run. (Or edit directly in Colab terminal.)
- Commit messages end with: `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>`
- Read `design.md` before any UI work.

## IMMEDIATE NEXT STEP (pick up from here)
```bash
# In Colab terminal:
git -C /content/ARUNA-ISRO pull

# Transfer data (if not already done):
# from google.colab import drive; drive.mount('/drive')
# !cp /drive/MyDrive/physics_cache.zip /content/ARUNA-ISRO/data/
# !unzip /content/ARUNA-ISRO/data/physics_cache.zip -d /content/ARUNA-ISRO/data/

# Then:
cd /content/ARUNA-ISRO
python python/scripts/validate_l3.py \
    --physics-cache data/physics_cache \
    --goes-events data/goes_events.parquet \
    --out experiments/l3_first_run \
    --all-modes --calibrate

# Then freeze baseline and start loop:
python python/scripts/aruna_loop.py freeze-baseline \
    --physics-cache data/physics_cache \
    --goes-events data/goes_events.parquet

python python/scripts/aruna_loop.py run \
    --physics-cache data/physics_cache \
    --goes-events data/goes_events.parquet \
    --max-experiments 3
```
