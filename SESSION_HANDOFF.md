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
- Colab clone path: `/content/aruna_repo`. Pull latest before doing anything.
- Three-tier: `python/` (science, FastAPI), `api/` (Node/Fastify gateway), `web/` (React+Vite). Streamlit is NOT the frontend.
- **L3 nowcasting is pure algorithm — NO GPU.** Runs locally or in the Colab terminal directly.
- **L4 (TCN) needs the Colab GPU.**
- `models/Aruna.ipynb` is the Colab notebook.

## Data (critical)
- `data/dataset/{train,val,test}.parquet` — event-level resplit windowed dataset (600×19 features per window). **NEVER modify val.parquet or test.parquet** — unbiased eval.
- `train_balanced_augmented.parquet` — rebalanced train (QUIET downsampled to ~2× flares, X aug 2×).
- `physics_cache/physics_cache/*.parquet` — **165 daily L2 PhysicsFrame files** (2024-02 → 2026-06, includes the X6.3 day **2024-02-22**). Schema verified: timestamp, flare_phase, hope_flag, + 19 physics features. Transferred to Colab via Drive `physics_cache.zip` → unzip to `data/physics_cache/`.
- `goes_events.parquet` — GOES flare catalogue = **ground truth**. Regenerate with `python python/scripts/download_goes_labels.py --start 2023-09-01 --end 2026-06-20 --out data/goes_events.parquet` (needs internet).
- Raw FITS: only 2 recent quiet days locally; **X6.3 raw FITS fixture is EMPTY** (so no true raw→L2 chain on X6.3 — start from physics_cache).
- Data ceiling: ~161 flare events total (112 train / 24 val / 25 test), ~20 X events. Per-class X accuracy is **data-capped** — not a code bug.

## What's DONE
- Python pipeline L0–L7 implemented (261 tests pass). API gateway done. React Mission Control page built; other ~13 pages are stubs.
- **L4 TCN forecaster: trained + validated.** Locked operating policy in `config/config.yaml` → `forecasting.alert_policy`:
  - `mx_threshold: 0.01`, `persistence_windows: 10`, `temperature: 1.9669`
  - Honest test result (event-level, n=12 M+X events): **58% event recall (7/12, 95% CI [25%,83%]), 0.17 false alarms/day, 23.7 min mean lead.** Window-level TSS ≈ 0.39, ECE ≈ 0.008–0.02.
  - **Ensemble + physics gate were DROPPED** (both hurt: XGB ~0.20 TSS dragged the ensemble; physics gate vetoed valid detections).
  - Threshold is **selected on val, frozen, reported on test** (no leakage). Headline = event-level operational metrics, NOT TSS.
- Model artifacts: `models/tcn_v1.pt`, `feature_scaler_19.pkl`, `tcn_v1.threshold.npy`, `tcn_temperature.npy`.

## THE REFRAME (current focus)
The PS wants BOTH nowcast (L3) and forecast (L4). We over-invested in the L4 TCN.
**L3 nowcasting is implemented but NEVER validated end-to-end** — and the user's
goals (accurate class detection, missing-data robustness, low FP+FN) live in L3,
not the TCN:
- Class accuracy comes from L3's **deterministic GOES flux classification**, not the TCN's weak softmax (M=13%, X=0%).
- Missing-data robustness is **native to L3's independent-per-instrument design**.

## L3 architecture (`python/aruna/l3_nowcasting/`)
- `changepoint.py` — PELT changepoint detection (per instrument)
- `baseline.py` — rolling P85 baseline → trigger threshold
- `consensus.py` — dual-trigger FSM (QUIET→HOPE→ONSET→PEAK→QUIET) combining SoLEXS+HEL1OS → NowcastEvents
- `goes_class.py` — flux → GOES class (B/C/M/X) by thresholds
- `catalog.py` — master catalogue store
- `orchestrator.py` — wires all of it; `process(PhysicsFrame)` one frame at a time

## AGREED PLAN — 3 stages (do in order)
**Stage 1 (PRIORITY): Validate L3 end-to-end.** Build `python/scripts/validate_l3.py` (runs locally, no GPU):
1. Adapt physics_cache rows → `PhysicsFrame`, stream day-by-day into `NowcastOrchestrator.process()`, collect emitted events = master catalogue.
2. Match L3 FLARE_ONSET/PEAK to GOES events by time (peak within tolerance).
3. Report PS metrics: detection recall/TPR (split low B/C vs high M/X), precision & FA/day, detection delay, **GOES classification confusion matrix**, catalogue completeness.
4. Missing-instrument runs: HEL1OS zeroed (expect SOLEXS_ONLY still fires) and SoLEXS zeroed (currently fails → proves B2).
Then **fix blockers B1–B4** and re-run.

**Stage 2: Integrate L3→L4** end-to-end (physics_cache → detect → consensus → classify → master catalogue → forecast → threat). Measure latency + robustness.

**Stage 3: Improve L4 TCN** — ONLY after 1 & 2. Already at 58% recall; bounded gains.

## 4 BLOCKERS found in L3 (fix in Stage 1 — these ARE the user's goals)
- **B1 — Classification uncalibrated** (`goes_class.py`): `solexs_to_wm2_factor = 1.0e-8` is a placeholder → all B/C/M/X labels systematically wrong. **Fix: calibrate the scalar against GOES matched events.** (= "classify any class accurately")
- **B2 — Missing-data asymmetric** (`consensus.py`): has `SOLEXS_ONLY` advisory but NO `HEL1OS_ONLY` → if SoLEXS drops, HEL1OS alone emits nothing. **Fix: add symmetric HEL1OS-only mode.** (= "vice versa if anything goes missing")
- **B3 — HEL1OS not independently detected** (`orchestrator.py`): HEL "trigger" is a proxy (`dSHEI_dt>0 AND SHEI>-1.0`); the HEL changepoint result is computed then DISCARDED. PS wants independent detection in both. **Fix: use HEL changepoint as a true trigger.**
- **B4 — Timing assumes gapless data** (`orchestrator.py`): `t_sec = frame_count` → data gaps desync baseline/timing. **Fix: use real frame timestamps.**

## User's goals (verbatim intent)
Very precise detection of any flare class from SoLEXS+HEL1OS, and vice versa if
one instrument's data is missing; minimize **both** false positives (false
alarms) AND false negatives (missed flares); test the system end-to-end. Use the
Colab GPU for train/debug/test of the model + weights.

## Workflow constraints
- Edit `.py` scripts in repo → commit/push → `git pull` in Colab → run. (Or edit directly in Colab terminal now.)
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`
- Read `design.md` before any UI work (brand/colors/copy source of truth).

## IMMEDIATE NEXT STEP
Build `python/scripts/validate_l3.py` (Stage 1 harness), run it locally/Colab to
get the **honest L3 baseline vs GOES**, then fix B1–B4 and re-run. We were
mid-brainstorm with the Stage 1 design presented and awaiting approval on:
(a) score against GOES as ground truth, (b) fixing B1–B4 is in-scope for Stage 1.
