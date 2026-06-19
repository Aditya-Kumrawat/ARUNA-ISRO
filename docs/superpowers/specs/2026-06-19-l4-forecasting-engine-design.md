# L4 Forecasting Engine — Design Spec

**Date:** 2026-06-19
**Phase:** 5
**Status:** Approved — ready for implementation
**PRD reference:** §8 (L4 — Forecasting Engine)
**Prerequisite:** L3 Nowcasting Engine complete (41 tests passing, 11 commits, Session 5)

---

## 1. Objective

Consume a sliding 10-minute window (600 frames) of `PhysicsFrame` objects from L2/L3, run a
dual-stream TCN + Moirai zero-shot ensemble, calibrate outputs with MAPIE conformal prediction,
and publish `ForecastFrame` objects to `aruna.forecast.frames` for every horizon in
`[5, 10, 15, 30, 60]` minutes.

This is the predictive layer — the difference between detecting what is happening (L3) and
warning **before** it happens.

---

## 2. Architecture Decision — Why This Stack

### Option A: Full TCN + Moirai ensemble with MAPIE (PRD-specified) ✅ CHOSEN
**Advantages:**
- TCN learns causal temporal patterns from 600-frame windows (10 min × 1 s cadence)
- Moirai zero-shot handles the small-data regime without fine-tuning
- MAPIE conformal prediction gives statistically guaranteed 90% coverage intervals
- Asymmetric loss (α=0.2) bakes in the operationally critical under-prediction penalty
- Cross-attention between thermal/non-thermal streams learns the HOPE lead-time implicitly

**Disadvantages:**
- Two models to load at runtime (memory pressure: ~500 MB combined)
- Moirai requires `uni2ts` — a heavy dependency with its own install footprint
- Online inference latency depends on model size

**Mitigations:**
- Moirai is loaded lazily with graceful degradation (`moirai_available` flag)
- TCN is a lightweight custom model; `models/tcn_v1.pt` is the primary path
- MAPIE operates on the ensemble output, not on raw logits — fast at inference

### Option B: TCN only, skip Moirai
Rejected. The xfail `test_ensemble_beats_best_single_model_by_tss_0_03` explicitly
requires ensemble > single model. Moirai provides complementary signal in the small-data
regime where TCN may overfit.

### Option C: scikit-learn gradient boost
Rejected. Cannot model temporal dependencies in the 600-frame sequence natively.
Architecturally inconsistent with the dual-stream PhysicsFrame signal structure.

---

## 3. Critical Non-Negotiable Design Constraints

These come directly from GEMINI.md and PRD §8 — deviation is not permitted:

1. **Asymmetric loss α=0.2** — `loss = α * over_pred² + (1-α) * under_pred²`.
   Under-predicting M/X is penalised 4× more than over-predicting. **Never substitute MSE.**

2. **Temporal split only** — Train 2023-09-02→2024-06-30, Val 2024-07-01→2024-12-31,
   Test 2025-01-01→2025-06-30. No random shuffling. 24-hour buffer between splits.

3. **Moirai context clip fallback** — When context > 512 steps: clip to last 512,
   raise `tcn_weight` to 0.7, lower `moirai_weight` to 0.3. Log the truncation.

4. **X3+ confidence forced to 1.0** — Regardless of model output; mirrors L3 behaviour.

5. **ForecastFrame already defined** in `aruna/models/forecast.py` — import it, never redefine.

6. **No model weights committed to git** — `models/tcn_v1.pt` is excluded via `.gitignore`.
   Tests use a randomly-initialised TCN (same architecture) as a deterministic stand-in.

---

## 4. Data Flow

```
PhysicsFrame stream (from Kafka or in-memory)
  → FrameBuffer (deque, maxlen=600)
      │
      ├─ FeatureExtractor
      │      → (tensor_thermal, tensor_nonthermal, tensor_shared)    # (B, T, F_i)
      │
      ├─ DualStreamTCN.forward(...)
      │      → logits (B, 5)                                          # 5-class
      │
      ├─ MoiraiForecaster.predict(context)   [optional; skipped if unavailable]
      │      → moirai_probs (B, 5)
      │
      ├─ EnsembleWeighter.combine(tcn_probs, moirai_probs)
      │      → ensemble_probs (B, 5)
      │
      ├─ MapieCalibrator.calibrate(ensemble_probs)
      │      → calibrated_probs, prediction_sets, ece
      │
      └─ ForecastOrchestrator.assemble(...)
             → ForecastFrame (one per horizon)
             → Kafka publish: aruna.forecast.frames
```

---

## 5. Module Map

```
python/aruna/l4_forecasting/
  __init__.py          # public re-exports
  models.py            # ForecastConfig (all thresholds from config.yaml)
  features.py          # FeatureExtractor — PhysicsFrame window → 3 tensors
  tcn_model.py         # DualStreamTCN + TemporalBlock + loader
  moirai_model.py      # MoiraiForecaster — lazy import, graceful degradation
  ensemble.py          # EnsembleWeighter — weighted combination + clip fallback logic
  calibrator.py        # MapieCalibrator — MAPIE conformal wrapper
  forecaster.py        # ForecastOrchestrator — stateful entry point, publishes ForecastFrame
```

---

## 6. Module Specifications

### 6.1 `models.py` — ForecastConfig

All thresholds from `config.yaml forecasting:` block. No magic numbers in module files.

```python
@dataclass
class ForecastConfig:
    sequence_length: int = 600              # 10-min window at 1s cadence
    forecast_horizons: tuple = (5, 10, 15, 30, 60)  # minutes
    n_features: int = 24

    # Ensemble weights
    tcn_weight: float = 0.5
    moirai_weight: float = 0.5
    moirai_context_length: int = 512        # clip threshold
    moirai_fallback_tcn_weight: float = 0.7

    # Loss
    alpha: float = 0.2                      # asymmetric loss under-pred weight
    gamma: float = 2.0                      # focal loss

    # MAPIE
    mapie_coverage: float = 0.90
    mapie_alpha: float = 0.10

    # Model paths
    tcn_weights_path: str = "models/tcn_v1.pt"
    mapie_calibration_path: str = "models/mapie_calibration.pkl"

    # Temporal split (read-only reference — used by split validator only)
    train_start: str = "2023-09-02"
    train_end:   str = "2024-06-30"
    val_start:   str = "2024-07-01"
    val_end:     str = "2024-12-31"
    test_start:  str = "2025-01-01"
    test_end:    str = "2025-06-30"
    buffer_hours: int = 24

    # Kafka
    kafka_topic_input:  str = "aruna.nowcast.events"
    kafka_topic_output: str = "aruna.forecast.frames"

    # Pipeline metadata
    pipeline_version: str = "2.0.0"
    model_version: str = "tcn_v1+moirai_base"
```

### 6.2 `features.py` — FeatureExtractor

Maps a list of `PhysicsFrame` objects (the sliding window) into three PyTorch tensors.

**Feature split (matching PRD §8.3, N=24 total):**
- `tensor_thermal` (8 features): `T_MK, EM_cm3, DNI, tnpsv_magnitude, tnpsv_angle_rad, dT_dt, dEM_dt, d2T_dt2`
- `tensor_nonthermal` (8 features): `SHEI, dSHEI_dt, FeKa_flux, hel1os_10_25kev, hel1os_25_50kev, hel1os_50_70kev, QPP_period_sec, QPP_amplitude`
- `tensor_shared` (8 features): `HFD, solexs_3_6kev, solexs_6_10kev` + 5 one-hot `flare_phase` flags

**NaN handling:**
- `HFD`: NaN during warm-up (<600 frames) — replace with `0.0`; HFD is an enrichment feature, not a core trigger
- `QPP_period_sec`, `QPP_amplitude`: `None` when no pulsation — replace with `0.0`
- Any NaN in plasma diagnostics (fit failed): propagate `0.0` with a flag tracked in `ForecastFrame.physics_state`

**Normalisation:**
- Apply per-feature `StandardScaler` fitted on the training split
- Scaler parameters stored as `models/feature_scaler.pkl`
- `FeatureExtractor(scaler=None)` → raw counts (for tests without a trained scaler)

```python
class FeatureExtractor:
    def __init__(self, scaler=None): ...

    def extract(
        self,
        frames: list[PhysicsFrame]   # len must equal sequence_length
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (thermal, nonthermal, shared), each shape (1, T, F_i)."""
```

### 6.3 `tcn_model.py` — DualStreamTCN + loader

Implements the exact PRD §8.4 architecture. **Key details:**

**`TemporalBlock`:** Standard TCN residual block:
- `Conv1d(in, out, kernel_size, dilation, padding=same)` → BatchNorm → ReLU → Dropout
- Residual projection when `in_channels != out_channels`

**`DualStreamTCN`:**
- `tcn_thermal`: channels `[64, 64, 128, 128]`, kernel 7
- `tcn_nonthermal`: channels `[64, 64, 128, 128]`, kernel 7
- `cross_attention`: `nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)`
- `shared_encoder`: channels `[32, 64]`, kernel 5
- `classifier`: `Linear(320, 256) → ReLU → Dropout(0.3) → Linear(256, 5)`
- No Softmax in forward — applied externally to allow asymmetric loss

**Model loader:**
```python
def load_tcn(path: str, device: str = "cpu") -> DualStreamTCN:
    """
    Load weights from path. If path not found (test environment),
    return a randomly-initialised model in eval() mode.
    Logs at WARNING level when weights file is absent.
    """
```

### 6.4 `moirai_model.py` — MoiraiForecaster

**Lazy import strategy:** `uni2ts` is expensive. Import inside the class constructor,
catch `ImportError`, set `self._available = False`.

```python
class MoiraiForecaster:
    def __init__(self, context_length: int = 512): ...

    @property
    def available(self) -> bool: ...

    def predict(
        self,
        feature_history: np.ndarray,    # shape (T, N_features) — single feature used
        context_length: int = 512,
    ) -> np.ndarray:
        """
        Returns 5-class probability vector shape (5,).
        If unavailable: returns uniform [0.2, 0.2, 0.2, 0.2, 0.2].
        If context > 512: clip to last 512, log MOIRAI_CONTEXT_CLIP.
        """
```

**What Moirai actually forecasts:** DNI trajectory (single time-series input) — the simplest
valid usage of Moirai-1.0-R-base. The output distribution is mapped to the 5-class GOES
probability space via a linear calibration layer fitted during training.

### 6.5 `ensemble.py` — EnsembleWeighter

Pure function module. No state.

```python
class EnsembleWeighter:
    def __init__(self, config: ForecastConfig): ...

    def combine(
        self,
        tcn_probs: np.ndarray,          # shape (5,)
        moirai_probs: np.ndarray,        # shape (5,) — or uniform if unavailable
        context_was_clipped: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """
        Returns (ensemble_probs, effective_tcn_weight, effective_moirai_weight).
        If context_was_clipped: tcn_weight → 0.7, moirai_weight → 0.3.
        """
```

### 6.6 `calibrator.py` — MapieCalibrator

Thin wrapper around `mapie.classification.MapieClassifier`.

```python
class MapieCalibrator:
    def __init__(self, config: ForecastConfig): ...

    def load(self, path: str) -> None:
        """Load fitted MAPIE from pickle. If not found: operate in uncalibrated mode."""

    def calibrate(
        self,
        probs: np.ndarray,              # shape (5,)
    ) -> tuple[np.ndarray, list[str], float]:
        """
        Returns:
          calibrated_probs: shape (5,)
          prediction_set:   list of class labels in the 90% conformal set
          ece:              expected calibration error (scalar)

        If no calibrator loaded: return probs unchanged, prediction_set = [argmax_class], ece = 0.0.
        """
```

### 6.7 `forecaster.py` — ForecastOrchestrator

Stateful. Owns the frame buffer and all sub-modules. Single entry point.

```python
class ForecastOrchestrator:
    def __init__(
        self,
        config: ForecastConfig | None = None,
        kafka_producer=None,
    ): ...

    def process(
        self,
        frame: PhysicsFrame,
        nowcast_event: NowcastEvent | None = None,
    ) -> list[ForecastFrame] | None:
        """
        Append frame to internal deque(maxlen=600).
        When buffer has >= sequence_length frames:
          1. FeatureExtractor.extract(buffer)
          2. TCN forward pass
          3. Moirai predict (if available)
          4. EnsembleWeighter.combine(...)
          5. MapieCalibrator.calibrate(...)
          6. Assemble one ForecastFrame per horizon in config.forecast_horizons
          7. Publish each ForecastFrame to Kafka (if producer configured)
          8. Return list[ForecastFrame]
        Before buffer fills: return None (warm-up, expected)

        HOPE acceleration: if nowcast_event is HOPE_PRECURSOR, run immediately
        without waiting for full buffer (use whatever frames are available).
        """

    def reset(self) -> None:
        """Clear buffer. Called between replay events."""

    def close(self) -> None: ...
```

**Warm-up note:** The 10-minute buffer means the forecaster produces no output for the
first 600 frames. This is correct and expected. Tests use a 600-frame synthetic window
injected directly. The X6.3 fixture (120 frames) tests the HOPE-accelerated path only.

---

## 7. The Asymmetric Loss — Implementation Detail

**This is the most critical implementation decision.**

```python
def asymmetric_focal_loss(
    logits: torch.Tensor,   # (B, 5) — raw TCN output
    targets: torch.Tensor,  # (B, 5) — soft one-hot labels
    alpha: float = 0.2,
    gamma: float = 2.0,
) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    over_pred  = torch.clamp(probs - targets, min=0.0)
    under_pred = torch.clamp(targets - probs, min=0.0)
    loss = alpha * over_pred**2 + (1.0 - alpha) * under_pred**2
    # Focal weighting — downweight easy negatives
    pt = (probs * targets).sum(dim=1)
    focal_weight = (1.0 - pt) ** gamma
    return (focal_weight.unsqueeze(1) * loss).sum(dim=1).mean()
```

Why `alpha=0.2` (not 0.8)?
- `alpha` weights the **over-prediction** penalty
- Under-prediction penalty = `(1 - 0.2) = 0.8` → 4× heavier
- Missing an M5+ flare (under-prediction) is operationally catastrophic
- False alarm (over-prediction) causes unnecessary satellite safing — expensive but recoverable

---

## 8. Testing Strategy

### 8.1 Unit tests (per module, synthetic tensors)

| Class | Key tests |
|-------|-----------|
| `TestForecastConfig` | Defaults match config.yaml; validation errors on bad values |
| `TestFeatureExtractor` | 600-frame window → correct tensor shapes; NaN→0 substitution; scaler applied |
| `TestDualStreamTCN` | Forward pass produces (1, 5) logits; random weights deterministic in eval(); loader warning on missing file |
| `TestMoiraiForecaster` | Unavailable path returns uniform probs; clip fires on >512 steps; available path (mocked) returns shape (5,) |
| `TestEnsembleWeighter` | Default weights sum to 1.0; clipped context shifts to 0.7/0.3; output always sums to 1.0 |
| `TestMapieCalibrator` | No-calibrator mode returns input unchanged; fitted calibrator produces prediction sets; ECE scalar |

### 8.2 Integration tests (ForecastOrchestrator)

| Test | PRD §8.10 criterion |
|------|---------------------|
| `test_forecast_output_within_10_seconds` | Latency < 10s on 600-frame buffer |
| `test_all_forecasts_include_confidence_interval` | Every ForecastFrame has non-empty prediction_set |
| `test_moirai_context_clip_fallback_bumps_tcn_weight` | >512 context → tcn_weight=0.7 in output |
| `test_no_temporal_leakage` | train_end < val_start - 24h; val_end < test_start - 24h |
| `test_ece_under_0_05_after_mapie_calibration` | ECE < 0.05 (requires fitted calibrator fixture) |
| `test_under_prediction_under_5_pct_for_m5_plus` | Loss gradient favours recall over precision |
| `test_lead_time_mae_under_90s_on_hope_events` | HOPE-accelerated path vs HOPE ground truth lead times |

### 8.3 Data-gate tests (xfail until training data available)

- `test_tss_over_0_65_on_m_x_validation_2024_h2` — needs trained tcn_v1.pt
- `test_roc_auc_over_0_92_on_binary_mx_held_out` — needs trained tcn_v1.pt
- `test_ensemble_beats_best_single_model_by_tss_0_03` — needs trained tcn_v1.pt + Moirai

### 8.4 Synthetic fixtures needed

`conftest.py` additions:
- `synthetic_physics_window(n=600, phase="QUIET")` → list of 600 PhysicsFrames with stable quiet-Sun values
- `synthetic_hope_window(n=120)` → 120 PhysicsFrames with HOPE=True + rising T trajectory
- `fake_nowcast_event(alert_type="HOPE_PRECURSOR")` → NowcastEvent for trigger testing

---

## 9. Build Order (TDD — failing test first)

1. `models.py` — ForecastConfig (write test, implement, pass)
2. `features.py` — FeatureExtractor (write test, implement, pass)
3. `tcn_model.py` — TemporalBlock → DualStreamTCN → loader (write test, implement, pass)
4. `moirai_model.py` — MoiraiForecaster, unavailable path first (write test, implement, pass)
5. `ensemble.py` — EnsembleWeighter (write test, implement, pass)
6. `calibrator.py` — MapieCalibrator (write test, implement, pass)
7. `forecaster.py` — ForecastOrchestrator (write tests, implement all 10 xfails → pass what can be passed; data-gate the model-quality ones)
8. `__init__.py` — public re-exports
9. Update `test_l4_forecasting.py` — replace xfail stubs with real tests

---

## 10. Dependencies

```bash
pip install torch torchvision  # DualStreamTCN — required
pip install mapie               # MAPIE conformal calibration — required
# uni2ts is optional (Moirai) — install if available:
pip install uni2ts              # may require: pip install salesforce-uni2ts
```

No new mandatory dependencies beyond `torch` and `mapie`.
`uni2ts` failure is always caught and handled gracefully.

---

## 11. Global Constraints (inherited from pipeline)

- Python 3.11 — no walrus operator in tests for compat
- All thresholds in `ForecastConfig` — never hardcoded in module files
- `ForecastOrchestrator` not thread-safe — one instance per Kafka partition
- All tests pass offline: no Kafka, no PRADAN, no model weights required
- `ForecastFrame` is already defined in `aruna/models/forecast.py` — import, never redefine
- Git: never commit `models/*.pt` or `models/*.pkl` — they are gitignored

---

## 12. Acceptance Criteria (PRD §8.10)

```
✓ TSS > 0.65 on M+X class events (validation set 2024 H2)             [data-gate: tcn_v1.pt]
✓ ROC-AUC > 0.92 on binary M+/X detection (held-out test set)        [data-gate: tcn_v1.pt]
✓ Expected Calibration Error < 0.05 after MAPIE calibration           [synthetic fixture]
✓ Lead time MAE < 90 seconds on HOPE-triggered events                 [synthetic fixture]
✓ Ensemble beats best single model by TSS ≥ 0.03                      [data-gate: tcn_v1.pt]
✓ No temporal leakage verified                                         [config split test]
✓ Forecast output within 10 seconds of trigger                        [latency test]
✓ All forecasts include calibrated confidence interval                 [structural test]
✓ Under-prediction rate < 5% for M5+ events                           [loss function test]
✓ Moirai context clip → tcn_weight = 0.7                              [unit test]
```
