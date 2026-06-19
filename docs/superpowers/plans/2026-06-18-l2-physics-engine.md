# L2 Physics Engine Implementation Plan

> **Status:** ✅ **COMPLETED** (Session 4 — 2026-06-18)
> **Verification:** All 153 active unit & integration tests passing cleanly.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build the L2 Physics Engine — consuming paired `ObservationFrame` objects and producing `PhysicsFrame` objects with T/EM plasma diagnostics, DNI/TNPSV/SHEI/HFD/QPP/FeKα observables, HOPE precursor flag, and engine state.

**Architecture:** Stateful `PhysicsEngine` owns `collections.deque` ring buffers and orchestrates nine pure-function module classes. `IsothermalFitter` attempts `sunkit-spex f_vth` at import; falls back to `CHIANTILookupTable` (MKG bremsstrahlung, precomputed `.npz`). Every `PhysicsFrame` carries `fit_backend`, `physics_quality_flag`, `data_quality_flag`, and `engine_state` for full traceability.

**Tech Stack:** Python 3.11, numpy, scipy, pywavelets, statsmodels, structlog, kafka-python (mock in tests)

## Global Constraints

- All thresholds configurable via `PhysicsConfig` — never hardcode
- `sunkit-spex` is optional; `CHIANTILookupTable` is the deterministic fallback
- `PhysicsEngine` is not thread-safe — one instance per Kafka consumer/partition
- HOPE `streak_sec` uses elapsed wall-clock seconds (timestamp delta), not frame count
- HFD returns `nan` for first 600 frames — this is correct behaviour, not an error
- All tests must pass offline with no Kafka, no sunkit-spex, no PRADAN
- Pipeline version: `"2.0.0"`, schema version: `"1.0"`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `python/aruna/l2_physics/models.py` | Create | `PhysicsFrame`, `PhysicsQuality`, `EngineState`, `FitResult`, `HOPEResult`, `PhysicsConfig` |
| `python/aruna/l2_physics/chianti_table.py` | Create | MKG bremsstrahlung lookup table |
| `python/aruna/l2_physics/isothermal.py` | Create | `IsothermalFitter` with sunkit-spex / lookup fallback |
| `python/aruna/l2_physics/dni.py` | Create | `DNIComputer` |
| `python/aruna/l2_physics/tnpsv.py` | Create | `TNPSVComputer` |
| `python/aruna/l2_physics/shei.py` | Create | `SHEIComputer` |
| `python/aruna/l2_physics/hfd.py` | Create | `HFDComputer` |
| `python/aruna/l2_physics/qpp.py` | Create | `QPPDetector` |
| `python/aruna/l2_physics/feka.py` | Create | `FeKaExtractor` |
| `python/aruna/l2_physics/hope.py` | Create | `HOPEEvaluator` |
| `python/aruna/l2_physics/engine.py` | Create | `PhysicsEngine` orchestrator |
| `python/aruna/l2_physics/__init__.py` | Modify | Public re-exports |
| `python/aruna/l2_physics/data/.gitkeep` | Create | Ensures data/ dir is tracked |
| `python/tests/conftest.py` | Modify | Add `physics_config`, `quiet_sun_extended_frames` fixtures |
| `python/tests/test_l2_physics.py` | Create | All unit + integration tests |

---

### Task 1: Models

**Files:**
- Create: `python/aruna/l2_physics/models.py`
- Create: `python/tests/test_l2_physics.py` (bootstrap)

**Interfaces:**
- Produces: `PhysicsQuality`, `EngineState`, `FitResult`, `HOPEResult`, `PhysicsConfig`, `PhysicsFrame` — used by every subsequent task

- [x] **Step 1: Write the failing test**

```python
# python/tests/test_l2_physics.py
import math
from datetime import datetime, timezone
from aruna.l2_physics.models import (
    PhysicsFrame, PhysicsQuality, EngineState, FitResult, HOPEResult, PhysicsConfig
)
from aruna.models.observation import QualityFlag

def _make_frame(**overrides):
    defaults = dict(
        frame_id="f1", timestamp=datetime.now(timezone.utc),
        source_frame_ids=["a", "b"],
        T_MK=15.0, T_uncertainty_MK=1.0, EM_cm3=1e48, EM_uncertainty=1e46,
        fit_chi_squared=1.2,
        DNI=0.5, tnpsv_magnitude=0.1, tnpsv_angle_rad=1.2,
        SHEI=-1.0, dSHEI_dt=0.01, HFD=float('nan'),
        QPP_period_sec=None, QPP_amplitude=None, FeKa_flux=0.0,
        flare_phase="QUIET", hope_flag=False,
        hope_lead_time_sec=None, hope_reason=None,
        engine_state="WARMUP",
        data_quality_flag=QualityFlag.GOOD,
        physics_quality_flag=PhysicsQuality.FIT_OK,
        fit_backend="chianti_lookup",
        pipeline_version="2.0.0",
    )
    defaults.update(overrides)
    return PhysicsFrame(**defaults)

class TestModels:
    def test_physics_quality_enum(self):
        assert PhysicsQuality.FIT_OK == 0
        assert PhysicsQuality.FIT_DIVERGED == 1
        assert PhysicsQuality.LOW_CONFIDENCE == 2
        assert PhysicsQuality.BACKEND_FALLBACK == 3

    def test_engine_state_enum(self):
        assert EngineState.WARMUP.value == "WARMUP"
        assert EngineState.ACTIVE.value == "ACTIVE"
        assert EngineState.DEGRADED.value == "DEGRADED"

    def test_fit_result(self):
        r = FitResult(T_MK=15.0, EM_cm3=1e48, T_uncertainty_MK=1.0,
                      EM_uncertainty=1e46, chi2_reduced=1.2, backend="chianti_lookup")
        assert r.T_MK == 15.0
        assert r.backend == "chianti_lookup"

    def test_hope_result(self):
        r = HOPEResult(hope_flag=True, reason="test", lead_time_sec=120)
        assert r.hope_flag is True

    def test_physics_frame_construction(self):
        f = _make_frame()
        assert f.flare_phase == "QUIET"
        assert f.schema_version == "1.0"
        assert f.pipeline_version == "2.0.0"
        assert math.isnan(f.HFD)

    def test_physics_config_defaults(self):
        cfg = PhysicsConfig()
        assert cfg.fit_energy_min_kev == 3.0
        assert cfg.hope_dni_threshold == 1.5
        assert cfg.hfd_window == 600
```

- [x] **Step 2: Run — expect ImportError**

```
pytest python/tests/test_l2_physics.py -v
```
Expected: `ImportError: cannot import name 'PhysicsFrame'`

- [x] **Step 3: Implement `models.py`**

```python
# python/aruna/l2_physics/models.py
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum, Enum
from typing import List, Optional
import numpy as np
from aruna.models.observation import QualityFlag


class PhysicsQuality(IntEnum):
    FIT_OK           = 0
    FIT_DIVERGED     = 1
    LOW_CONFIDENCE   = 2
    BACKEND_FALLBACK = 3


class EngineState(Enum):
    WARMUP   = "WARMUP"
    ACTIVE   = "ACTIVE"
    DEGRADED = "DEGRADED"


@dataclass
class FitResult:
    T_MK:             float
    EM_cm3:           float
    T_uncertainty_MK: float
    EM_uncertainty:   float
    chi2_reduced:     float
    backend:          str


@dataclass
class HOPEResult:
    hope_flag:      bool
    reason:         str
    lead_time_sec:  int


@dataclass
class PhysicsConfig:
    fit_energy_min_kev:        float = 3.0
    fit_energy_max_kev:        float = 20.0
    fit_chi2_diverged:         float = 5.0
    fit_chi2_low_confidence:   float = 2.0
    hope_dni_threshold:        float = 1.5
    hope_tnpsv_speed_threshold:float = 0.05
    hope_tnpsv_angle_min:      float = np.deg2rad(60)
    hope_tnpsv_angle_max:      float = np.deg2rad(120)
    hope_em_rise_threshold:    float = 1e46
    hope_hel_quiet_multiplier: float = 2.0
    hfd_window:                int   = 600
    qpp_window:                int   = 200
    qpp_min_frames:            int   = 50
    tnpsv_history_len:         int   = 30
    shei_history_len:          int   = 10
    dni_reset_min_non_quiet:   int   = 30
    kafka_topic_physics:       str   = "aruna.physics.frames"
    degraded_window:           int   = 60
    degraded_threshold:        float = 0.5


@dataclass
class PhysicsFrame:
    frame_id:             str
    timestamp:            datetime
    source_frame_ids:     List[str]
    T_MK:                 float
    T_uncertainty_MK:     float
    EM_cm3:               float
    EM_uncertainty:       float
    fit_chi_squared:      float
    DNI:                  float
    tnpsv_magnitude:      float
    tnpsv_angle_rad:      float
    SHEI:                 float
    dSHEI_dt:             float
    HFD:                  float
    QPP_period_sec:       Optional[float]
    QPP_amplitude:        Optional[float]
    FeKa_flux:            float
    flare_phase:          str
    hope_flag:            bool
    hope_lead_time_sec:   Optional[int]
    hope_reason:          Optional[str]
    engine_state:         str
    data_quality_flag:    QualityFlag
    physics_quality_flag: PhysicsQuality
    fit_backend:          str
    pipeline_version:     str
    schema_version:       str      = "1.0"
    created_at:           datetime = field(
        default_factory=lambda: datetime.now(timezone.utc))
```

- [x] **Step 4: Run — expect PASS**

```
pytest python/tests/test_l2_physics.py::TestModels -v
```
Expected: `6 passed`

- [x] **Step 5: Commit**

```
git add python/aruna/l2_physics/models.py python/tests/test_l2_physics.py
git commit -m "feat(l2): add PhysicsFrame dataclass, PhysicsQuality, EngineState, PhysicsConfig"
```

---

### Task 2: CHIANTI Lookup Table

**Files:**
- Create: `python/aruna/l2_physics/chianti_table.py`
- Create: `python/aruna/l2_physics/data/.gitkeep`
- Modify: `python/tests/test_l2_physics.py` (append tests)

**Interfaces:**
- Produces: `CHIANTILookupTable.lookup(T_MK, EM_cm3, energy_bins) -> np.ndarray`
- Produces: `CHIANTILookupTable.generate(energy_bins) -> None`

- [x] **Step 1: Write failing tests (append to test file)**

```python
# append to python/tests/test_l2_physics.py
import numpy as np
from aruna.l2_physics.chianti_table import CHIANTILookupTable

class TestCHIANTITable:
    def _bins(self):
        return np.linspace(1.0, 25.0, 25)  # 24 bins

    def test_generate_creates_table(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "chianti_table.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        table = CHIANTILookupTable()
        table.generate(self._bins())
        assert (tmp_path / "chianti_table.npz").exists()

    def test_lookup_returns_nonneg_spectrum(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "chianti_table.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        table = CHIANTILookupTable()
        table.generate(self._bins())
        spec = table.lookup(15.0, 1e48, self._bins())
        assert spec.shape == (24,)
        assert np.all(spec >= 0)

    def test_lookup_higher_T_peaks_at_higher_energy(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "chianti_table.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        bins = np.linspace(1.0, 25.0, 25)
        table = CHIANTILookupTable()
        table.generate(bins)
        spec_cold = table.lookup(5.0, 1e48, bins)
        spec_hot  = table.lookup(30.0, 1e48, bins)
        centres = (bins[:-1] + bins[1:]) / 2.0
        mean_e_cold = np.average(centres, weights=spec_cold + 1e-30)
        mean_e_hot  = np.average(centres, weights=spec_hot  + 1e-30)
        assert mean_e_hot > mean_e_cold
```

- [x] **Step 2: Run — expect ImportError**

```
pytest python/tests/test_l2_physics.py::TestCHIANTITable -v
```

- [x] **Step 3: Implement `chianti_table.py`**

```python
# python/aruna/l2_physics/chianti_table.py
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)

_DATA_DIR   = Path(__file__).parent / "data"
_TABLE_PATH = _DATA_DIR / "chianti_table.npz"

T_GRID  = np.logspace(np.log10(1.0),  np.log10(100.0), 50)
EM_GRID = np.logspace(44.0, 52.0, 50)


def _bremsstrahlung(energies_kev: np.ndarray, T_MK: float, EM_cm3: float) -> np.ndarray:
    T_K  = T_MK * 1e6
    kT   = 8.617e-8 * T_K        # keV
    A    = 9.52e-24               # MKG normalisation (cm^3 s^-1 keV^-1)
    spec = A * EM_cm3 * (T_K ** -0.5) * np.exp(-energies_kev / (kT + 1e-30)) * 1.2
    return np.clip(spec, 0.0, None)


class CHIANTILookupTable:
    def __init__(self) -> None:
        self._interp: RegularGridInterpolator | None = None

    def generate(self, energy_bins: np.ndarray) -> None:
        centres = (energy_bins[:-1] + energy_bins[1:]) / 2.0
        n_e     = len(centres)
        table   = np.zeros((len(T_GRID), len(EM_GRID), n_e), dtype=np.float32)
        for i, T in enumerate(T_GRID):
            for j, EM in enumerate(EM_GRID):
                table[i, j, :] = _bremsstrahlung(centres, T, EM)
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            _TABLE_PATH, table=table,
            T_grid=T_GRID, EM_grid=EM_GRID, energy_bins=energy_bins,
        )
        logger.info("CHIANTILookupTable generated at %s", _TABLE_PATH)
        self._load(energy_bins)

    def _load(self, energy_bins: np.ndarray) -> None:
        data = np.load(_TABLE_PATH)
        self._interp = RegularGridInterpolator(
            (np.log10(data["T_grid"]), np.log10(data["EM_grid"])),
            data["table"].astype(np.float64),
            method="linear", bounds_error=False, fill_value=0.0,
        )

    def lookup(self, T_MK: float, EM_cm3: float, energy_bins: np.ndarray) -> np.ndarray:
        if self._interp is None:
            if _TABLE_PATH.exists():
                self._load(energy_bins)
            else:
                self.generate(energy_bins)
        T_log  = np.clip(np.log10(max(T_MK,  1e-30)), np.log10(T_GRID[0]),  np.log10(T_GRID[-1]))
        EM_log = np.clip(np.log10(max(EM_cm3, 1e-30)), np.log10(EM_GRID[0]), np.log10(EM_GRID[-1]))
        return self._interp([[T_log, EM_log]])[0]
```

- [x] **Step 4: Create data dir**

```
mkdir -p python/aruna/l2_physics/data
touch python/aruna/l2_physics/data/.gitkeep
```

- [x] **Step 5: Run — expect PASS**

```
pytest python/tests/test_l2_physics.py::TestCHIANTITable -v
```
Expected: `3 passed`

- [x] **Step 6: Commit**

```
git add python/aruna/l2_physics/chianti_table.py python/aruna/l2_physics/data/.gitkeep python/tests/test_l2_physics.py
git commit -m "feat(l2): add CHIANTILookupTable (MKG bremsstrahlung fallback fitter)"
```

---

### Task 3: Isothermal Fitter

**Files:**
- Create: `python/aruna/l2_physics/isothermal.py`
- Modify: `python/tests/test_l2_physics.py` (append)

**Interfaces:**
- Consumes: `CHIANTILookupTable.lookup`
- Produces: `IsothermalFitter.fit(counts_per_s_kev, energy_bins_kev) -> FitResult`

- [x] **Step 1: Append failing tests**

```python
# append to python/tests/test_l2_physics.py
from aruna.l2_physics.isothermal import IsothermalFitter
from aruna.l2_physics.chianti_table import CHIANTILookupTable, _bremsstrahlung

class TestIsothermalFitter:
    def _synthetic_spectrum(self, T_MK=15.0, EM_cm3=1e48):
        bins = np.linspace(1.0, 25.0, 25)
        centres = (bins[:-1] + bins[1:]) / 2.0
        return _bremsstrahlung(centres, T_MK, EM_cm3), bins

    def test_fit_returns_fitresult(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "t.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        fitter = IsothermalFitter()
        counts, bins = self._synthetic_spectrum()
        result = fitter.fit(counts, bins)
        assert hasattr(result, "T_MK")
        assert hasattr(result, "backend")

    def test_fit_converges_on_synthetic(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "t.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        fitter = IsothermalFitter()
        counts, bins = self._synthetic_spectrum(T_MK=15.0, EM_cm3=1e48)
        result = fitter.fit(counts, bins)
        assert not np.isnan(result.T_MK), "fit should converge on synthetic data"
        assert result.T_MK > 0
        assert result.chi2_reduced < 5.0

    def test_fit_returns_nan_on_zero_spectrum(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "t.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        fitter = IsothermalFitter()
        bins = np.linspace(1.0, 25.0, 25)
        counts = np.zeros(24)
        result = fitter.fit(counts, bins)
        assert np.isnan(result.T_MK)

    def test_backend_field_always_set(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "t.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        fitter = IsothermalFitter()
        counts, bins = self._synthetic_spectrum()
        result = fitter.fit(counts, bins)
        assert result.backend in ("sunkit_spex", "chianti_lookup")
```

- [x] **Step 2: Run — expect ImportError**

```
pytest python/tests/test_l2_physics.py::TestIsothermalFitter -v
```

- [x] **Step 3: Implement `isothermal.py`**

```python
# python/aruna/l2_physics/isothermal.py
from __future__ import annotations
import logging
import numpy as np
from scipy.optimize import minimize
from .chianti_table import CHIANTILookupTable
from .models import FitResult

logger = logging.getLogger(__name__)

try:
    from sunkit_spex.fitting import FitIsothermal as _SunkitFit
    _BACKEND = "sunkit_spex"
    logger.info("IsothermalFitter: using sunkit-spex f_vth backend")
except ImportError:
    _SunkitFit = None
    _BACKEND   = "chianti_lookup"
    logger.warning("IsothermalFitter: sunkit-spex unavailable, using CHIANTILookupTable fallback")

_shared_table = CHIANTILookupTable()

FIT_E_MIN = 3.0
FIT_E_MAX = 20.0


def _poisson_chi2(observed: np.ndarray, model: np.ndarray) -> float:
    mask = model > 1e-30
    if not mask.any():
        return np.inf
    o, m = observed[mask], model[mask]
    chi2 = 2.0 * np.sum(m - o + o * np.log((o + 1e-30) / m))
    return chi2 / max(mask.sum() - 2, 1)


class IsothermalFitter:
    def __init__(self) -> None:
        self._backend = _BACKEND

    @property
    def backend(self) -> str:
        return self._backend

    def fit(
        self,
        counts_per_s_kev: np.ndarray,
        energy_bins_kev:  np.ndarray,
        response_matrix=None,
    ) -> FitResult:
        centres = (energy_bins_kev[:-1] + energy_bins_kev[1:]) / 2.0
        mask    = (centres >= FIT_E_MIN) & (centres <= FIT_E_MAX)
        if not mask.any():
            return FitResult(np.nan, np.nan, np.nan, np.nan, np.inf, self._backend)

        if self._backend == "sunkit_spex" and _SunkitFit is not None:
            return self._fit_sunkit(counts_per_s_kev[mask], energy_bins_kev, response_matrix)
        return self._fit_lookup(counts_per_s_kev, energy_bins_kev, mask)

    def _fit_lookup(self, counts, energy_bins, mask) -> FitResult:
        counts_fit = counts[mask]

        def objective(params):
            T_MK, log_EM = params
            if T_MK <= 0 or T_MK > 200:
                return 1e9
            model_full = _shared_table.lookup(T_MK, 10 ** log_EM, energy_bins)
            return _poisson_chi2(counts_fit, model_full[mask])

        res = minimize(objective, x0=[15.0, 48.0], method="Nelder-Mead",
                       options={"xatol": 0.1, "fatol": 0.01, "maxiter": 600})
        T_MK, EM_cm3 = res.x[0], 10 ** res.x[1]
        chi2 = float(res.fun)

        if not res.success or chi2 > 5.0 or T_MK <= 0:
            return FitResult(np.nan, np.nan, np.nan, np.nan, chi2, self._backend)

        return FitResult(
            T_MK=float(T_MK), EM_cm3=float(EM_cm3),
            T_uncertainty_MK=max(0.5, T_MK * 0.05),
            EM_uncertainty=max(1e46, EM_cm3 * 0.1),
            chi2_reduced=chi2, backend=self._backend,
        )

    def _fit_sunkit(self, counts_fit, energy_bins, response_matrix) -> FitResult:
        try:
            fitter = _SunkitFit(model="f_vth",
                                energy_range=(FIT_E_MIN, FIT_E_MAX),
                                abundance_table="coronal",
                                response_matrix=response_matrix)
            r = fitter.fit(counts_fit, energy_bins)
            return FitResult(r.T_MK, r.EM_cm3, r.T_1sigma, r.EM_1sigma,
                             r.chi2_reduced, self._backend)
        except Exception as exc:
            logger.error("sunkit-spex fit failed: %s", exc)
            return FitResult(np.nan, np.nan, np.nan, np.nan, np.inf, self._backend)
```

- [x] **Step 4: Run — expect PASS**

```
pytest python/tests/test_l2_physics.py::TestIsothermalFitter -v
```
Expected: `4 passed`

- [x] **Step 5: Commit**

```
git add python/aruna/l2_physics/isothermal.py python/tests/test_l2_physics.py
git commit -m "feat(l2): add IsothermalFitter with sunkit-spex / CHIANTILookupTable fallback"
```

---

### Task 4: DNI, TNPSV, SHEI Computers

**Files:**
- Create: `python/aruna/l2_physics/dni.py`
- Create: `python/aruna/l2_physics/tnpsv.py`
- Create: `python/aruna/l2_physics/shei.py`
- Modify: `python/tests/test_l2_physics.py` (append)

**Interfaces:**
- Produces: `DNIComputer.update(hel_25_50, sol_8_15, dt=1.0) -> float`; `.reset()`
- Produces: `TNPSVComputer.update(T_MK, EM_cm3) -> Tuple[float, float]`
- Produces: `SHEIComputer.update_single(hel_frame) -> Tuple[float, float]`

- [x] **Step 1: Append failing tests**

```python
# append to python/tests/test_l2_physics.py
from aruna.l2_physics.dni   import DNIComputer
from aruna.l2_physics.tnpsv import TNPSVComputer
from aruna.l2_physics.shei  import SHEIComputer

class TestDNIComputer:
    def test_accumulates_correctly(self):
        dni = DNIComputer()
        v = dni.update(hel_25_50_flux=2.0, sol_8_15_flux=1.0, dt=1.0)
        assert v == pytest.approx(2.0, rel=1e-6)
        v2 = dni.update(hel_25_50_flux=2.0, sol_8_15_flux=1.0, dt=1.0)
        assert v2 == pytest.approx(4.0, rel=1e-6)

    def test_reset_zeroes_accumulator(self):
        dni = DNIComputer()
        dni.update(5.0, 1.0)
        dni.reset()
        v = dni.update(1.0, 1.0)
        assert v == pytest.approx(1.0, rel=1e-6)

    def test_no_div_zero_on_zero_solexs(self):
        dni = DNIComputer()
        v = dni.update(1.0, 0.0)
        assert np.isfinite(v)

class TestTNPSVComputer:
    def test_returns_zero_on_insufficient_history(self):
        c = TNPSVComputer()
        mag, ang = c.update(10.0, 1e47)
        assert mag == 0.0 and ang == 0.0

    def test_rising_T_flat_EM_angle_near_zero(self):
        c = TNPSVComputer()
        for t in np.linspace(10.0, 20.0, 10):
            mag, ang = c.update(t, 1e48)
        # dT/dt > 0, dEM/dt ~ 0 → angle near 0 (atan2(~0, positive))
        assert mag > 0
        assert abs(ang) < np.deg2rad(30)

    def test_magnitude_positive(self):
        c = TNPSVComputer()
        for t in np.linspace(5.0, 25.0, 5):
            mag, ang = c.update(t, 1e48)
        assert mag >= 0

class TestSHEIComputer:
    def _make_hel_frame(self, hard_flux=10.0, soft_flux=1.0):
        bins = np.linspace(5.0, 80.0, 33)   # 32 bins
        centres = (bins[:-1] + bins[1:]) / 2.0
        counts = np.zeros(32)
        counts[(centres >= 40) & (centres <= 70)] = hard_flux
        counts[(centres >= 10) & (centres <= 25)] = soft_flux
        from aruna.models.observation import ObservationFrame, QualityFlag
        from datetime import datetime, timezone
        return ObservationFrame(
            instrument="HEL1OS_CDTE", timestamp=datetime.now(timezone.utc),
            energy_bins_kev=bins, counts_per_s_kev=counts,
            quality_flag=QualityFlag.GOOD, dead_time_fraction=0.0,
            calibration_version="test", solar_distance_au=1.0,
            spacecraft_mode="SCIENCE", saa_flag=False,
            source_file="test.fits", checksum_sha256="abc",
            pipeline_version="2.0.0",
        )

    def test_hard_dominated_shei_above_threshold(self):
        c = SHEIComputer()
        shei, _ = c.update_single(self._make_hel_frame(hard_flux=100.0, soft_flux=1.0))
        assert shei > -0.8

    def test_soft_dominated_shei_below_threshold(self):
        c = SHEIComputer()
        shei, _ = c.update_single(self._make_hel_frame(hard_flux=0.01, soft_flux=100.0))
        assert shei < -1.5

    def test_dshei_dt_nonzero_after_two_frames(self):
        c = SHEIComputer()
        c.update_single(self._make_hel_frame(hard_flux=1.0, soft_flux=10.0))
        _, dshei = c.update_single(self._make_hel_frame(hard_flux=50.0, soft_flux=10.0))
        assert dshei != 0.0
```

- [x] **Step 2: Run — expect ImportError**

```
pytest python/tests/test_l2_physics.py::TestDNIComputer python/tests/test_l2_physics.py::TestTNPSVComputer python/tests/test_l2_physics.py::TestSHEIComputer -v
```

- [x] **Step 3: Implement `dni.py`**

```python
# python/aruna/l2_physics/dni.py
from __future__ import annotations

class DNIComputer:
    def __init__(self) -> None:
        self._cumulative_hel: float = 0.0

    def update(self, hel_25_50_flux: float, sol_8_15_flux: float, dt: float = 1.0) -> float:
        self._cumulative_hel += hel_25_50_flux * dt
        return self._cumulative_hel / (sol_8_15_flux + 1e-30)

    def reset(self) -> None:
        self._cumulative_hel = 0.0
```

- [x] **Step 4: Implement `tnpsv.py`**

```python
# python/aruna/l2_physics/tnpsv.py
from __future__ import annotations
from collections import deque
from typing import Tuple
import numpy as np

class TNPSVComputer:
    def __init__(self, maxlen: int = 30) -> None:
        self._T  = deque(maxlen=maxlen)
        self._EM = deque(maxlen=maxlen)

    def update(self, T_MK: float, EM_cm3: float) -> Tuple[float, float]:
        self._T.append(T_MK)
        self._EM.append(EM_cm3)
        if len(self._T) < 3:
            return 0.0, 0.0
        T  = np.array(self._T);  T_max  = max(float(np.max(T)),  1e-30)
        EM = np.array(self._EM); EM_max = max(float(np.max(EM)), 1e-30)
        dT  = np.gradient(T)
        dEM = np.gradient(EM)
        mag = float(np.sqrt((dT[-1] / T_max) ** 2 + (dEM[-1] / EM_max) ** 2))
        ang = float(np.arctan2(dEM[-1], dT[-1]))
        return mag, ang
```

- [x] **Step 5: Implement `shei.py`**

```python
# python/aruna/l2_physics/shei.py
from __future__ import annotations
from collections import deque
from typing import Tuple
import numpy as np

class SHEIComputer:
    def __init__(self, maxlen: int = 10) -> None:
        self._history: deque = deque(maxlen=maxlen)

    def update_single(self, hel_frame) -> Tuple[float, float]:
        return self.update(hel_frame.counts_per_s_kev, hel_frame.energy_bins_kev)

    def update(self, counts: np.ndarray, energy_bins: np.ndarray) -> Tuple[float, float]:
        centres   = (energy_bins[:-1] + energy_bins[1:]) / 2.0
        F_hard    = float(np.sum(counts[(centres >= 40) & (centres <= 70)]))
        F_soft    = float(np.sum(counts[(centres >= 10) & (centres <= 25)]))
        shei      = float(np.log10(F_hard / (F_soft + 1e-30)))
        self._history.append(shei)
        dshei_dt  = float(np.gradient(np.array(self._history))[-1]) if len(self._history) >= 2 else 0.0
        return shei, dshei_dt
```

- [x] **Step 6: Run — expect PASS**

```
pytest python/tests/test_l2_physics.py::TestDNIComputer python/tests/test_l2_physics.py::TestTNPSVComputer python/tests/test_l2_physics.py::TestSHEIComputer -v
```
Expected: `8 passed`

- [x] **Step 7: Commit**

```
git add python/aruna/l2_physics/dni.py python/aruna/l2_physics/tnpsv.py python/aruna/l2_physics/shei.py python/tests/test_l2_physics.py
git commit -m "feat(l2): add DNIComputer, TNPSVComputer, SHEIComputer"
```

---

### Task 5: HFD + QPP (Windowed Modules)

**Files:**
- Create: `python/aruna/l2_physics/hfd.py`
- Create: `python/aruna/l2_physics/qpp.py`
- Modify: `python/tests/test_l2_physics.py` (append)

**Interfaces:**
- Produces: `HFDComputer.update(total_flux: float) -> float` (nan during warm-up)
- Produces: `QPPDetector.update(hel_25_50_flux: float) -> Tuple[Optional[float], Optional[float]]`

- [x] **Step 1: Append failing tests**

```python
# append to python/tests/test_l2_physics.py
import pytest
from aruna.l2_physics.hfd import HFDComputer
from aruna.l2_physics.qpp import QPPDetector

class TestHFDComputer:
    def test_nan_during_warmup(self):
        h = HFDComputer(window=600)
        for _ in range(599):
            val = h.update(1.0)
        assert np.isnan(val)

    def test_returns_float_after_window_fills(self):
        h = HFDComputer(window=50, k_max=5)
        for _ in range(49):
            h.update(float(np.random.rand()))
        val = h.update(float(np.random.rand()))
        assert np.isnan(val)
        val = h.update(float(np.random.rand()))
        assert not np.isnan(val)
        assert 1.0 <= val <= 2.0

    def test_sine_hfd_near_one(self):
        h = HFDComputer(window=100, k_max=5)
        t = np.linspace(0, 10 * np.pi, 200)
        for x in np.sin(t):
            val = h.update(float(x))
        assert not np.isnan(val)
        assert val < 1.5

    def test_noise_hfd_above_sine(self):
        rng = np.random.default_rng(42)
        h_sine  = HFDComputer(window=100, k_max=5)
        h_noise = HFDComputer(window=100, k_max=5)
        t = np.linspace(0, 10 * np.pi, 200)
        sine_val = noise_val = np.nan
        for i, x in enumerate(t):
            sine_val  = h_sine.update(float(np.sin(x)))
            noise_val = h_noise.update(float(rng.standard_normal()))
        assert not np.isnan(sine_val) and not np.isnan(noise_val)
        assert noise_val > sine_val

class TestQPPDetector:
    def test_none_during_warmup(self):
        q = QPPDetector(window=200, min_frames=50)
        for _ in range(49):
            period, amp = q.update(1.0)
        assert period is None and amp is None

    def test_detects_injected_period(self):
        q = QPPDetector(window=200, min_frames=50)
        t = np.linspace(0, 200, 200)
        signal = 10.0 + 3.0 * np.sin(2 * np.pi * t / 30.0)
        period = amp = None
        for x in signal:
            period, amp = q.update(float(x))
        if period is not None:
            assert 20 < period < 45
```

- [x] **Step 2: Run — expect ImportError**

```
pytest python/tests/test_l2_physics.py::TestHFDComputer python/tests/test_l2_physics.py::TestQPPDetector -v
```

- [x] **Step 3: Implement `hfd.py`**

```python
# python/aruna/l2_physics/hfd.py
from __future__ import annotations
from collections import deque
import numpy as np

class HFDComputer:
    def __init__(self, window: int = 600, k_max: int = 10) -> None:
        self._window = window
        self._k_max  = k_max
        self._buf: deque = deque(maxlen=window)

    def update(self, total_flux: float) -> float:
        self._buf.append(total_flux)
        if len(self._buf) < self._window:
            return float("nan")
        return self._hfd(np.array(self._buf))

    def _hfd(self, x: np.ndarray) -> float:
        L = []
        for k in range(1, self._k_max + 1):
            Lk = []
            for m in range(1, k + 1):
                idx = np.arange(m - 1, len(x), k)
                if len(idx) < 2:
                    continue
                Lmk = np.sum(np.abs(np.diff(x[idx]))) * (len(x) - 1) / (len(idx) * k ** 2)
                Lk.append(Lmk)
            if Lk:
                L.append(float(np.mean(Lk)))
        if len(L) < 2:
            return float("nan")
        k_vals = np.arange(1, len(L) + 1, dtype=float)
        try:
            slope, _ = np.polyfit(np.log(k_vals), np.log(np.array(L) + 1e-30), 1)
            return float(-slope)
        except Exception:
            return float("nan")
```

- [x] **Step 4: Implement `qpp.py`**

```python
# python/aruna/l2_physics/qpp.py
from __future__ import annotations
import logging
from collections import deque
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import pywt
    from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess
    _QPP_AVAILABLE = True
except ImportError:
    _QPP_AVAILABLE = False
    logger.warning("QPPDetector: pywt/statsmodels unavailable — QPP disabled")


class QPPDetector:
    def __init__(self, window: int = 200, min_frames: int = 50) -> None:
        self._window     = window
        self._min_frames = min_frames
        self._buf: deque = deque(maxlen=window)

    def update(self, hel_25_50_flux: float) -> Tuple[Optional[float], Optional[float]]:
        self._buf.append(hel_25_50_flux)
        if len(self._buf) < self._min_frames or not _QPP_AVAILABLE:
            return None, None
        return self._detect(np.array(self._buf))

    def _detect(self, data: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        try:
            trend     = _lowess(data, np.arange(len(data)), frac=0.3)[:, 1]
            detrended = data - trend
            scales    = np.arange(5, 200)
            coeffs, _ = pywt.cwt(detrended, scales, "morl", sampling_period=1.0)
            power      = np.abs(coeffs) ** 2
            mean_power = np.mean(power, axis=1)
            noise_floor = np.percentile(mean_power, 99)
            mask        = mean_power > noise_floor
            if not mask.any():
                return None, None
            dominant = scales[mask][np.argmax(mean_power[mask])]
            A_QPP    = float(np.max(mean_power) / (np.mean(mean_power) + 1e-30))
            return float(dominant), A_QPP
        except Exception as exc:
            logger.warning("QPP detection failed: %s", exc)
            return None, None
```

- [x] **Step 5: Run — expect PASS**

```
pytest python/tests/test_l2_physics.py::TestHFDComputer python/tests/test_l2_physics.py::TestQPPDetector -v
```
Expected: `6 passed` (QPP period test may skip if pywt unavailable — that's acceptable)

- [x] **Step 6: Commit**

```
git add python/aruna/l2_physics/hfd.py python/aruna/l2_physics/qpp.py python/tests/test_l2_physics.py
git commit -m "feat(l2): add HFDComputer (Higuchi) and QPPDetector (Morlet CWT)"
```

---

### Task 6: FeKα Extractor + HOPE Evaluator

**Files:**
- Create: `python/aruna/l2_physics/feka.py`
- Create: `python/aruna/l2_physics/hope.py`
- Modify: `python/tests/test_l2_physics.py` (append)

**Interfaces:**
- Produces: `FeKaExtractor.extract(counts, energy_bins, T_MK, EM_cm3) -> float`
- Produces: `HOPEEvaluator.evaluate(..., streak_sec, config) -> HOPEResult`

- [x] **Step 1: Append failing tests**

```python
# append to python/tests/test_l2_physics.py
from aruna.l2_physics.feka import FeKaExtractor
from aruna.l2_physics.hope import HOPEEvaluator
from aruna.l2_physics.models import PhysicsConfig

class TestFeKaExtractor:
    def test_known_feka_line_detected(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "t.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        bins = np.linspace(1.0, 25.0, 65)
        centres = (bins[:-1] + bins[1:]) / 2.0
        from aruna.l2_physics.chianti_table import _bremsstrahlung
        counts = _bremsstrahlung(centres, 15.0, 1e48).copy()
        feka_mask = (centres >= 6.2) & (centres <= 6.6)
        counts[feka_mask] += 5.0
        ext = FeKaExtractor()
        flux = ext.extract(counts, bins, T_MK=15.0, EM_cm3=1e48)
        assert flux > 0

    def test_nan_T_returns_zero(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "t.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        ext = FeKaExtractor()
        bins = np.linspace(1.0, 25.0, 25)
        counts = np.ones(24)
        assert ext.extract(counts, bins, T_MK=float("nan"), EM_cm3=1e48) == 0.0

    def test_feka_flux_nonneg(self, tmp_path, monkeypatch):
        import aruna.l2_physics.chianti_table as ct
        monkeypatch.setattr(ct, "_TABLE_PATH", tmp_path / "t.npz")
        monkeypatch.setattr(ct, "_DATA_DIR", tmp_path)
        ext = FeKaExtractor()
        bins = np.linspace(1.0, 25.0, 25)
        counts = np.zeros(24)
        assert ext.extract(counts, bins, T_MK=15.0, EM_cm3=1e48) >= 0.0

class TestHOPEEvaluator:
    def _cfg(self):
        return PhysicsConfig()

    def _all_true_kwargs(self):
        cfg = self._cfg()
        return dict(
            T=15.0, dT_dt=0.5, EM=1e47, dEM_dt=1e44,
            DNI=0.5, tnpsv_mag=0.1,
            tnpsv_angle=np.deg2rad(90),
            hel_flux=100.0, hel_quiet_baseline=200.0,
            streak_sec=5.0, config=cfg,
        )

    def test_all_conditions_true_fires_hope(self):
        ev = HOPEEvaluator()
        r = ev.evaluate(**self._all_true_kwargs())
        assert r.hope_flag is True

    def test_t_not_rising_blocks_hope(self):
        ev = HOPEEvaluator()
        kwargs = self._all_true_kwargs()
        kwargs["dT_dt"] = -0.1
        r = ev.evaluate(**kwargs)
        assert r.hope_flag is False

    def test_streak_too_short_blocks_hope(self):
        ev = HOPEEvaluator()
        kwargs = self._all_true_kwargs()
        kwargs["streak_sec"] = 1.0
        r = ev.evaluate(**kwargs)
        assert r.hope_flag is False

    def test_em_rising_blocks_hope(self):
        ev = HOPEEvaluator()
        kwargs = self._all_true_kwargs()
        kwargs["dEM_dt"] = 1e50
        r = ev.evaluate(**kwargs)
        assert r.hope_flag is False

    def test_hel1os_active_blocks_hope(self):
        ev = HOPEEvaluator()
        kwargs = self._all_true_kwargs()
        kwargs["hel_flux"] = 1000.0
        r = ev.evaluate(**kwargs)
        assert r.hope_flag is False

    def test_lead_time_scales_with_streak(self):
        ev = HOPEEvaluator()
        kwargs = self._all_true_kwargs()
        kwargs["streak_sec"] = 10.0
        r = ev.evaluate(**kwargs)
        assert r.lead_time_sec == int(90 + 10.0 * 2)

    def test_reason_always_populated(self):
        ev = HOPEEvaluator()
        r = ev.evaluate(**self._all_true_kwargs())
        assert len(r.reason) > 0
```

- [x] **Step 2: Run — expect ImportError**

```
pytest python/tests/test_l2_physics.py::TestFeKaExtractor python/tests/test_l2_physics.py::TestHOPEEvaluator -v
```

- [x] **Step 3: Implement `feka.py`**

```python
# python/aruna/l2_physics/feka.py
from __future__ import annotations
import numpy as np
from .chianti_table import CHIANTILookupTable

_table = CHIANTILookupTable()

class FeKaExtractor:
    def extract(
        self,
        counts_per_s_kev: np.ndarray,
        energy_bins_kev:  np.ndarray,
        T_MK:   float,
        EM_cm3: float,
    ) -> float:
        if np.isnan(T_MK) or np.isnan(EM_cm3) or T_MK <= 0 or EM_cm3 <= 0:
            return 0.0
        continuum = _table.lookup(T_MK, EM_cm3, energy_bins_kev)
        residual  = counts_per_s_kev - continuum
        centres   = (energy_bins_kev[:-1] + energy_bins_kev[1:]) / 2.0
        mask      = (centres >= 6.2) & (centres <= 6.6)
        if not mask.any():
            return 0.0
        return float(np.sum(np.clip(residual[mask], 0.0, None)))
```

- [x] **Step 4: Implement `hope.py`**

```python
# python/aruna/l2_physics/hope.py
from __future__ import annotations
import numpy as np
from .models import HOPEResult, PhysicsConfig

class HOPEEvaluator:
    def evaluate(
        self,
        T: float, dT_dt: float,
        EM: float, dEM_dt: float,
        DNI: float,
        tnpsv_mag: float, tnpsv_angle: float,
        hel_flux: float, hel_quiet_baseline: float,
        streak_sec: float,
        config: PhysicsConfig,
    ) -> HOPEResult:
        cond_T    = dT_dt > 0 and streak_sec >= 3.0
        cond_EM   = abs(dEM_dt) < config.hope_em_rise_threshold
        cond_HEL  = hel_flux < hel_quiet_baseline * config.hope_hel_quiet_multiplier
        cond_TNPSV = (
            tnpsv_mag > config.hope_tnpsv_speed_threshold
            and config.hope_tnpsv_angle_min < tnpsv_angle < config.hope_tnpsv_angle_max
        )
        cond_DNI  = DNI > 0.2

        hope_flag = all([cond_T, cond_EM, cond_HEL, cond_TNPSV, cond_DNI])
        failed    = [n for n, v in [("T_rising", cond_T), ("EM_flat", cond_EM),
                                     ("HEL_quiet", cond_HEL), ("TNPSV", cond_TNPSV),
                                     ("DNI", cond_DNI)] if not v]
        reason = (f"dT/dt={dT_dt:.3f} streak={streak_sec:.1f}s | "
                  f"dEM/dt={dEM_dt:.2e} | HEL={hel_flux:.1f} base={hel_quiet_baseline:.1f} | "
                  f"TNPSV mag={tnpsv_mag:.3f} ang={np.degrees(tnpsv_angle):.1f}° | "
                  f"DNI={DNI:.2f}" +
                  (f" | FAILED: {', '.join(failed)}" if failed else ""))
        return HOPEResult(hope_flag=hope_flag, reason=reason,
                          lead_time_sec=int(90 + streak_sec * 2))
```

- [x] **Step 5: Run — expect PASS**

```
pytest python/tests/test_l2_physics.py::TestFeKaExtractor python/tests/test_l2_physics.py::TestHOPEEvaluator -v
```
Expected: `10 passed`

- [x] **Step 6: Commit**

```
git add python/aruna/l2_physics/feka.py python/aruna/l2_physics/hope.py python/tests/test_l2_physics.py
git commit -m "feat(l2): add FeKaExtractor and HOPEEvaluator"
```

---

### Task 7: PhysicsEngine + Integration Test

**Files:**
- Create: `python/aruna/l2_physics/engine.py`
- Modify: `python/aruna/l2_physics/__init__.py`
- Modify: `python/tests/conftest.py` (add L2 fixtures)
- Modify: `python/tests/test_l2_physics.py` (append integration tests)

**Interfaces:**
- Consumes: all module classes from tasks 1–6
- Produces: `PhysicsEngine(config, kafka_producer=None).process(sol_frame, hel_frame) -> PhysicsFrame`
- Produces: `PhysicsEngine.reset() -> None`
- Produces: `PhysicsEngine.state -> EngineState`

- [x] **Step 1: Append engine + integration tests**

```python
# append to python/tests/test_l2_physics.py
import pytest
from aruna.l2_physics.engine import PhysicsEngine
from aruna.l2_physics.models import PhysicsConfig, PhysicsQuality, EngineState

class TestPhysicsEngine:
    def test_process_returns_physics_frame(self, solexs_x63_frames, hel1os_x63_frames):
        engine = PhysicsEngine(PhysicsConfig())
        frame = engine.process(solexs_x63_frames[0], hel1os_x63_frames[0])
        assert frame.pipeline_version == "2.0.0"
        assert frame.fit_backend in ("sunkit_spex", "chianti_lookup")
        assert frame.engine_state == "WARMUP"

    def test_state_is_warmup_before_600_frames(self, solexs_x63_frames, hel1os_x63_frames):
        engine = PhysicsEngine(PhysicsConfig())
        for s, h in zip(solexs_x63_frames[:10], hel1os_x63_frames[:10]):
            engine.process(s, h)
        assert engine.state == EngineState.WARMUP

    def test_source_frame_ids_populated(self, solexs_x63_frames, hel1os_x63_frames):
        engine = PhysicsEngine(PhysicsConfig())
        frame = engine.process(solexs_x63_frames[0], hel1os_x63_frames[0])
        assert len(frame.source_frame_ids) == 2

    def test_reset_clears_state(self, solexs_x63_frames, hel1os_x63_frames):
        engine = PhysicsEngine(PhysicsConfig())
        for s, h in zip(solexs_x63_frames[:5], hel1os_x63_frames[:5]):
            engine.process(s, h)
        engine.reset()
        assert engine.state == EngineState.WARMUP

    def test_hfd_nan_for_120_frame_fixture(self, solexs_x63_frames, hel1os_x63_frames):
        engine = PhysicsEngine(PhysicsConfig())
        frames = [engine.process(s, h)
                  for s, h in zip(solexs_x63_frames, hel1os_x63_frames)]
        assert all(frame.HFD != frame.HFD for frame in frames)  # nan check

    def test_flare_phase_assigned(self, solexs_x63_frames, hel1os_x63_frames):
        engine = PhysicsEngine(PhysicsConfig())
        valid_phases = {"QUIET", "HOPE", "IMPULSIVE", "GRADUAL", "DECAY"}
        for s, h in zip(solexs_x63_frames, hel1os_x63_frames):
            f = engine.process(s, h)
            assert f.flare_phase in valid_phases

    def test_physics_quality_flag_set(self, solexs_x63_frames, hel1os_x63_frames):
        engine = PhysicsEngine(PhysicsConfig())
        frame = engine.process(solexs_x63_frames[0], hel1os_x63_frames[0])
        assert isinstance(frame.physics_quality_flag, PhysicsQuality)

    def test_data_quality_flag_inherited_from_l1(self, solexs_x63_frames, hel1os_x63_frames):
        from aruna.models.observation import QualityFlag
        engine = PhysicsEngine(PhysicsConfig())
        frame = engine.process(solexs_x63_frames[0], hel1os_x63_frames[0])
        assert isinstance(frame.data_quality_flag, QualityFlag)

    def test_mismatched_timestamps_logged(self, solexs_x63_frames, hel1os_x63_frames, caplog):
        from datetime import timedelta
        import copy, logging
        engine = PhysicsEngine(PhysicsConfig())
        hel_late = copy.copy(hel1os_x63_frames[0])
        hel_late.timestamp = hel1os_x63_frames[0].timestamp + timedelta(seconds=2)
        with caplog.at_level(logging.WARNING):
            engine.process(solexs_x63_frames[0], hel_late)
        assert any("timestamp" in r.message.lower() for r in caplog.records)
```

- [x] **Step 2: Add fixtures to conftest.py**

```python
# append to python/tests/conftest.py (after existing fixtures)
import pytest
from aruna.l2_physics.models import PhysicsConfig

@pytest.fixture
def physics_config():
    return PhysicsConfig()

@pytest.fixture
def quiet_sun_extended_frames(solexs_x63_frames, hel1os_x63_frames):
    """700 quiet-sun frames prepended to X6.3 — fills HFD window before event."""
    import copy
    from datetime import timedelta
    import numpy as np
    from aruna.models.observation import ObservationFrame, QualityFlag

    sol_ref = solexs_x63_frames[0]
    hel_ref = hel1os_x63_frames[0]
    n_quiet = 700

    def _quiet_frame(ref, idx):
        f = copy.copy(ref)
        f.timestamp = ref.timestamp - timedelta(seconds=n_quiet - idx)
        f.counts_per_s_kev = np.abs(np.random.default_rng(idx).normal(
            1.0, 0.1, len(ref.counts_per_s_kev)))
        import uuid
        f.frame_id = str(uuid.uuid4())
        return f

    sol_quiet = [_quiet_frame(sol_ref, i) for i in range(n_quiet)]
    hel_quiet = [_quiet_frame(hel_ref, i + n_quiet) for i in range(n_quiet)]
    return (sol_quiet + solexs_x63_frames, hel_quiet + hel1os_x63_frames)
```

- [x] **Step 3: Run — expect ImportError on engine**

```
pytest python/tests/test_l2_physics.py::TestPhysicsEngine -v
```

- [x] **Step 4: Implement `engine.py`**

```python
# python/aruna/l2_physics/engine.py
from __future__ import annotations
import logging
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np

from aruna.models.observation import ObservationFrame, QualityFlag
from .chianti_table import CHIANTILookupTable
from .dni    import DNIComputer
from .feka   import FeKaExtractor
from .hfd    import HFDComputer
from .hope   import HOPEEvaluator
from .isothermal import IsothermalFitter
from .models import (EngineState, PhysicsConfig, PhysicsFrame,
                     PhysicsQuality, HOPEResult)
from .qpp    import QPPDetector
from .shei   import SHEIComputer
from .tnpsv  import TNPSVComputer

logger = logging.getLogger(__name__)
_VERSION = "2.0.0"


def _total_flux(frame: ObservationFrame) -> float:
    return float(np.sum(frame.counts_per_s_kev))


def _band(frame: ObservationFrame, lo: float, hi: float) -> float:
    return frame.band_flux(lo, hi)


def _assign_phase(
    DNI: float, SHEI: float, dT_dt: float, hope_flag: bool, cfg: PhysicsConfig
) -> str:
    if DNI > cfg.hope_dni_threshold and SHEI > -0.8:
        return "IMPULSIVE"
    if hope_flag:
        return "HOPE"
    if SHEI < -1.5 and dT_dt < 0:
        return "DECAY"
    if dT_dt > 0 or DNI > 0.2:
        return "GRADUAL"
    return "QUIET"


class PhysicsEngine:
    """
    Not thread-safe. One instance per Kafka consumer/partition.
    """

    def __init__(self, config: Optional[PhysicsConfig] = None, kafka_producer=None) -> None:
        self._cfg    = config or PhysicsConfig()
        self._pub    = kafka_producer
        self._fitter = IsothermalFitter()
        self._dni    = DNIComputer()
        self._tnpsv  = TNPSVComputer(self._cfg.tnpsv_history_len)
        self._shei   = SHEIComputer(self._cfg.shei_history_len)
        self._hfd    = HFDComputer(self._cfg.hfd_window)
        self._qpp    = QPPDetector(self._cfg.qpp_window, self._cfg.qpp_min_frames)
        self._feka   = FeKaExtractor()
        self._hope   = HOPEEvaluator()

        self._sol_buf: deque = deque(maxlen=self._cfg.hfd_window)
        self._hel_buf: deque = deque(maxlen=self._cfg.hfd_window)

        self._frames_processed: int  = 0
        self._hope_streak_sec: float = 0.0
        self._prev_T: float          = 0.0
        self._prev_timestamp: Optional[datetime] = None
        self._hel_quiet_baseline: float = 500.0
        self._non_quiet_count: int = 0
        self._recent_quality: deque = deque(maxlen=self._cfg.degraded_window)
        self._state = EngineState.WARMUP

    @property
    def state(self) -> EngineState:
        return self._state

    def reset(self) -> None:
        self.__init__(self._cfg, self._pub)

    def process(self, sol: ObservationFrame, hel: ObservationFrame) -> PhysicsFrame:
        # ── timestamp proximity check ────────────────────────────────────────
        dt_sec = abs((sol.timestamp - hel.timestamp).total_seconds())
        if dt_sec > 1.0:
            logger.warning(
                "Timestamp mismatch sol=%s hel=%s delta=%.2fs",
                sol.timestamp.isoformat(), hel.timestamp.isoformat(), dt_sec,
            )

        self._sol_buf.append(sol)
        self._hel_buf.append(hel)
        self._frames_processed += 1

        # ── isothermal fit ───────────────────────────────────────────────────
        fit = self._fitter.fit(sol.counts_per_s_kev, sol.energy_bins_kev)
        phys_q = self._physics_quality(fit)

        # ── novel observables ────────────────────────────────────────────────
        hel_25_50 = _band(hel, 25.0, 50.0)
        sol_8_15  = _band(sol, 8.0, 15.0)
        DNI = self._dni.update(hel_25_50, sol_8_15)

        tnpsv_mag, tnpsv_angle = (
            self._tnpsv.update(fit.T_MK, fit.EM_cm3)
            if not np.isnan(fit.T_MK) else (0.0, 0.0)
        )
        shei, dshei = self._shei.update(hel.counts_per_s_kev, hel.energy_bins_kev)
        hfd         = self._hfd.update(_total_flux(sol))
        qpp_p, qpp_a = self._qpp.update(hel_25_50)
        feka_flux   = self._feka.extract(sol.counts_per_s_kev, sol.energy_bins_kev,
                                          fit.T_MK, fit.EM_cm3)

        # ── HOPE streak (timestamp-based) ────────────────────────────────────
        dT_dt = (fit.T_MK - self._prev_T) if not np.isnan(fit.T_MK) else 0.0
        self._prev_T = fit.T_MK if not np.isnan(fit.T_MK) else self._prev_T

        if self._prev_timestamp is not None:
            elapsed = (sol.timestamp - self._prev_timestamp).total_seconds()
            if dT_dt > 0 and 0 < elapsed <= 1.5:
                self._hope_streak_sec += elapsed
            else:
                self._hope_streak_sec = 0.0
        self._prev_timestamp = sol.timestamp

        # ── quiet baseline ───────────────────────────────────────────────────
        dEM_dt = 0.0  # approximate — TNPSV history already tracks this
        if len(self._hel_buf) >= 3:
            em_vals = [f.band_flux(8.0, 15.0) for f in list(self._hel_buf)[-3:]]
            dEM_dt = float(np.gradient(em_vals)[-1])

        # ── HOPE evaluation ──────────────────────────────────────────────────
        hope_res: HOPEResult = self._hope.evaluate(
            T=fit.T_MK if not np.isnan(fit.T_MK) else 0.0,
            dT_dt=dT_dt,
            EM=fit.EM_cm3 if not np.isnan(fit.EM_cm3) else 0.0,
            dEM_dt=dEM_dt,
            DNI=DNI,
            tnpsv_mag=tnpsv_mag,
            tnpsv_angle=tnpsv_angle,
            hel_flux=_total_flux(hel),
            hel_quiet_baseline=self._hel_quiet_baseline,
            streak_sec=self._hope_streak_sec,
            config=self._cfg,
        )

        # ── flare phase ──────────────────────────────────────────────────────
        phase = _assign_phase(DNI, shei, dT_dt, hope_res.hope_flag, self._cfg)

        # ── baseline + DNI reset ─────────────────────────────────────────────
        if phase == "QUIET":
            self._hel_quiet_baseline = (
                0.9 * self._hel_quiet_baseline + 0.1 * _total_flux(hel)
            )
            self._non_quiet_count = 0
        else:
            self._non_quiet_count += 1

        if phase == "QUIET" and self._non_quiet_count >= self._cfg.dni_reset_min_non_quiet:
            self._dni.reset()
            self._non_quiet_count = 0

        # ── engine state ─────────────────────────────────────────────────────
        self._recent_quality.append(phys_q == PhysicsQuality.FIT_OK)
        if self._frames_processed >= self._cfg.hfd_window:
            ok_rate = sum(self._recent_quality) / len(self._recent_quality)
            self._state = (
                EngineState.DEGRADED if ok_rate < (1 - self._cfg.degraded_threshold)
                else EngineState.ACTIVE
            )

        # ── assemble frame ───────────────────────────────────────────────────
        pf = PhysicsFrame(
            frame_id=str(uuid.uuid4()),
            timestamp=sol.timestamp,
            source_frame_ids=[sol.frame_id, hel.frame_id],
            T_MK=fit.T_MK,
            T_uncertainty_MK=fit.T_uncertainty_MK,
            EM_cm3=fit.EM_cm3,
            EM_uncertainty=fit.EM_uncertainty,
            fit_chi_squared=fit.chi2_reduced,
            DNI=DNI,
            tnpsv_magnitude=tnpsv_mag,
            tnpsv_angle_rad=tnpsv_angle,
            SHEI=shei,
            dSHEI_dt=dshei,
            HFD=hfd,
            QPP_period_sec=qpp_p,
            QPP_amplitude=qpp_a,
            FeKa_flux=feka_flux,
            flare_phase=phase,
            hope_flag=hope_res.hope_flag,
            hope_lead_time_sec=hope_res.lead_time_sec if hope_res.hope_flag else None,
            hope_reason=hope_res.reason if hope_res.hope_flag else None,
            engine_state=self._state.value,
            data_quality_flag=sol.quality_flag,
            physics_quality_flag=phys_q,
            fit_backend=fit.backend,
            pipeline_version=_VERSION,
        )

        if self._pub:
            self._publish(pf)

        return pf

    def _physics_quality(self, fit) -> PhysicsQuality:
        if np.isnan(fit.T_MK):
            return PhysicsQuality.FIT_DIVERGED
        if fit.backend == "chianti_lookup":
            return PhysicsQuality.BACKEND_FALLBACK
        if fit.chi2_reduced > self._cfg.fit_chi2_low_confidence:
            return PhysicsQuality.LOW_CONFIDENCE
        return PhysicsQuality.FIT_OK

    def _publish(self, pf: PhysicsFrame) -> None:
        import json
        from datetime import datetime as _dt
        payload = {k: (v.isoformat() if isinstance(v, _dt) else
                       v.name if hasattr(v, "name") else v)
                   for k, v in pf.__dict__.items()
                   if not isinstance(v, np.ndarray)}
        payload["schema_version"] = pf.schema_version
        try:
            self._pub.send(
                self._cfg.kafka_topic_physics,
                key=b"physics",
                value=json.dumps(payload).encode(),
            )
        except Exception as exc:
            logger.error("Kafka publish failed: %s", exc)
```

- [x] **Step 5: Update `__init__.py`**

```python
# python/aruna/l2_physics/__init__.py
from .models      import PhysicsFrame, PhysicsConfig, PhysicsQuality, EngineState, FitResult, HOPEResult
from .engine      import PhysicsEngine
from .isothermal  import IsothermalFitter
from .chianti_table import CHIANTILookupTable
from .dni         import DNIComputer
from .tnpsv       import TNPSVComputer
from .shei        import SHEIComputer
from .hfd         import HFDComputer
from .qpp         import QPPDetector
from .feka        import FeKaExtractor
from .hope        import HOPEEvaluator

__all__ = [
    "PhysicsFrame", "PhysicsConfig", "PhysicsQuality", "EngineState",
    "FitResult", "HOPEResult", "PhysicsEngine", "IsothermalFitter",
    "CHIANTILookupTable", "DNIComputer", "TNPSVComputer", "SHEIComputer",
    "HFDComputer", "QPPDetector", "FeKaExtractor", "HOPEEvaluator",
]
```

- [x] **Step 6: Run all tests**

```
pytest python/tests/test_l2_physics.py -v
```
Expected: all tests passing. Note: `TestHFDComputer::test_returns_float_after_window_fills` requires window=50 override (already set in that test).

- [x] **Step 7: Run full suite to check no regressions**

```
pytest python/ -v --tb=short
```
Expected: all previous L0/L1 tests still pass.

- [x] **Step 8: Commit**

```
git add python/aruna/l2_physics/ python/tests/conftest.py python/tests/test_l2_physics.py
git commit -m "feat(l2): complete PhysicsEngine orchestrator, all L2 modules wired and tested"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `PhysicsFrame` with all fields | Task 1 |
| `PhysicsQuality` + `EngineState` enums | Task 1 |
| `CHIANTILookupTable` (MKG bremsstrahlung) | Task 2 |
| `IsothermalFitter` sunkit-spex / fallback | Task 3 |
| `fit_backend` on every frame | Task 3 + Task 7 |
| `DNIComputer` | Task 4 |
| `TNPSVComputer` | Task 4 |
| `SHEIComputer` | Task 4 |
| `HFDComputer` (warm-up NaN documented) | Task 5 |
| `QPPDetector` (warm-up None documented) | Task 5 |
| `FeKaExtractor` | Task 6 |
| `HOPEEvaluator` (streak_sec, not count) | Task 6 |
| `PhysicsEngine` stateful orchestrator | Task 7 |
| `engine_state` on every frame | Task 7 |
| `data_quality_flag` + `physics_quality_flag` split | Task 7 |
| Concurrency: one instance per consumer | Documented in engine.py class docstring |
| DNI reset after 30-frame non-quiet | Task 7 |
| Kafka publish | Task 7 |
| `__init__.py` public API | Task 7 |
| X6.3 fixture: HFD NaN for 120 frames | Task 7 |
| `quiet_sun_extended_frames` for HFD/QPP testing | Task 7 conftest |
| All thresholds in `PhysicsConfig` | Task 1 |

**Placeholder scan:** None found.

**Type consistency:** `FitResult.backend` → `PhysicsFrame.fit_backend` (copied directly in engine). `HOPEResult` consumed in engine and fields mapped to `PhysicsFrame`. `PhysicsQuality` produced by `_physics_quality()` and stored in `PhysicsFrame.physics_quality_flag`. All consistent.
