# Session 7 Design Specification: L5 Analogs + L6 Threat + L7 LLM Briefing

## L6 Threat Intelligence Engine
- **`rules.py`**:
  - `NAVIC_BASE_RISK` maps GOES bands (B, C, M1, M5, X1, X5) to risk levels ('NEGLIGIBLE', 'LOW', 'MODERATE', 'HIGH', 'VERY_HIGH', 'EXTREME') and base probabilities.
  - `HF_BLACKOUT` maps bands to blackout duration and severity labels ('NONE', 'MINOR', 'MODERATE', 'STRONG', 'SEVERE', 'COMPLETE').
  - `GAGANYAAN` maps bands to advisory strings ('NOMINAL', 'ELEVATED', 'HIGH — SHELTER ADVISORY', 'CRITICAL — SHELTER REQUIRED', 'EMERGENCY — EVA ABORT').
  - `SAFING_REGISTRY` contains ISRO operational satellite objects.
- **`navic.py`**:
  - `compute_navic_risk` maps GOES subclasses into bands and adjusts risk levels based on confidence, flare phase, flux midpoint multiplier, and clamps X-class events to at least `VERY_HIGH`.
- **`bulletin.py`**:
  - `format_bulletin` builds SWMC-formatted operational text bulletins.
- **`__init__.py`**:
  - `ThreatOrchestrator` integrates all rules and calculations.

## L5 Historical Analog Retrieval Engine
- **`embedder.py`**:
  - `PhysicsFeatureEmbedder` extracts 17 physics features from `NowcastEvent` metadata or `PhysicsFrame` data and normalizes them via `StandardScaler` and L2 normalization.
- **`index.py`**:
  - `AnalogIndex` manages a FAISS index, scaling statistics, and automatic indexing.
- **`dtw.py`**:
  - Runs Dynamic Time Warping (DTW) on raw timeseries of top-10 candidates to find the top-3 best matches.

## L7 LLM Briefing Narrator
- **`client.py`**:
  - Supports Azure OpenAI, Mock, and Ollama clients conforming to `BriefingLLM` protocol.
- **`guard.py`**:
  - Validates that the generated briefing contains the required fields and no novel numbers (hallucination guard).
- **`template.py`**:
  - Offers a deterministic fallback formatter when LLM fails or is offline.
