# L7 LLM Briefing Engine — Design Spec

**Date:** 2026-06-19
**Status:** Approved
**PRD reference:** §11

---

## Context

L7 is the final layer of the ARUNA pipeline. It receives outputs from L3 (NowcastEvent), L4 (ForecastFrame), L5 (AnalogMatch list), and L6 (ThreatOutput), assembles them into a `BriefingContext`, and produces a human-readable `AlertBriefing` for mission operations officers.

The LLM narrates — it never computes science. Every number in the output must trace back to a field in `BriefingContext`. If the LLM fails or hallucinates, the deterministic fallback template always produces a valid briefing.

---

## File Structure

```
python/aruna/l7_llm/
  __init__.py       ← BriefingOrchestrator (pure, synchronous façade)
  models.py         ← BriefingContext, BriefingJSON, AlertBriefing dataclasses
  client.py         ← BriefingLLM protocol + AzureLangChainClient + MockLLMClient
  factory.py        ← LLMFactory.from_config(config) -> BriefingLLM
  prompt.py         ← PromptBuilder — assembles system + user prompt, versioned
  guard.py          ← structured field validation against BriefingContext
  renderer.py       ← BriefingJSON -> human-readable bulletin string
  template.py       ← deterministic fallback formatter (no LLM)
  kafka.py          ← Kafka consumer/producer (isolated from orchestrator)

python/aruna/l5_analogs/embedder.py  ← add FlareEmbedder Protocol (5-line stub)
```

---

## Data Models

```python
# l7_llm/models.py

@dataclass
class BriefingContext:
    nowcast: NowcastEvent
    forecast: ForecastFrame
    threat: ThreatOutput
    analogs: list[AnalogMatch]          # top-3, or [NO_ANALOG]

@dataclass
class BriefingJSON:
    """Structured output from the LLM — validated before rendering."""
    goes_class: str                     # must match ctx.nowcast.goes_label exactly
    confidence_pct: float               # must be within ±2 of ctx.nowcast.confidence * 100
    lead_time_min: float                # must be within ±2 of ctx.forecast.lead_time_estimate_sec / 60
    navic_risk_summary: str             # must be a value present in ctx.threat.navic_risk
    hf_risk: str                        # must match ctx.threat.hf_risk exactly
    hf_duration_min: int                # must match ctx.threat.hf_duration_min exactly
    radiation_risk: str                 # must match ctx.threat.radiation_risk exactly
    gaganyaan_advisory: str             # must match ctx.threat.gaganyaan_advisory exactly
    summary: str                        # free narrative (≤ 3 sentences)
    recommended_action: str             # free narrative (≤ 2 sentences)
    operator_note: str                  # free narrative (≤ 1 sentence)
    analog_class: str                   # goes_class of top analog, or "NONE"
    analog_similarity_pct: float        # similarity * 100, or 0.0

@dataclass
class AlertBriefing:
    briefing_text: str
    briefing_source: str                # "azure_gpt-4.1-mini" | "azure_gpt-4.1" | "deterministic"
    hallucination_violations: list[str] # empty if guard passed
    context_used: BriefingContext
    created_at: str                     # ISO 8601 UTC
```

---

## Components

### `factory.py` — LLMFactory

```python
class LLMFactory:
    @staticmethod
    def from_config(config: dict) -> BriefingLLM:
        provider = config.get("llm", {}).get("provider", "mock")
        model    = config.get("llm", {}).get("model", "gpt-4.1-mini")
        if provider == "azure":
            return AzureLangChainClient(model=model)
        elif provider == "ollama":
            return OllamaClient(model=model)
        else:
            return MockLLMClient()
```

Config (`config.yaml`):
```yaml
llm:
  provider: azure          # azure | ollama | mock
  model: gpt-4.1-mini      # or gpt-4.1
  api_version: "2024-02-01"
  max_tokens: 400
  temperature: 0.1
```

Env vars read by `AzureLangChainClient`: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`.

---

### `client.py` — BriefingLLM protocol

```python
class BriefingLLM(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> BriefingJSON: ...

class AzureLangChainClient:
    """Uses langchain_openai.AzureChatOpenAI with response_format=json_object."""
    def generate(self, system_prompt: str, user_prompt: str) -> BriefingJSON:
        llm = AzureChatOpenAI(
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_deployment=self.model,
            api_version=self.api_version,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
        # Raises BriefingLLMError on network failure after 3 retries
        ...

class MockLLMClient:
    """Returns a hardcoded valid BriefingJSON. Used in all tests."""
    def generate(self, system_prompt: str, user_prompt: str) -> BriefingJSON:
        # Returns canned BriefingJSON matching whatever BriefingContext is passed
        ...
```

---

### `prompt.py` — PromptBuilder

```python
PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """
You are ARUNA's operational briefing module for ISRO mission operations.

RULES (non-negotiable):
- Use ONLY the values present in the CONTEXT JSON below.
- Do NOT estimate, infer, calculate, or invent any scientific values.
- If information is unavailable, write "unavailable".
- Output ONLY valid JSON matching the schema provided.
- Do NOT exceed 3 sentences for summary, 2 for recommended_action, 1 for operator_note.
"""

class PromptBuilder:
    version: str = PROMPT_VERSION

    def build(self, ctx: BriefingContext) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt)."""
        context_dict = {
            "goes_class": ctx.nowcast.goes_label,
            "confidence_pct": round(ctx.nowcast.confidence * 100, 1),
            "lead_time_min": round(ctx.forecast.lead_time_estimate_sec / 60, 1),
            "navic_risk": ctx.threat.navic_risk,
            "hf_risk": ctx.threat.hf_risk,
            "hf_duration_min": ctx.threat.hf_duration_min,
            "radiation_risk": ctx.threat.radiation_risk,
            "gaganyaan_advisory": ctx.threat.gaganyaan_advisory,
            "dos": ctx.threat.dos,
            "donts": ctx.threat.donts,
            "analog_class": ctx.analogs[0].goes_class if ctx.analogs[0].event_id != "NO_ANALOG" else "NONE",
            "analog_similarity_pct": round(ctx.analogs[0].similarity * 100, 1),
        }
        schema = {field: "..." for field in BriefingJSON.__dataclass_fields__}
        user_prompt = (
            f"CONTEXT:\n{json.dumps(context_dict, indent=2)}\n\n"
            f"OUTPUT SCHEMA:\n{json.dumps(schema, indent=2)}"
        )
        return SYSTEM_PROMPT, user_prompt
```

---

### `guard.py` — Structured field validator

```python
def validate(briefing: BriefingJSON, ctx: BriefingContext) -> tuple[bool, list[str]]:
    violations = []

    # goes_class must match exactly
    if briefing.goes_class != ctx.nowcast.goes_label:
        violations.append(f"goes_class mismatch: {briefing.goes_class} != {ctx.nowcast.goes_label}")

    # confidence within ±2%
    expected_conf = round(ctx.nowcast.confidence * 100, 1)
    if abs(briefing.confidence_pct - expected_conf) > 2.0:
        violations.append(f"confidence_pct drift: {briefing.confidence_pct} vs {expected_conf}")

    # lead_time within ±2 min
    expected_lt = ctx.forecast.lead_time_estimate_sec / 60
    if abs(briefing.lead_time_min - expected_lt) > 2.0:
        violations.append(f"lead_time_min drift: {briefing.lead_time_min} vs {expected_lt:.1f}")

    # hf_duration_min must match exactly
    if briefing.hf_duration_min != ctx.threat.hf_duration_min:
        violations.append(f"hf_duration_min mismatch: {briefing.hf_duration_min} != {ctx.threat.hf_duration_min}")

    # navic_risk_summary must be a value in the threat navic_risk dict
    allowed_risks = set(ctx.threat.navic_risk.values())
    if briefing.navic_risk_summary not in allowed_risks:
        violations.append(f"navic_risk_summary '{briefing.navic_risk_summary}' not in {allowed_risks}")

    return len(violations) == 0, violations
```

---

### `renderer.py` — BriefingJSON → bulletin string

```python
def render(briefing: BriefingJSON, ctx: BriefingContext) -> str:
    """Converts validated structured output into a human-readable operations bulletin."""
    ...
```

Produces plain-English multi-section bulletin (class/confidence/lead time → NavIC/HF/radiation → Gaganyaan → actions → analog).

---

### `template.py` — Deterministic fallback

Always valid, no LLM, zero network dependency:

```
ARUNA SOLAR FLARE ALERT — {created_at} UTC
{goes_class} | Confidence: {confidence_pct}% | Lead Time: {lead_time_min} min

NavIC: {navic_risk_worst_region}
HF Comms: {hf_risk} blackout ~{hf_duration_min} min
Radiation: {radiation_risk} | Gaganyaan: {gaganyaan_advisory}

DO:   {dos}
AVOID: {donts}

Closest historical analog: {analog_class} ({analog_similarity_pct}% match)
```

---

### `__init__.py` — BriefingOrchestrator

```python
class BriefingOrchestrator:
    def __init__(self, llm: BriefingLLM | None = None, config: dict | None = None):
        self.llm = llm or LLMFactory.from_config(config or load_config())
        self.prompt_builder = PromptBuilder()

    def process(self, ctx: BriefingContext) -> AlertBriefing:
        system_prompt, user_prompt = self.prompt_builder.build(ctx)
        try:
            briefing_json = self.llm.generate(system_prompt, user_prompt)
            ok, violations = validate(briefing_json, ctx)
            if ok:
                text = render(briefing_json, ctx)
                source = f"azure_{self.llm.model}" if hasattr(self.llm, "model") else "llm"
            else:
                text = format_deterministic(ctx)
                source = "deterministic"
        except BriefingLLMError:
            ok = False
            violations = ["LLM unavailable"]
            text = format_deterministic(ctx)
            source = "deterministic"

        return AlertBriefing(
            briefing_text=text,
            briefing_source=source,
            hallucination_violations=[] if ok else violations,
            context_used=ctx,
            created_at=datetime.utcnow().isoformat() + "Z",
        )
```

---

## FlareEmbedder Protocol (L5 fix)

Added to `python/aruna/l5_analogs/embedder.py`:

```python
from typing import Protocol
import numpy as np

class FlareEmbedder(Protocol):
    def encode(self, item, event=None) -> np.ndarray: ...
```

5 lines, same file, no behavior change.

---

## Test Strategy (7 xfail → pass)

All tests use `MockLLMClient` — no Azure call in CI. Azure key only needed for live demo.

| Test | Method |
|------|--------|
| `test_hallucination_guard_passes_on_all_test_alerts` | `MockLLMClient` returns valid canned `BriefingJSON`; guard passes |
| `test_briefing_generated_within_15_seconds` | `MockLLMClient` is instant; assert elapsed < 15s |
| `test_fallback_activates_when_guard_fails` | Inject `MockLLMClient` with bad confidence_pct; assert `source=="deterministic"` |
| `test_every_briefing_has_required_fields` | Assert `briefing_text` contains goes_class, navic, lead time strings |
| `test_no_novel_numbers_in_briefing` | Guard catches field drift before render; no novel numbers reach bulletin |
| `test_llm_runs_fully_offline` | `MockLLMClient` makes no network call; `monkeypatch` blocks socket if needed |
| `test_deterministic_fallback_valid_without_ollama` | Call `format_deterministic(ctx)` directly; assert non-empty with required fields |

---

## Dependencies to add

```
langchain-openai>=0.1
```

Add to `python/requirements.txt`.
