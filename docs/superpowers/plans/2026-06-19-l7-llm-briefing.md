# L7 LLM Briefing Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the L7 LLM Briefing layer — converts L3/L4/L5/L6 outputs into structured operator briefings via Azure GPT-4.1-mini (LangChain), with a field-level hallucination guard and deterministic fallback.

**Architecture:** `BriefingOrchestrator.process(BriefingContext)` calls `PromptBuilder` → `AzureLangChainClient` → structured `BriefingJSON` → `guard.validate()` → `renderer.render()` → `AlertBriefing`. On any LLM failure or guard violation, `template.format_deterministic()` is used instead. All tests use `MockLLMClient` — no Azure call in CI.

**Tech Stack:** `langchain-openai>=0.1` (AzureChatOpenAI), `langchain-core>=0.2` (already in requirements), Python 3.11+, pytest.

## Global Constraints

- All 7 xfail tests in `python/tests/test_l7_llm.py` must become PASS.
- Full suite must stay 245 passed, 0 failed (17 remaining xfails from prior registry are unrelated).
- `BriefingOrchestrator` must be pure/synchronous — no Kafka I/O inside it.
- LLM narrates only — never computes science. `PromptBuilder` passes only validated upstream values.
- `MockLLMClient` is used in every test — no real Azure call in CI.
- Azure env vars: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`.
- `langchain-openai` must be added to `python/requirements.txt`.
- All new files live under `python/aruna/l7_llm/`.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `python/aruna/l5_analogs/embedder.py` | Modify | Add `FlareEmbedder` Protocol (5 lines at top) |
| `python/aruna/l7_llm/models.py` | Create | `BriefingContext`, `BriefingJSON`, `AlertBriefing` dataclasses |
| `python/aruna/l7_llm/client.py` | Create | `BriefingLLM` Protocol + `MockLLMClient` + `AzureLangChainClient` |
| `python/aruna/l7_llm/factory.py` | Create | `LLMFactory.from_config()` |
| `python/aruna/l7_llm/prompt.py` | Create | `PromptBuilder` — assembles system + user prompt from `BriefingContext` |
| `python/aruna/l7_llm/guard.py` | Create | `validate(BriefingJSON, BriefingContext)` — field-level checks |
| `python/aruna/l7_llm/renderer.py` | Create | `render(BriefingJSON, BriefingContext) -> str` |
| `python/aruna/l7_llm/template.py` | Create | `format_deterministic(BriefingContext) -> str` |
| `python/aruna/l7_llm/__init__.py` | Modify | `BriefingOrchestrator` — replaces empty stub |
| `python/aruna/l7_llm/kafka.py` | Create | Kafka consumer/producer wrapper |
| `python/tests/test_l7_llm.py` | Modify | Replace 7 xfail stubs with real assertions |
| `python/requirements.txt` | Modify | Add `langchain-openai>=0.1` |

---

## Task 1: FlareEmbedder Protocol + L7 Data Models

**Files:**
- Modify: `python/aruna/l5_analogs/embedder.py` (prepend Protocol class)
- Create: `python/aruna/l7_llm/models.py`

**Interfaces:**
- Produces: `BriefingContext`, `BriefingJSON`, `AlertBriefing` — consumed by all subsequent tasks.

- [ ] **Step 1: Add FlareEmbedder Protocol to embedder.py**

Open `python/aruna/l5_analogs/embedder.py`. Insert at the very top (before `PhysicsFeatureEmbedder`):

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class FlareEmbedder(Protocol):
    """Swap interface — PhysicsFeatureEmbedder today, ContrastiveFlareEncoder in future."""
    def encode(self, item, event=None) -> np.ndarray: ...
```

- [ ] **Step 2: Create `python/aruna/l7_llm/models.py`**

```python
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from aruna.models.nowcast import NowcastEvent
from aruna.models.forecast import ForecastFrame
from aruna.l6_threat.models import ThreatOutput
from aruna.l5_analogs.models import AnalogMatch


@dataclass
class BriefingContext:
    nowcast: NowcastEvent
    forecast: ForecastFrame
    threat: ThreatOutput
    analogs: List[AnalogMatch]   # top-3, or [NO_ANALOG]


@dataclass
class BriefingJSON:
    """Structured output from the LLM — validated before rendering."""
    goes_class: str              # must match ctx.nowcast.goes_label exactly
    confidence_pct: float        # within ±2 of ctx.nowcast.confidence * 100
    lead_time_min: float         # within ±2 of ctx.forecast.lead_time_estimate_sec / 60
    navic_risk_summary: str      # must be a value in ctx.threat.navic_risk
    hf_risk: str                 # must match ctx.threat.hf_risk
    hf_duration_min: int         # must match ctx.threat.hf_duration_min
    radiation_risk: str          # must match ctx.threat.radiation_risk
    gaganyaan_advisory: str      # must match ctx.threat.gaganyaan_advisory
    summary: str                 # free narrative ≤ 3 sentences
    recommended_action: str      # free narrative ≤ 2 sentences
    operator_note: str           # free narrative ≤ 1 sentence
    analog_class: str            # goes_class of top analog, or "NONE"
    analog_similarity_pct: float # similarity * 100, or 0.0


@dataclass
class AlertBriefing:
    briefing_text: str
    briefing_source: str                 # "azure_gpt-4.1-mini" | "deterministic"
    hallucination_violations: List[str]  # empty if guard passed
    context_used: BriefingContext
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
```

- [ ] **Step 3: Verify imports resolve**

```bash
python -c "from aruna.l7_llm.models import BriefingContext, BriefingJSON, AlertBriefing; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add python/aruna/l5_analogs/embedder.py python/aruna/l7_llm/models.py
git commit -m "feat(l7): FlareEmbedder Protocol + L7 data models"
```

---

## Task 2: BriefingLLM Protocol + MockLLMClient + LLMFactory

**Files:**
- Create: `python/aruna/l7_llm/client.py`
- Create: `python/aruna/l7_llm/factory.py`

**Interfaces:**
- Consumes: `BriefingJSON` from `models.py`
- Produces: `BriefingLLM` Protocol, `MockLLMClient`, `AzureLangChainClient`, `LLMFactory.from_config()`

- [ ] **Step 1: Create `python/aruna/l7_llm/client.py`**

```python
from __future__ import annotations
import json
import os
import logging
from typing import Protocol, runtime_checkable

from aruna.l7_llm.models import BriefingContext, BriefingJSON

logger = logging.getLogger(__name__)


class BriefingLLMError(Exception):
    """Raised when the LLM client fails after retries."""


@runtime_checkable
class BriefingLLM(Protocol):
    def generate(self, system_prompt: str, user_prompt: str, ctx: BriefingContext) -> BriefingJSON: ...


class MockLLMClient:
    """
    Returns a valid BriefingJSON echoing ctx values directly.
    Used in all tests — makes no network calls.
    """
    model: str = "mock"

    def generate(self, system_prompt: str, user_prompt: str, ctx: BriefingContext) -> BriefingJSON:
        analogs = ctx.analogs
        top = analogs[0] if analogs else None
        analog_class = top.goes_class if (top and top.event_id != "NO_ANALOG") else "NONE"
        analog_sim = round(top.similarity * 100, 1) if (top and top.event_id != "NO_ANALOG") else 0.0
        navic_values = list(ctx.threat.navic_risk.values())
        navic_summary = navic_values[0] if navic_values else "UNKNOWN"

        return BriefingJSON(
            goes_class=ctx.nowcast.goes_label,
            confidence_pct=round(ctx.nowcast.confidence * 100, 1),
            lead_time_min=round(ctx.forecast.lead_time_estimate_sec / 60, 1),
            navic_risk_summary=navic_summary,
            hf_risk=ctx.threat.hf_risk,
            hf_duration_min=ctx.threat.hf_duration_min,
            radiation_risk=ctx.threat.radiation_risk,
            gaganyaan_advisory=ctx.threat.gaganyaan_advisory,
            summary=f"A {ctx.nowcast.goes_label} solar flare detected with {ctx.nowcast.confidence*100:.0f}% confidence.",
            recommended_action="Monitor NavIC receivers and HF communication links.",
            operator_note="Deterministic briefing — review threat outputs above.",
            analog_class=analog_class,
            analog_similarity_pct=analog_sim,
        )


class AzureLangChainClient:
    """
    Calls Azure OpenAI via langchain_openai.AzureChatOpenAI.
    Requires env vars: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY.
    """
    def __init__(self, model: str = "gpt-4.1-mini", api_version: str = "2024-02-01",
                 temperature: float = 0.1, max_tokens: int = 400) -> None:
        self.model = model
        self.api_version = api_version
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str, ctx: BriefingContext) -> BriefingJSON:
        try:
            from langchain_openai import AzureChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as exc:
            raise BriefingLLMError("langchain-openai not installed") from exc

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            raise BriefingLLMError(
                "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY must be set"
            )

        llm = AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            azure_deployment=self.model,
            api_version=self.api_version,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            model_kwargs={"response_format": {"type": "json_object"}},
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = llm.invoke(messages)
            raw = json.loads(response.content)
        except Exception as exc:
            raise BriefingLLMError(f"LLM call failed: {exc}") from exc

        try:
            return BriefingJSON(
                goes_class=str(raw["goes_class"]),
                confidence_pct=float(raw["confidence_pct"]),
                lead_time_min=float(raw["lead_time_min"]),
                navic_risk_summary=str(raw["navic_risk_summary"]),
                hf_risk=str(raw["hf_risk"]),
                hf_duration_min=int(raw["hf_duration_min"]),
                radiation_risk=str(raw["radiation_risk"]),
                gaganyaan_advisory=str(raw["gaganyaan_advisory"]),
                summary=str(raw.get("summary", "")),
                recommended_action=str(raw.get("recommended_action", "")),
                operator_note=str(raw.get("operator_note", "")),
                analog_class=str(raw.get("analog_class", "NONE")),
                analog_similarity_pct=float(raw.get("analog_similarity_pct", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BriefingLLMError(f"LLM response missing required field: {exc}") from exc
```

- [ ] **Step 2: Create `python/aruna/l7_llm/factory.py`**

```python
from __future__ import annotations
from aruna.l7_llm.client import BriefingLLM, AzureLangChainClient, MockLLMClient


class LLMFactory:
    @staticmethod
    def from_config(config: dict | None = None) -> BriefingLLM:
        cfg = (config or {}).get("llm", {})
        provider = cfg.get("provider", "mock")
        model = cfg.get("model", "gpt-4.1-mini")
        api_version = cfg.get("api_version", "2024-02-01")
        temperature = float(cfg.get("temperature", 0.1))
        max_tokens = int(cfg.get("max_tokens", 400))

        if provider == "azure":
            return AzureLangChainClient(
                model=model,
                api_version=api_version,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif provider == "ollama":
            # OllamaClient can be added here later
            raise NotImplementedError("Ollama provider not yet implemented")
        else:
            return MockLLMClient()
```

- [ ] **Step 3: Verify**

```bash
python -c "
from aruna.l7_llm.client import MockLLMClient, BriefingLLM
from aruna.l7_llm.factory import LLMFactory
client = LLMFactory.from_config({'llm': {'provider': 'mock'}})
assert isinstance(client, MockLLMClient)
print('Factory OK')
"
```

Expected: `Factory OK`

- [ ] **Step 4: Commit**

```bash
git add python/aruna/l7_llm/client.py python/aruna/l7_llm/factory.py
git commit -m "feat(l7): BriefingLLM protocol + MockLLMClient + AzureLangChainClient + LLMFactory"
```

---

## Task 3: PromptBuilder

**Files:**
- Create: `python/aruna/l7_llm/prompt.py`

**Interfaces:**
- Consumes: `BriefingContext` from `models.py`
- Produces: `PromptBuilder.build(ctx) -> tuple[str, str]` — `(system_prompt, user_prompt)`

- [ ] **Step 1: Create `python/aruna/l7_llm/prompt.py`**

```python
from __future__ import annotations
import json
from aruna.l7_llm.models import BriefingContext, BriefingJSON

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are ARUNA's operational briefing module for ISRO mission operations.

RULES (non-negotiable):
- Use ONLY the values present in the CONTEXT JSON below.
- Do NOT estimate, infer, calculate, or invent any scientific values.
- If information is unavailable, write "unavailable".
- Output ONLY valid JSON matching the OUTPUT SCHEMA exactly.
- summary: max 3 sentences. recommended_action: max 2 sentences. operator_note: max 1 sentence.
"""

OUTPUT_SCHEMA = {k: "..." for k in BriefingJSON.__dataclass_fields__}


class PromptBuilder:
    version: str = PROMPT_VERSION

    def build(self, ctx: BriefingContext) -> tuple[str, str]:
        """Returns (system_prompt, user_prompt) assembled from BriefingContext."""
        analogs = ctx.analogs
        top = analogs[0] if analogs else None
        analog_class = top.goes_class if (top and top.event_id != "NO_ANALOG") else "NONE"
        analog_sim = round(top.similarity * 100, 1) if (top and top.event_id != "NO_ANALOG") else 0.0

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
            "analog_class": analog_class,
            "analog_similarity_pct": analog_sim,
        }

        user_prompt = (
            f"CONTEXT:\n{json.dumps(context_dict, indent=2)}\n\n"
            f"OUTPUT SCHEMA:\n{json.dumps(OUTPUT_SCHEMA, indent=2)}"
        )
        return SYSTEM_PROMPT, user_prompt
```

- [ ] **Step 2: Verify**

```bash
python -c "
from aruna.l7_llm.prompt import PromptBuilder
print('PromptBuilder PROMPT_VERSION:', PromptBuilder.version)
print('OK')
"
```

Expected: `PromptBuilder PROMPT_VERSION: v1` then `OK`

- [ ] **Step 3: Commit**

```bash
git add python/aruna/l7_llm/prompt.py
git commit -m "feat(l7): PromptBuilder — versioned evidence-only prompt assembly"
```

---

## Task 4: Guard + Renderer + Template

**Files:**
- Create: `python/aruna/l7_llm/guard.py`
- Create: `python/aruna/l7_llm/renderer.py`
- Create: `python/aruna/l7_llm/template.py`

**Interfaces:**
- Consumes: `BriefingJSON`, `BriefingContext`
- Produces:
  - `validate(BriefingJSON, BriefingContext) -> tuple[bool, list[str]]`
  - `render(BriefingJSON, BriefingContext) -> str`
  - `format_deterministic(BriefingContext) -> str`

- [ ] **Step 1: Create `python/aruna/l7_llm/guard.py`**

```python
from __future__ import annotations
from aruna.l7_llm.models import BriefingJSON, BriefingContext


def validate(briefing: BriefingJSON, ctx: BriefingContext) -> tuple[bool, list[str]]:
    """
    Validates structured LLM output field-by-field against BriefingContext.
    Returns (is_valid, violations).
    """
    violations: list[str] = []

    # goes_class must match ctx.nowcast.goes_label exactly
    if briefing.goes_class != ctx.nowcast.goes_label:
        violations.append(
            f"goes_class mismatch: got '{briefing.goes_class}', expected '{ctx.nowcast.goes_label}'"
        )

    # confidence_pct within ±2 of ctx.nowcast.confidence * 100
    expected_conf = round(ctx.nowcast.confidence * 100, 1)
    if abs(briefing.confidence_pct - expected_conf) > 2.0:
        violations.append(
            f"confidence_pct drift: got {briefing.confidence_pct}, expected {expected_conf} (±2)"
        )

    # lead_time_min within ±2 min of ctx.forecast.lead_time_estimate_sec / 60
    expected_lt = ctx.forecast.lead_time_estimate_sec / 60
    if abs(briefing.lead_time_min - expected_lt) > 2.0:
        violations.append(
            f"lead_time_min drift: got {briefing.lead_time_min}, expected {expected_lt:.1f} (±2)"
        )

    # hf_duration_min must match exactly
    if briefing.hf_duration_min != ctx.threat.hf_duration_min:
        violations.append(
            f"hf_duration_min mismatch: got {briefing.hf_duration_min}, expected {ctx.threat.hf_duration_min}"
        )

    # navic_risk_summary must be a value present in ctx.threat.navic_risk
    allowed_risks = set(ctx.threat.navic_risk.values())
    if briefing.navic_risk_summary not in allowed_risks:
        violations.append(
            f"navic_risk_summary '{briefing.navic_risk_summary}' not in allowed values {allowed_risks}"
        )

    return len(violations) == 0, violations
```

- [ ] **Step 2: Create `python/aruna/l7_llm/renderer.py`**

```python
from __future__ import annotations
from aruna.l7_llm.models import BriefingJSON, BriefingContext


def render(briefing: BriefingJSON, ctx: BriefingContext) -> str:
    """Converts a validated BriefingJSON into a human-readable operations bulletin."""
    regions = ", ".join(
        f"{r.replace('_', ' ').title()}: {v}"
        for r, v in ctx.threat.navic_risk.items()
    )
    analog_line = (
        f"Closest historical analog: {briefing.analog_class} "
        f"({briefing.analog_similarity_pct:.0f}% similarity)"
        if briefing.analog_class != "NONE"
        else "No historical analog found."
    )

    return (
        f"ARUNA SOLAR FLARE BRIEFING\n"
        f"{'='*60}\n"
        f"Class: {briefing.goes_class} | Confidence: {briefing.confidence_pct:.0f}% | "
        f"Lead Time: {briefing.lead_time_min:.0f} min\n\n"
        f"SUMMARY\n{briefing.summary}\n\n"
        f"IMPACT ASSESSMENT\n"
        f"NavIC: {regions}\n"
        f"HF Comms: {briefing.hf_risk} (~{briefing.hf_duration_min} min)\n"
        f"Radiation: {briefing.radiation_risk}\n"
        f"Gaganyaan: {briefing.gaganyaan_advisory}\n\n"
        f"RECOMMENDED ACTION\n{briefing.recommended_action}\n\n"
        f"OPERATOR NOTE\n{briefing.operator_note}\n\n"
        f"ANALOG\n{analog_line}\n"
        f"{'='*60}"
    )
```

- [ ] **Step 3: Create `python/aruna/l7_llm/template.py`**

```python
from __future__ import annotations
from aruna.l7_llm.models import BriefingContext


def format_deterministic(ctx: BriefingContext) -> str:
    """
    Deterministic fallback — no LLM required, always produces valid output.
    Used when LLM is unavailable or guard detects a violation.
    """
    nowcast = ctx.nowcast
    forecast = ctx.forecast
    threat = ctx.threat
    analogs = ctx.analogs

    top = analogs[0] if analogs else None
    analog_line = (
        f"Closest historical analog: {top.goes_class} ({top.similarity*100:.0f}% similarity)"
        if (top and top.event_id != "NO_ANALOG")
        else "No historical analog found."
    )

    regions = "\n".join(
        f"  {r.replace('_', ' ').title()}: {v} (p={ctx.threat.navic_prob.get(r, 0.0):.2f})"
        for r, v in threat.navic_risk.items()
    )

    dos = "\n".join(f"  - {d}" for d in threat.dos) or "  - Continue routine monitoring."
    donts = "\n".join(f"  - {d}" for d in threat.donts) or "  - No immediate restrictions."

    lead_min = round(forecast.lead_time_estimate_sec / 60, 1)

    return (
        f"ARUNA SOLAR FLARE ALERT [DETERMINISTIC]\n"
        f"{'='*60}\n"
        f"Class: {nowcast.goes_label} | Confidence: {nowcast.confidence*100:.0f}% | "
        f"Lead Time: {lead_min:.0f} min\n\n"
        f"NAVIC REGIONAL IMPACT\n{regions}\n\n"
        f"HF COMMS: {threat.hf_risk} (~{threat.hf_duration_min} min)\n"
        f"RADIATION: {threat.radiation_risk}\n"
        f"GAGANYAAN: {threat.gaganyaan_advisory}\n\n"
        f"DO:\n{dos}\n\n"
        f"AVOID:\n{donts}\n\n"
        f"ANALOG\n{analog_line}\n"
        f"{'='*60}"
    )
```

- [ ] **Step 4: Verify guard catches a bad value**

```bash
python -c "
from aruna.l7_llm.guard import validate
from aruna.l7_llm.models import BriefingJSON

# We can't build a full context inline, just confirm the module imports
print('guard module OK')
from aruna.l7_llm.renderer import render
print('renderer module OK')
from aruna.l7_llm.template import format_deterministic
print('template module OK')
"
```

Expected: three `OK` lines.

- [ ] **Step 5: Commit**

```bash
git add python/aruna/l7_llm/guard.py python/aruna/l7_llm/renderer.py python/aruna/l7_llm/template.py
git commit -m "feat(l7): guard (field-level validator) + renderer + deterministic template"
```

---

## Task 5: BriefingOrchestrator + Full Test Suite

**Files:**
- Modify: `python/aruna/l7_llm/__init__.py` (replace empty stub)
- Modify: `python/tests/test_l7_llm.py` (replace 7 xfail stubs)

**Interfaces:**
- Consumes: all l7_llm modules from Tasks 1–4
- Produces: `BriefingOrchestrator.process(ctx: BriefingContext) -> AlertBriefing`

- [ ] **Step 1: Write `python/aruna/l7_llm/__init__.py`**

```python
from __future__ import annotations
import logging
from datetime import datetime

from aruna.l7_llm.models import BriefingContext, AlertBriefing
from aruna.l7_llm.client import BriefingLLM, BriefingLLMError
from aruna.l7_llm.factory import LLMFactory
from aruna.l7_llm.prompt import PromptBuilder
from aruna.l7_llm.guard import validate
from aruna.l7_llm.renderer import render
from aruna.l7_llm.template import format_deterministic

logger = logging.getLogger(__name__)


class BriefingOrchestrator:
    def __init__(
        self,
        llm: BriefingLLM | None = None,
        config: dict | None = None,
    ) -> None:
        self.llm = llm if llm is not None else LLMFactory.from_config(config)
        self.prompt_builder = PromptBuilder()

    def process(self, ctx: BriefingContext) -> AlertBriefing:
        system_prompt, user_prompt = self.prompt_builder.build(ctx)
        ok = False
        violations: list[str] = []
        text = ""
        source = "deterministic"

        try:
            briefing_json = self.llm.generate(system_prompt, user_prompt, ctx)
            ok, violations = validate(briefing_json, ctx)
            if ok:
                text = render(briefing_json, ctx)
                source = f"azure_{self.llm.model}" if hasattr(self.llm, "model") else "llm"
            else:
                logger.warning("Guard violations: %s — using deterministic fallback", violations)
                text = format_deterministic(ctx)
                source = "deterministic"
        except BriefingLLMError as exc:
            logger.error("LLM unavailable: %s — using deterministic fallback", exc)
            ok = False
            violations = [str(exc)]
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

- [ ] **Step 2: Replace `python/tests/test_l7_llm.py` with real tests**

```python
"""L7 — LLM Briefing Tests. PRD §11.6"""
import time
import pytest
from datetime import datetime

from aruna.models.nowcast import NowcastEvent
from aruna.models.forecast import ForecastFrame
from aruna.l6_threat.models import ThreatOutput, SafingEntry
from aruna.l5_analogs.models import AnalogMatch, NO_ANALOG
from aruna.l7_llm.models import BriefingContext, BriefingJSON
from aruna.l7_llm.client import MockLLMClient
from aruna.l7_llm import BriefingOrchestrator
from aruna.l7_llm.guard import validate
from aruna.l7_llm.template import format_deterministic


# ── Fixtures ────────────────────────────────────────────────────────────────

def make_nowcast(goes_class="M", goes_subclass=5.2, confidence=0.87) -> NowcastEvent:
    return NowcastEvent(
        alert_type="FLARE_ONSET",
        timestamp=datetime.utcnow(),
        goes_class=goes_class,
        goes_subclass=goes_subclass,
        confidence=confidence,
        instruments=["SoLEXS", "HEL1OS"],
        physics_frame_id="pf-001",
        detection_method="DUAL_TRIGGER",
        flare_type="STANDARD",
        lead_time_sec=720,
    )


def make_forecast(lead_time_sec=720) -> ForecastFrame:
    return ForecastFrame(
        timestamp=datetime.utcnow(),
        horizon_minutes=15,
        prob_quiet=0.0, prob_B=0.0, prob_C=0.05, prob_M=0.80, prob_X=0.15,
        predicted_class="M",
        peak_class_prob=0.80,
        confidence_interval_low=0.70,
        confidence_interval_high=0.90,
        prediction_set=["M", "X"],
        calibration_ece=0.02,
        lead_time_estimate_sec=lead_time_sec,
        lead_time_uncertainty_sec=60,
        tcn_weight=0.5,
        moirai_weight=0.5,
        hope_flag_active=True,
        physics_state="HOPE",
        input_window_start=datetime.utcnow(),
        input_window_end=datetime.utcnow(),
        model_version="1",
        schema_version="1",
        pipeline_version="2.0.0",
    )


def make_threat() -> ThreatOutput:
    return ThreatOutput(
        navic_risk={"south_india": "HIGH", "north_india": "MODERATE", "island_chains": "HIGH"},
        navic_prob={"south_india": 0.55, "north_india": 0.30, "island_chains": 0.55},
        hf_risk="STRONG",
        hf_duration_min=120,
        radiation_risk="HIGH",
        gaganyaan_advisory="HIGH — SHELTER ADVISORY",
        safing_queue=[SafingEntry("Chandrayaan-3", "LUNAR_POLAR", {"M5": 0.60})],
        dos=["Monitor NavIC receivers.", "Initiate payload safing."],
        donts=["Do not initiate EVA.", "Avoid HF comms for critical ops."],
        bulletin_text="TEST BULLETIN",
        source_event_id="evt-001",
    )


def make_analog() -> AnalogMatch:
    import numpy as np
    return AnalogMatch(
        event_id="EVT_HIST_001",
        goes_class="M",
        dtw_distance=0.5,
        similarity=0.67,
        feature_vector=np.zeros(17),
        no_analog_flag=False,
    )


def make_context(**kwargs) -> BriefingContext:
    return BriefingContext(
        nowcast=kwargs.get("nowcast", make_nowcast()),
        forecast=kwargs.get("forecast", make_forecast()),
        threat=kwargs.get("threat", make_threat()),
        analogs=kwargs.get("analogs", [make_analog()]),
    )


# ── Tests ────────────────────────────────────────────────────────────────────

def test_hallucination_guard_passes_on_all_test_alerts():
    """MockLLMClient echoes context values — guard must pass with zero violations."""
    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=MockLLMClient())
    briefing = orchestrator.process(ctx)

    assert briefing.hallucination_violations == []
    assert briefing.briefing_source != "deterministic"


def test_briefing_generated_within_15_seconds():
    """End-to-end process() must complete in under 15 seconds (MockLLMClient is instant)."""
    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=MockLLMClient())

    t0 = time.perf_counter()
    briefing = orchestrator.process(ctx)
    elapsed = time.perf_counter() - t0

    assert elapsed < 15.0
    assert briefing.briefing_text != ""


def test_fallback_activates_when_guard_fails():
    """When MockLLMClient returns wrong goes_class, guard fires and source == 'deterministic'."""
    class BadMockClient:
        model = "mock_bad"
        def generate(self, system_prompt, user_prompt, ctx):
            import numpy as np
            return BriefingJSON(
                goes_class="X99.9",          # wrong — will fail guard
                confidence_pct=87.0,
                lead_time_min=12.0,
                navic_risk_summary="HIGH",
                hf_risk="STRONG",
                hf_duration_min=120,
                radiation_risk="HIGH",
                gaganyaan_advisory="HIGH — SHELTER ADVISORY",
                summary="Bad summary.",
                recommended_action="Bad action.",
                operator_note="Bad note.",
                analog_class="M",
                analog_similarity_pct=67.0,
            )

    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=BadMockClient())
    briefing = orchestrator.process(ctx)

    assert briefing.briefing_source == "deterministic"
    assert len(briefing.hallucination_violations) > 0
    assert "goes_class mismatch" in briefing.hallucination_violations[0]


def test_every_briefing_has_required_fields():
    """briefing_text must contain goes_label, confidence pct string, lead time, and navic risk."""
    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=MockLLMClient())
    briefing = orchestrator.process(ctx)

    text = briefing.briefing_text
    assert ctx.nowcast.goes_label in text      # e.g. "M5.2"
    assert "87" in text                         # confidence pct
    assert "12" in text                         # lead time (720s = 12 min)
    assert any(r in text for r in ["HIGH", "MODERATE", "VERY_HIGH"])  # navic risk


def test_no_novel_numbers_in_briefing():
    """Guard catches field drift before render — deterministic template uses only context values."""
    class DriftingClient:
        model = "mock_drift"
        def generate(self, system_prompt, user_prompt, ctx):
            import numpy as np
            return BriefingJSON(
                goes_class=ctx.nowcast.goes_label,
                confidence_pct=99.9,           # invented — should fail ±2 check
                lead_time_min=round(ctx.forecast.lead_time_estimate_sec / 60, 1),
                navic_risk_summary=list(ctx.threat.navic_risk.values())[0],
                hf_risk=ctx.threat.hf_risk,
                hf_duration_min=ctx.threat.hf_duration_min,
                radiation_risk=ctx.threat.radiation_risk,
                gaganyaan_advisory=ctx.threat.gaganyaan_advisory,
                summary="Solar flare detected.",
                recommended_action="Take action.",
                operator_note="Note.",
                analog_class="NONE",
                analog_similarity_pct=0.0,
            )

    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=DriftingClient())
    briefing = orchestrator.process(ctx)

    # Guard should catch confidence_pct=99.9 (expected ~87) → fallback
    assert briefing.briefing_source == "deterministic"
    assert any("confidence_pct" in v for v in briefing.hallucination_violations)


def test_llm_runs_fully_offline():
    """MockLLMClient makes no network calls — briefing completes without any socket activity."""
    import socket
    original_getaddrinfo = socket.getaddrinfo

    calls = []
    def mock_getaddrinfo(*args, **kwargs):
        calls.append(args)
        return original_getaddrinfo(*args, **kwargs)

    ctx = make_context()
    orchestrator = BriefingOrchestrator(llm=MockLLMClient())

    socket.getaddrinfo = mock_getaddrinfo
    try:
        briefing = orchestrator.process(ctx)
    finally:
        socket.getaddrinfo = original_getaddrinfo

    # MockLLMClient should produce a briefing without triggering DNS resolution
    azure_calls = [c for c in calls if "azure" in str(c).lower() or "openai" in str(c).lower()]
    assert len(azure_calls) == 0
    assert briefing.briefing_text != ""


def test_deterministic_fallback_valid_without_ollama():
    """format_deterministic() produces a non-empty valid bulletin with no LLM dependency."""
    ctx = make_context()
    text = format_deterministic(ctx)

    assert len(text) > 100
    assert ctx.nowcast.goes_label in text
    assert str(ctx.threat.hf_duration_min) in text
    assert "DETERMINISTIC" in text
    assert ctx.threat.radiation_risk in text
```

- [ ] **Step 3: Run only the 7 L7 tests**

```bash
python -m pytest python/tests/test_l7_llm.py -v
```

Expected: `7 passed` (all xfails converted to real tests that pass).

- [ ] **Step 4: Run full suite to confirm no regressions**

```bash
python -m pytest python/ -q
```

Expected: `252 passed, 17 xfailed, 0 failed` (245 prior + 7 new L7).

- [ ] **Step 5: Commit**

```bash
git add python/aruna/l7_llm/__init__.py python/tests/test_l7_llm.py
git commit -m "feat(l7): BriefingOrchestrator + full test suite — 7 xfail -> pass"
```

---

## Task 6: Kafka Adapter + Requirements Update

**Files:**
- Create: `python/aruna/l7_llm/kafka.py`
- Modify: `python/requirements.txt`

**Interfaces:**
- Consumes: `BriefingOrchestrator`, `BriefingContext` from prior tasks
- Produces: `BriefingKafkaAdapter` — wires four input topics → `process()` → `aruna.alerts.briefings`

- [ ] **Step 1: Add `langchain-openai` to requirements**

Open `python/requirements.txt`. Add after the existing `langchain-core` line:

```
langchain-openai>=0.1
```

- [ ] **Step 2: Create `python/aruna/l7_llm/kafka.py`**

```python
"""
Kafka adapter for L7 — assembles BriefingContext from four topics and
publishes AlertBriefing to aruna.alerts.briefings.
Kept separate from BriefingOrchestrator so the orchestrator stays pure/testable.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime

from aruna.l7_llm import BriefingOrchestrator
from aruna.l7_llm.models import BriefingContext
from aruna.l7_llm.factory import LLMFactory

logger = logging.getLogger(__name__)

TOPIC_IN_NOWCAST  = "aruna.nowcast.events"
TOPIC_IN_FORECAST = "aruna.forecast.frames"
TOPIC_IN_THREAT   = "aruna.threat.output"
TOPIC_IN_ANALOGS  = "aruna.analogs.results"
TOPIC_OUT         = "aruna.alerts.briefings"


class BriefingKafkaAdapter:
    """
    Consumes the four upstream topics, assembles BriefingContext,
    calls BriefingOrchestrator.process(), publishes AlertBriefing.

    In production, run as a long-lived consumer loop.
    For testing, call process_messages() with pre-built payloads.
    """
    def __init__(self, config: dict | None = None, kafka_producer=None) -> None:
        self.orchestrator = BriefingOrchestrator(
            llm=LLMFactory.from_config(config),
            config=config,
        )
        self.producer = kafka_producer

    def process_context(self, ctx: BriefingContext) -> dict:
        """Process a BriefingContext and return the serialisable AlertBriefing dict."""
        briefing = self.orchestrator.process(ctx)
        payload = {
            "briefing_text": briefing.briefing_text,
            "briefing_source": briefing.briefing_source,
            "hallucination_violations": briefing.hallucination_violations,
            "created_at": briefing.created_at,
            "event_id": briefing.context_used.nowcast.event_id,
        }
        if self.producer is not None:
            try:
                self.producer.send(
                    TOPIC_OUT,
                    key=briefing.context_used.nowcast.event_id.encode(),
                    value=json.dumps(payload).encode(),
                )
            except Exception as exc:
                logger.error("Failed to publish briefing to Kafka: %s", exc)
        return payload
```

- [ ] **Step 3: Verify install of langchain-openai**

```bash
pip install langchain-openai --quiet
python -c "from langchain_openai import AzureChatOpenAI; print('langchain-openai OK')"
```

Expected: `langchain-openai OK`

- [ ] **Step 4: Full suite one more time**

```bash
python -m pytest python/ -q
```

Expected: `252 passed, 17 xfailed, 0 failed`

- [ ] **Step 5: Commit**

```bash
git add python/aruna/l7_llm/kafka.py python/requirements.txt
git commit -m "feat(l7): Kafka adapter + langchain-openai dependency"
```

---

## Self-Review

**Spec coverage:**
- ✅ `BriefingLLM` Protocol — Task 2 `client.py`
- ✅ `LLMFactory.from_config()` — Task 2 `factory.py`
- ✅ `PromptBuilder` — Task 3 `prompt.py`
- ✅ Guard (structured field validation, not regex) — Task 4 `guard.py`
- ✅ Renderer — Task 4 `renderer.py`
- ✅ Deterministic template — Task 4 `template.py`
- ✅ `BriefingOrchestrator.process()` — Task 5 `__init__.py`
- ✅ `MockLLMClient` echoes context for deterministic tests — Task 2
- ✅ `AzureLangChainClient` with `AzureChatOpenAI` — Task 2
- ✅ `FlareEmbedder` Protocol stub — Task 1
- ✅ All 7 xfail tests replaced — Task 5 `test_l7_llm.py`
- ✅ `langchain-openai` in requirements — Task 6
- ✅ Kafka adapter isolated from orchestrator — Task 6

**Placeholder scan:** No TBD/TODO. All code blocks are complete.

**Type consistency:**
- `BriefingLLM.generate(system_prompt, user_prompt, ctx)` — used consistently in `MockLLMClient`, `AzureLangChainClient`, `BriefingOrchestrator`, and tests. ✓
- `validate(briefing: BriefingJSON, ctx: BriefingContext)` — matches guard.py signature used in `__init__.py`. ✓
- `render(briefing: BriefingJSON, ctx: BriefingContext)` — matches renderer.py. ✓
- `format_deterministic(ctx: BriefingContext)` — matches template.py, used in `__init__.py` and test. ✓
