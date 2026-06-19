from __future__ import annotations
import json
import os
import logging
from typing import Protocol, runtime_checkable

from aruna.l7_llm.models import BriefingContext, BriefingJSON

logger = logging.getLogger(__name__)


class BriefingLLMError(Exception):
    """Raised when the LLM client fails."""


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

    Required env vars:
      AZURE_OPENAI_ENDPOINT   = https://uaenorth.api.cognitive.microsoft.com/
      AZURE_OPENAI_API_KEY    = <your key>

    Optional env vars (override constructor defaults):
      AZURE_OPENAI_API_VERSION      = 2024-12-01-preview
      AZURE_OPENAI_MINI_DEPLOYMENT  = gpt-4-1-mini
    """
    def __init__(
        self,
        model: str = "gpt-4-1-mini",
        api_version: str = "2024-12-01-preview",
        temperature: float = 0.1,
        max_tokens: int = 400,
    ) -> None:
        # Env vars override constructor args
        self.model       = os.environ.get("AZURE_OPENAI_MINI_DEPLOYMENT", model)
        self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", api_version)
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def generate(self, system_prompt: str, user_prompt: str, ctx: BriefingContext) -> BriefingJSON:
        try:
            from langchain_openai import AzureChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
        except ImportError as exc:
            raise BriefingLLMError("langchain-openai not installed") from exc

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key  = os.environ.get("AZURE_OPENAI_API_KEY")
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
