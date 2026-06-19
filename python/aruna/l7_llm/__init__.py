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
