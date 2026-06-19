"""
Kafka adapter for L7 — assembles BriefingContext from four upstream topics and
publishes AlertBriefing to aruna.alerts.briefings.
Kept separate from BriefingOrchestrator so the orchestrator stays pure/testable.
"""
from __future__ import annotations
import json
import logging

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
