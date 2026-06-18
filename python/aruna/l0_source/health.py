import asyncio
import dataclasses
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import SourceStatus
from .session import PradanSession, SessionState
from .supplementary import ALL_SUPPLEMENTARY_CLIENTS

logger = logging.getLogger(__name__)

_PIPELINE_VERSION = "2.0.0"


class SourceHealthMonitor:
    def __init__(
        self,
        session: PradanSession,
        kafka_producer=None,
        kafka_topic: str = "aruna.source.health",
        poll_interval_sec: float = 30.0,
        health_file: Path = Path("data/health.json"),
        offline_fallback_sec: float = 300.0,
    ):
        self._session = session
        self._producer = kafka_producer
        self._topic = kafka_topic
        self._interval = poll_interval_sec
        self._health_file = Path(health_file)
        self._offline_fallback_sec = offline_fallback_sec
        self._pradan_offline_since: Optional[float] = None
        self._fallback_active = False
        self._pending: List[SourceStatus] = []

    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _check_pradan(self) -> SourceStatus:
        if self._session.state == SessionState.EXPIRED:
            status_str = "SESSION_EXPIRED"
        else:
            status_str = "ONLINE"

        if status_str in ("SESSION_EXPIRED", "OFFLINE"):
            if self._pradan_offline_since is None:
                self._pradan_offline_since = time.monotonic()
            elapsed = time.monotonic() - self._pradan_offline_since
            if elapsed >= self._offline_fallback_sec:
                self._fallback_active = True
                logger.warning("PRADAN offline %.0fs — activating NOAA fallback", elapsed)
        else:
            self._pradan_offline_since = None
            self._fallback_active = False

        return SourceStatus(
            source="pradan",
            status=status_str,
            latency_ms=0.0,
            last_success_utc=self._now_utc(),
            fallback_active=self._fallback_active,
            pipeline_version=_PIPELINE_VERSION,
        )

    def _publish(self, status: SourceStatus):
        payload = dataclasses.asdict(status)
        if self._producer is not None:
            try:
                self._producer.send(self._topic, value=payload)
                for pending in self._pending:
                    self._producer.send(self._topic, value=dataclasses.asdict(pending))
                self._pending.clear()
                return
            except Exception as exc:
                logger.warning("Kafka publish failed: %s — buffering to %s", exc, self._health_file)

        self._pending.append(status)
        self._health_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._health_file.with_suffix(".tmp")
        records = [dataclasses.asdict(s) for s in self._pending]
        tmp.write_text(json.dumps(records, indent=2))
        tmp.replace(self._health_file)

    async def _poll(self):
        self._publish(self._check_pradan())
        for name, client in ALL_SUPPLEMENTARY_CLIENTS.items():
            status_str, latency_ms = client.ping()
            self._publish(SourceStatus(
                source=name,
                status=status_str,
                latency_ms=latency_ms,
                last_success_utc=self._now_utc(),
                fallback_active=False,
                pipeline_version=_PIPELINE_VERSION,
            ))

    async def run(self):
        while True:
            await self._poll()
            await asyncio.sleep(self._interval)
