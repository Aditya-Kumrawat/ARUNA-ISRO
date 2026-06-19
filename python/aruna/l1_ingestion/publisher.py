from __future__ import annotations

import dataclasses
import logging
from datetime import datetime
from typing import List

import numpy as np

from ..models.observation import ObservationFrame

logger = logging.getLogger(__name__)

_PIPELINE_VERSION = "2.0.0"
_SCHEMA_VERSION = "1.0"


def _to_json(frame: ObservationFrame) -> dict:
    d = dataclasses.asdict(frame)
    for k, v in list(d.items()):
        if isinstance(v, np.ndarray):
            d[k] = v.tolist()
        elif isinstance(v, datetime):
            d[k] = v.isoformat()
        elif isinstance(v, (np.integer,)):
            d[k] = int(v)
        elif isinstance(v, (np.floating,)):
            d[k] = float(v)
    d['schema_version'] = _SCHEMA_VERSION
    d['pipeline_version'] = _PIPELINE_VERSION
    return d


class KafkaPublisher:
    def __init__(
        self,
        producer,
        topic_solexs: str = "aruna.observations.solexs",
        topic_hel1os: str = "aruna.observations.hel1os",
        topic_suit:   str = "aruna.observations.suit",
        topic_velc:   str = "aruna.observations.velc",
        topic_quality: str = "aruna.quality.summary",
    ):
        self._producer = producer
        self._t_solexs  = topic_solexs
        self._t_hel1os  = topic_hel1os
        self._t_suit    = topic_suit
        self._t_velc    = topic_velc
        self._t_quality = topic_quality

    def _topic_for(self, frame: ObservationFrame) -> str:
        if frame.is_solexs():
            return self._t_solexs
        if frame.is_hel1os():
            return self._t_hel1os
        if frame.is_suit():
            return self._t_suit
        if frame.is_velc():
            return self._t_velc
        return self._t_solexs  # fallback — should not happen

    def publish(self, frames: List[ObservationFrame]) -> None:
        counts = {'total': 0, 'good': 0, 'flagged': 0}
        for frame in frames:
            topic = self._topic_for(frame)
            payload = _to_json(frame)
            try:
                self._producer.send(topic, value=payload, key=frame.instrument.encode())
                counts['total'] += 1
                if frame.is_good():
                    counts['good'] += 1
                else:
                    counts['flagged'] += 1
            except Exception as exc:
                logger.error("Kafka publish failed for %s: %s", frame.frame_id, exc)

        summary = {
            'schema_version': _SCHEMA_VERSION,
            'pipeline_version': _PIPELINE_VERSION,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            **counts,
        }
        try:
            self._producer.send(self._t_quality, value=summary)
        except Exception as exc:
            logger.error("Quality summary publish failed: %s", exc)
