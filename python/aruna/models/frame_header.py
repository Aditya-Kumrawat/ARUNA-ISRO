from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime

_VALID_SOURCES = frozenset({"LIVE", "REPLAY", "SIMULATION"})


@dataclass
class FrameHeader:
    timestamp: datetime
    source: str                                    # LIVE | REPLAY | SIMULATION
    frame_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "2.0.0"
    quality: str = "GOOD"                         # GOOD | DEGRADED | STALE
    latency_ms: int = 0

    def __post_init__(self) -> None:
        if self.source not in _VALID_SOURCES:
            raise ValueError(f"source must be one of {_VALID_SOURCES}, got {self.source!r}")
