"""
Global data contract: L3 output.
NowcastEvent is a confirmed or precursor flare detection from the
dual-trigger consensus engine. Also written to the Master Flare Catalog.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

ALERT_TYPES = frozenset({
    "HOPE_PRECURSOR",
    "FLARE_ONSET",
    "FLARE_PEAK",
    "FLARE_END",
})

GOES_CLASSES = frozenset({"A", "B", "C", "M", "X", "SUB_A"})

DETECTION_METHODS = frozenset({
    "DUAL_TRIGGER",
    "SOLEXS_ONLY",       # degraded — HEL1OS offline
    "HEL1OS_ONLY",       # degraded — SoLEXS offline (B2 fix)
    "HOPE",              # precursor only, awaiting confirmation
    "SEP_EVENT",         # particle event, not a flare
})

FLARE_TYPES = frozenset({
    "STANDARD",
    "OVERLAPPING",
    "LIMB_OCCULTED",
    "SUB_A",
})


@dataclass
class NowcastEvent:
    """
    Flare detection event from L3.
    Published to: aruna.nowcast.events
    Also written to: master_flare_catalog (SQLite / PostgreSQL)
    """

    alert_type: str                    # one of ALERT_TYPES
    timestamp: datetime                # UTC — when the alert was issued
    goes_class: str                    # one of GOES_CLASSES
    goes_subclass: float               # e.g. 3.8 for M3.8
    confidence: float                  # 0.0–1.0
    instruments: List[str]             # which instruments triggered
    physics_frame_id: str              # provenance link to L2 PhysicsFrame
    detection_method: str              # one of DETECTION_METHODS
    flare_type: str                    # one of FLARE_TYPES

    # ── Optional — populated when available ──────────────────────────────────
    lead_time_sec: Optional[int] = None
    hope_reason: Optional[str] = None
    goes_catalog_match: Optional[str] = None  # matched GOES event ID
    noaa_ar_number: Optional[int] = None

    # ── Auto-generated ───────────────────────────────────────────────────────
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    provenance_hash: str = ""

    # ── Message bus metadata ─────────────────────────────────────────────────
    schema_version: str = "1.0"
    pipeline_version: str = "2.0.0"
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        if self.alert_type not in ALERT_TYPES:
            raise ValueError(f"alert_type must be one of {ALERT_TYPES}")
        if self.goes_class not in GOES_CLASSES:
            raise ValueError(f"goes_class must be one of {GOES_CLASSES}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.detection_method not in DETECTION_METHODS:
            raise ValueError(f"detection_method must be one of {DETECTION_METHODS}")

    @property
    def goes_label(self) -> str:
        return f"{self.goes_class}{self.goes_subclass:.1f}"

    def is_major(self) -> bool:
        """True for M5+ and any X-class event."""
        if self.goes_class == "X":
            return True
        if self.goes_class == "M" and self.goes_subclass >= 5.0:
            return True
        return False

    def is_confirmed(self) -> bool:
        return self.detection_method == "DUAL_TRIGGER"
