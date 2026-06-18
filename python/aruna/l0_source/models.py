from dataclasses import dataclass

VALID_STATUSES = frozenset({"ONLINE", "DEGRADED", "OFFLINE", "SESSION_EXPIRED"})
VALID_SOURCES = frozenset({
    "pradan", "noaa_swpc", "goes16_xrs", "hek",
    "jsoc_sharp", "cdaweb_ace", "cdaw_lasco", "gfz_kp",
})


@dataclass
class SourceStatus:
    source: str
    status: str
    latency_ms: float
    last_success_utc: str
    fallback_active: bool
    pipeline_version: str
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {self.status!r}. Must be one of {VALID_STATUSES}")
        if self.source not in VALID_SOURCES:
            raise ValueError(f"Invalid source: {self.source!r}. Must be one of {VALID_SOURCES}")
