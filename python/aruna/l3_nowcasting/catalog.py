"""
CatalogService — SQLite Master Flare Catalog.
Upserts NowcastEvent records idempotently by event_id.
Performance: <100ms per write.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Optional

from aruna.models.nowcast import NowcastEvent

logger = logging.getLogger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS master_flare_catalog (
    event_id           TEXT PRIMARY KEY,
    start_time         TEXT,
    hope_time          TEXT,
    onset_time         TEXT,
    peak_time          TEXT,
    end_time           TEXT,
    goes_class         TEXT NOT NULL,
    goes_subclass      REAL,
    peak_flux_solexs   REAL DEFAULT 0,
    peak_flux_hel1os   REAL DEFAULT 0,
    peak_T_MK          REAL DEFAULT 0,
    peak_EM_cm3        REAL DEFAULT 0,
    peak_DNI           REAL DEFAULT 0,
    hope_lead_time_sec INTEGER,
    hope_fired         INTEGER DEFAULT 0,
    instruments        TEXT,
    detection_method   TEXT,
    confidence         REAL,
    flare_type         TEXT,
    pipeline_version   TEXT,
    provenance_hash    TEXT,
    notes              TEXT
);
"""

_UPSERT = """
INSERT INTO master_flare_catalog
    (event_id, start_time, hope_time, onset_time, peak_time, end_time,
     goes_class, goes_subclass, peak_flux_solexs, peak_flux_hel1os,
     peak_T_MK, peak_EM_cm3, peak_DNI,
     hope_lead_time_sec, hope_fired, instruments,
     detection_method, confidence, flare_type, pipeline_version, provenance_hash)
VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(event_id) DO UPDATE SET
    hope_time          = COALESCE(master_flare_catalog.hope_time,  excluded.hope_time),
    onset_time         = COALESCE(master_flare_catalog.onset_time, excluded.onset_time),
    peak_time          = COALESCE(master_flare_catalog.peak_time,  excluded.peak_time),
    end_time           = COALESCE(master_flare_catalog.end_time,   excluded.end_time),
    peak_flux_solexs   = MAX(master_flare_catalog.peak_flux_solexs, excluded.peak_flux_solexs),
    peak_flux_hel1os   = MAX(master_flare_catalog.peak_flux_hel1os, excluded.peak_flux_hel1os),
    peak_T_MK          = MAX(master_flare_catalog.peak_T_MK,  excluded.peak_T_MK),
    peak_EM_cm3        = MAX(master_flare_catalog.peak_EM_cm3, excluded.peak_EM_cm3),
    peak_DNI           = MAX(master_flare_catalog.peak_DNI,    excluded.peak_DNI),
    confidence         = excluded.confidence
"""


class CatalogService:
    """SQLite Master Flare Catalog service."""

    def __init__(self, db_url: str = "sqlite:///aruna_catalog.db") -> None:
        self._is_postgres = db_url.startswith("postgresql://") or db_url.startswith("postgres://")
        self._conn: Optional[sqlite3.Connection] = None
        if not self._is_postgres:
            path = db_url.replace("sqlite:///", "").replace("sqlite://", "") or ":memory:"
            self._sqlite_path = path
            self._init()

    def _init(self) -> None:
        self._conn = sqlite3.connect(self._sqlite_path, check_same_thread=False)
        self._conn.execute(_DDL)
        self._conn.commit()

    def write(
        self,
        event: NowcastEvent,
        peak_T_MK: float = 0.0,
        peak_EM_cm3: float = 0.0,
        peak_flux_solexs: float = 0.0,
        peak_flux_hel1os: float = 0.0,
        peak_DNI: float = 0.0,
    ) -> None:
        """Write event to catalog. Exceptions are logged, not raised."""
        try:
            if self._is_postgres:
                logger.warning("PostgreSQL catalog not yet wired — event logged only")
                return
            ts = event.timestamp.isoformat()
            hope_time = ts if event.alert_type == "HOPE_PRECURSOR" else None
            onset_time = ts if event.alert_type == "FLARE_ONSET" else None
            peak_time = ts if event.alert_type == "FLARE_PEAK" else None
            end_time = ts if event.alert_type == "FLARE_END" else None
            start_time = ts if event.alert_type in ("HOPE_PRECURSOR", "FLARE_ONSET") else None
            self._conn.execute(
                _UPSERT,
                (
                    event.event_id,
                    start_time,
                    hope_time,
                    onset_time,
                    peak_time,
                    end_time,
                    event.goes_class,
                    event.goes_subclass,
                    peak_flux_solexs,
                    peak_flux_hel1os,
                    peak_T_MK,
                    peak_EM_cm3,
                    peak_DNI,
                    event.lead_time_sec,
                    1 if event.alert_type == "HOPE_PRECURSOR" else 0,
                    json.dumps(event.instruments),
                    event.detection_method,
                    event.confidence,
                    event.flare_type,
                    event.pipeline_version,
                    event.provenance_hash,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.error("Catalog write failed for %s: %s", event.event_id, exc)

    def fetch_event(self, event_id: str) -> Optional[dict]:
        """Fetch event by ID. Returns dict or None. Test use only."""
        if self._is_postgres or self._conn is None:
            return None
        cur = self._conn.execute(
            "SELECT * FROM master_flare_catalog WHERE event_id=?", (event_id,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))

    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
