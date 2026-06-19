"""
XL-1 Magnetogram Downloader — fetches SHARP parameters from JSOC.

Two access paths (tried in order):
  1. drms library (preferred): pip install drms  — clean, typed, native Pandas
  2. JSOC JSON HTTP API (fallback): no extra dependencies, works offline-ish

Both paths return a list[dict] of raw SHARP keyword rows.
Failure in both paths returns [] — caller must handle empty gracefully.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests

from .models import SHARPConfig

logger = logging.getLogger(__name__)

# ── Try to import drms (optional dependency) ─────────────────────────────────
try:
    import drms as _drms
    _DRMS_AVAILABLE = True
except ImportError:
    _DRMS_AVAILABLE = False
    logger.debug("drms not installed — using JSOC HTTP fallback for SHARP downloads")


def _tai_to_dt(tai_str: str) -> datetime:
    """Convert JSOC TAI string '2024.02.22_12:00:00_TAI' → UTC datetime."""
    s = tai_str.replace("_TAI", "").replace("_", "T").replace(".", "-", 2)
    try:
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


class JSOCDownloader:
    """
    Fetches SHARP CEA 720s parameters for a given HARP or date range.

    Usage
    -----
    dl = JSOCDownloader(config)
    rows = dl.fetch_latest(harpnum=4977)      # most recent 6h
    rows = dl.fetch_range(harpnum=4977,
                          start=datetime(...), end=datetime(...))
    """

    def __init__(self, config: Optional[SHARPConfig] = None) -> None:
        self._cfg = config or SHARPConfig()

    # ── Public API ─────────────────────────────────────────────────────────

    def fetch_latest(self, harpnum: int) -> List[Dict]:
        """Fetch the most recent cadence_minutes × lookback_hours/12 records."""
        end   = datetime.now(timezone.utc)
        start = end - timedelta(hours=self._cfg.lookback_hours)
        return self.fetch_range(harpnum, start, end)

    def fetch_range(
        self,
        harpnum: int,
        start: datetime,
        end: datetime,
    ) -> List[Dict]:
        """Fetch SHARP rows for harpnum in [start, end]. Returns [] on any error."""
        if _DRMS_AVAILABLE:
            rows = self._fetch_drms(harpnum, start, end)
            if rows is not None:
                return rows
            logger.warning("drms fetch failed — retrying with HTTP fallback")
        return self._fetch_http(harpnum, start, end)

    def list_active_harps(self, timestamp: Optional[datetime] = None) -> List[int]:
        """
        Return HARP numbers of all active regions visible at timestamp.
        Falls back to empty list on failure — caller must handle gracefully.
        """
        ts = timestamp or datetime.now(timezone.utc)
        ts_str = ts.strftime("%Y.%m.%d_%H:%M:%S_TAI")
        query = f"{self._cfg.jsoc_series}[][{ts_str}]"
        params = {
            "ds": query,
            "op": "rs_list",
            "key": "HARPNUM",
            "n": "50",
        }
        try:
            resp = requests.get(
                self._cfg.jsoc_base_url,
                params=params,
                timeout=15,
            )
            data = resp.json()
            rows = data.get("keywords", {}).get("HARPNUM", {}).get("values", [])
            return [int(r) for r in rows if r and str(r).strip().lstrip("-").isdigit() and int(r) > 0]
        except Exception as exc:
            logger.warning("list_active_harps failed: %s", exc)
            return []

    # ── Internal: drms path ─────────────────────────────────────────────────

    def _fetch_drms(
        self, harpnum: int, start: datetime, end: datetime
    ) -> Optional[List[Dict]]:
        try:
            c = _drms.Client(verbose=False)
            t0 = start.strftime("%Y.%m.%d_%H:%M:%S_TAI")
            t1 = end.strftime("%Y.%m.%d_%H:%M:%S_TAI")
            query = (
                f"{self._cfg.jsoc_series}[{harpnum}]"
                f"[{t0}/{(end - start).seconds // 60}m@{self._cfg.cadence_minutes}m]"
            )
            keys_str = ",".join(self._cfg.sharp_keywords) + ",T_REC,HARPNUM"
            df = c.query(query, key=keys_str, n=500)
            if df is None or df.empty:
                return []
            rows = df.to_dict(orient="records")
            # Parse T_REC strings to datetime
            for r in rows:
                r["_parsed_time"] = _tai_to_dt(str(r.get("T_REC", "")))
            return rows
        except Exception as exc:
            logger.debug("drms fetch error for HARP %d: %s", harpnum, exc)
            return None

    # ── Internal: HTTP fallback path ────────────────────────────────────────

    def _fetch_http(
        self, harpnum: int, start: datetime, end: datetime
    ) -> List[Dict]:
        """
        Direct JSOC JSON API query. Returns parsed rows or [] on failure.

        URL format:
          jsoc_info?ds=hmi.sharp_cea_720s[HARP][T_START/DURm@12m]
                  &op=rs_list&key=USFLUX,...&json=1
        """
        duration_min = int((end - start).total_seconds() / 60) + self._cfg.cadence_minutes
        t0_str = start.strftime("%Y.%m.%d_%H:%M:%S_TAI")
        ds = (
            f"{self._cfg.jsoc_series}[{harpnum}]"
            f"[{t0_str}/{duration_min}m@{self._cfg.cadence_minutes}m]"
        )
        keys_str = ",".join(self._cfg.sharp_keywords) + ",T_REC,HARPNUM"
        params = {"ds": ds, "op": "rs_list", "key": keys_str, "json": "1", "n": "500"}
        try:
            resp = requests.get(self._cfg.jsoc_base_url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("JSOC HTTP fetch failed for HARP %d: %s", harpnum, exc)
            return []

        return self._parse_jsoc_response(data)

    def _parse_jsoc_response(self, data: dict) -> List[Dict]:
        """Convert JSOC rs_list JSON format into a flat list of row dicts."""
        try:
            keywords = data.get("keywords", [])
            kw_names = [kw["name"] for kw in keywords]
            rows_raw = data.get("rows", [])
            parsed = []
            for row in rows_raw:
                record = dict(zip(kw_names, row))
                record["_parsed_time"] = _tai_to_dt(str(record.get("T_REC", "")))
                parsed.append(record)
            return parsed
        except Exception as exc:
            logger.warning("JSOC response parse error: %s", exc)
            return []
