"""
XL-1 Magnetogram Parser — converts raw JSOC rows to typed SHARPRecord objects.

Handles missing / DRMS_MISSING_VALUE sentinels (typically -2147483648 or NaN).
Any field that cannot be parsed is set to 0.0 — the caller should treat
zero-valued SHARP records as low-confidence, not as an error.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .models import SHARPRecord

logger = logging.getLogger(__name__)

# JSOC sentinel for missing data
_JSOC_MISSING = {-2147483648, -2147483647, -32768, -32767}
_JSOC_MISSING_FLOAT = {-9999.0, -99999.0}


def _safe_float(value, default: float = 0.0) -> float:
    """Parse a value from JSOC dict, returning default on missing/sentinel."""
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:              # NaN
            return default
        if int(f) in _JSOC_MISSING:
            return default
        if f in _JSOC_MISSING_FLOAT:
            return default
        return f
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        i = int(float(value))
        if i in _JSOC_MISSING:
            return default
        return i
    except (TypeError, ValueError):
        return default


def parse_row(row: Dict) -> Optional[SHARPRecord]:
    """
    Parse one JSOC row dict into a SHARPRecord.

    Returns None if the row cannot be meaningfully parsed (e.g., all-zero
    or missing timestamp). Callers filter these out.
    """
    t = row.get("_parsed_time")
    if t is None:
        return None
    if not isinstance(t, datetime):
        return None

    harpnum = _safe_int(row.get("HARPNUM", 0))
    if harpnum <= 0:
        return None

    record = SHARPRecord(
        harpnum=harpnum,
        noaa_ar=_safe_int(row.get("NOAA_AR", 0)) or None,
        t_rec=t,
        usflux=_safe_float(row.get("USFLUX")),
        totusjh=_safe_float(row.get("TOTUSJH")),
        totusjz=_safe_float(row.get("TOTUSJZ")),
        totbsq=_safe_float(row.get("TOTBSQ")),
        totpot=_safe_float(row.get("TOTPOT")),
        savncpp=_safe_float(row.get("SAVNCPP")),
        absnjzh=_safe_float(row.get("ABSNJZH")),
        r_value=_safe_float(row.get("R_VALUE")),
        meanshr=_safe_float(row.get("MEANSHR")),
        meangam=_safe_float(row.get("MEANGAM")),
        lat_fwt=_safe_float(row.get("LAT_FWT")),
        lon_fwt=_safe_float(row.get("LON_FWT")),
        area_acr=_safe_float(row.get("AREA_ACR")),
    )
    return record


def parse_rows(rows: List[Dict]) -> List[SHARPRecord]:
    """Parse a list of raw JSOC rows, silently dropping unparseable ones."""
    records = []
    for row in rows:
        try:
            r = parse_row(row)
            if r is not None:
                records.append(r)
        except Exception as exc:
            logger.debug("Skipping malformed SHARP row: %s", exc)
    records.sort(key=lambda r: r.t_rec)
    return records


def latest_record(records: List[SHARPRecord]) -> Optional[SHARPRecord]:
    """Return the most recent record from a sorted list, or None."""
    return records[-1] if records else None
