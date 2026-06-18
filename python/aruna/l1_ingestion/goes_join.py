from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import numpy as np

from ..models.observation import ObservationFrame

logger = logging.getLogger(__name__)


def _to_utc(t) -> datetime:
    """Parse t to a *naive* UTC datetime (no tzinfo) for comparison."""
    if isinstance(t, datetime):
        if t.tzinfo is not None:
            return t.replace(tzinfo=None)
        return t
    s = str(t).replace('Z', '+00:00')
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt


def _np_to_utc(t: np.datetime64) -> datetime:
    ts_sec = float((t - np.datetime64('1970-01-01T00:00:00', 'ns')) / np.timedelta64(1, 's'))
    return datetime.fromtimestamp(ts_sec, tz=timezone.utc)


class GoesLabelJoiner:
    def __init__(self, match_window_min: float = 3.0, query_window_min: float = 10.0):
        self._match_delta = timedelta(minutes=match_window_min)
        self._query_min = query_window_min

    def join(
        self,
        frames: List[ObservationFrame],
        hek_results=None,
    ) -> List[ObservationFrame]:
        """
        Annotate ObservationFrames with GOES class from HEK results.
        hek_results: list of dicts with 'event_starttime' and 'fl_goescls',
                     or a sunpy HEK results table, or None (offline).
        """
        if not frames:
            return frames

        events = []
        if hek_results is not None:
            for row in hek_results:
                try:
                    onset = _to_utc(row['event_starttime'])
                    cls_str = str(row.get('fl_goescls', '') or '').strip()
                    if not cls_str:
                        continue
                    letter = cls_str[0].upper()
                    subclass = float(cls_str[1:]) if len(cls_str) > 1 else 0.0
                    events.append((onset, letter, subclass))
                except (KeyError, ValueError, TypeError):
                    continue

        for frame in frames:
            ts = _to_utc(frame.timestamp) if not isinstance(frame.timestamp, np.datetime64) \
                else _np_to_utc(frame.timestamp)
            matched = False
            for onset, letter, subclass in events:
                if abs((ts - onset).total_seconds()) <= self._match_delta.total_seconds():
                    frame.goes_class = letter
                    frame.goes_subclass = subclass
                    matched = True
                    break
            if not matched:
                frame.goes_class = frame.goes_class or 'QUIET_SUN'

        return frames

    def query_hek(self, start: datetime, end: datetime):
        """Query SunPy HEK. Returns results or None if offline."""
        try:
            from sunpy.net import Hek, attrs as a
            hek = Hek()
            return hek.search(
                a.Time(start - timedelta(minutes=self._query_min),
                       end + timedelta(minutes=self._query_min)),
                a.hek.EventType("FL"),
            )
        except Exception as exc:
            logger.warning("HEK query failed (offline): %s", exc)
            return None
