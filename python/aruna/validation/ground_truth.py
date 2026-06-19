"""
Ground truth matching for ARUNA validation — PRD §20.3.

Matching rule:
  True Positive if onset time within ±3 minutes of GOES catalog
  AND predicted GOES class within ±1 subclass of actual.

Ground truth hierarchy (§20.3):
  1. SoLEXS team catalog (Sarwade et al.) — primary for post-2023-09
  2. NOAA GOES XRS via SunPy HEK — globally accepted standard
  3. Fermi/GBM — independent confirmation for M2+
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from aruna.models.nowcast import NowcastEvent
from aruna.validation.models import MatchedEvent

logger = logging.getLogger(__name__)

# PRD §20.3 constants
ONSET_WINDOW_MIN   = 3      # ±3 minutes
CLASS_TOLERANCE    = 1      # ±1 GOES subclass (integer)

# Temporal split definitions (PRD §8.9)
TRAIN_PERIOD    = ("2023-09-02", "2024-06-30")
VALIDATE_PERIOD = ("2024-07-01", "2024-12-31")
TEST_PERIOD     = ("2025-01-01", "2025-06-30")

_GOES_CLASS_ORDER = ["A", "B", "C", "M", "X"]


def _parse_goes_class(goes_cls_str: str) -> tuple[str, float]:
    """Parse '  M5.4' → ('M', 5.4). Returns ('QUIET', 0.0) on failure."""
    s = goes_cls_str.strip().upper()
    for letter in _GOES_CLASS_ORDER:
        if s.startswith(letter):
            try:
                subclass = float(s[1:]) if s[1:] else 1.0
                return letter, subclass
            except ValueError:
                pass
    return "QUIET", 0.0


def _class_within_tolerance(
    pred_class: str, pred_subclass: float,
    actual_class: str, actual_subclass: float,
    tolerance: int = CLASS_TOLERANCE,
) -> bool:
    """True if letters match AND |subclass_int_diff| ≤ tolerance."""
    if pred_class != actual_class:
        return False
    return abs(int(pred_subclass) - int(actual_subclass)) <= tolerance


def fetch_goes_catalog(
    start: datetime,
    end: datetime,
    _mock_catalog: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Fetch GOES flare catalog from SunPy HEK.
    Returns list of dicts with keys: 'start_time', 'goes_class', 'goes_subclass'.

    _mock_catalog: inject synthetic catalog for tests (skips network call).
    """
    if _mock_catalog is not None:
        return _mock_catalog

    try:
        from sunpy.net import Fido, attrs as a
        results = Fido.search(
            a.Time(start.isoformat(), end.isoformat()),
            a.hek.EventType("FL"),
            a.hek.OBS.Instrument("GOES"),
        )
        catalog = []
        for row in results["hek"]:
            letter, subclass = _parse_goes_class(str(row.get("fl_goescls", "")))
            if letter == "QUIET":
                continue
            t = row.get("event_starttime")
            if hasattr(t, "datetime"):
                t = t.datetime
            catalog.append({
                "start_time": t,
                "goes_class": letter,
                "goes_subclass": subclass,
            })
        return catalog
    except Exception as exc:
        logger.warning("SunPy HEK query failed: %s — returning empty catalog", exc)
        return []


def match_event(
    nowcast: NowcastEvent,
    catalog: List[dict],
) -> MatchedEvent:
    """
    Match one NowcastEvent against the GOES catalog.
    Returns a MatchedEvent with matched=True if a catalog entry satisfies
    the ±3 min onset + ±1 subclass rule.
    """
    onset = nowcast.timestamp
    if onset.tzinfo is None:
        onset = onset.replace(tzinfo=timezone.utc)

    window = timedelta(minutes=ONSET_WINDOW_MIN)
    best: Optional[dict] = None
    best_delta: float = float("inf")

    for entry in catalog:
        t_cat = entry["start_time"]
        if isinstance(t_cat, str):
            t_cat = datetime.fromisoformat(t_cat)
        if t_cat.tzinfo is None:
            t_cat = t_cat.replace(tzinfo=timezone.utc)

        delta = abs((onset - t_cat).total_seconds())
        if delta <= window.total_seconds():
            if _class_within_tolerance(
                nowcast.goes_class, nowcast.goes_subclass,
                entry["goes_class"], entry["goes_subclass"],
            ):
                if delta < best_delta:
                    best = entry
                    best_delta = delta

    matched = best is not None
    return MatchedEvent(
        predicted_class=nowcast.goes_class,
        actual_class=best["goes_class"] if matched else "QUIET",
        predicted_prob_M_plus=nowcast.confidence if nowcast.goes_class in {"M", "X"} else 0.0,
        confidence=nowcast.confidence,
        onset_time=onset,
        delta_sec=best_delta if matched else 0.0,
        matched=matched,
    )


def match_all(
    events: List[NowcastEvent],
    catalog: List[dict],
) -> List[MatchedEvent]:
    """Batch-match a list of NowcastEvents against the catalog."""
    return [match_event(e, catalog) for e in events]


def assert_no_temporal_leakage(events: List[NowcastEvent], split: str) -> None:
    """
    Raises ValueError if any event's timestamp falls outside the declared split.
    split: "train" | "validate" | "test"
    """
    periods = {
        "train":    TRAIN_PERIOD,
        "validate": VALIDATE_PERIOD,
        "test":     TEST_PERIOD,
    }
    if split not in periods:
        raise ValueError(f"Unknown split '{split}'. Choose: train, validate, test")

    start_str, end_str = periods[split]
    t_start = datetime.fromisoformat(start_str).replace(tzinfo=timezone.utc)
    t_end   = datetime.fromisoformat(end_str).replace(tzinfo=timezone.utc)

    for ev in events:
        ts = ev.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if not (t_start <= ts <= t_end):
            raise ValueError(
                f"Temporal leakage: event {ev.event_id} at {ts.isoformat()} "
                f"is outside {split} period [{start_str}, {end_str}]"
            )
