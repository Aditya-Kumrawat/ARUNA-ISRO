#!/usr/bin/env python3
"""
download_goes_labels.py — Download the NOAA GOES flare event catalog for the
full Aditya-L1 science operations period and save to data/goes_events.parquet.

This is the LABEL SOURCE for TCN training. SoLEXS/HEL1OS are the inputs;
GOES XRS event classification (B/C/M/X) is the ground truth.

Usage:
    python python/scripts/download_goes_labels.py \
        --start 2023-09-01 \
        --end   2026-06-20 \
        --out   data/goes_events.parquet

Outputs:
    data/goes_events.parquet  —  columns:
        start_time  (datetime64[ns, UTC])
        peak_time   (datetime64[ns, UTC])
        end_time    (datetime64[ns, UTC])
        goes_class  (str, e.g. "X6.3")
        goes_class_letter  (str, "B"/"C"/"M"/"X")
        goes_class_number  (float, e.g. 6.3)
        noaa_ar     (str, active region number)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ── GOES class parsing ─────────────────────────────────────────────────────────

_CLASS_ORDER = {"A": 0, "B": 1, "C": 2, "M": 3, "X": 4}


def parse_goes_class(raw: str) -> tuple[str, float]:
    """
    Parse a GOES class string like 'X6.3' or 'M2.8' into (letter, number).
    Returns ('', 0.0) for unrecognised / missing values.
    """
    raw = str(raw).strip().upper()
    if not raw or raw in ("NONE", "NAN", "N/A", ""):
        return "", 0.0
    letter = raw[0]
    if letter not in _CLASS_ORDER:
        return "", 0.0
    try:
        number = float(raw[1:]) if len(raw) > 1 else 1.0
    except ValueError:
        number = 1.0
    return letter, number


def class_to_label(letter: str) -> int:
    """Map GOES letter to 0-4 integer label (QUIET=0, B=1, C=2, M=3, X=4)."""
    return _CLASS_ORDER.get(letter, 0)


# ── Primary download via SunPy HEK ────────────────────────────────────────────

def _download_via_hek(start: str, end: str) -> pd.DataFrame:
    """Query LMSAL HEK for GOES-classified flare events."""
    try:
        from sunpy.net import Fido, attrs as a
    except ImportError:
        raise ImportError("sunpy not installed. Run: pip install sunpy")

    print(f"Querying HEK for FL events {start} → {end} …")
    result = Fido.search(
        a.Time(start, end),
        a.hek.EventType("FL"),
    )
    if not result:
        return pd.DataFrame()

    table = result["hek"]
    rows = []
    for row in table:
        letter, number = parse_goes_class(str(row.get("fl_goescls", "")))
        if not letter:
            continue  # skip events without a GOES class
        rows.append({
            "start_time": pd.Timestamp(str(row["event_starttime"]), tz="UTC"),
            "peak_time":  pd.Timestamp(str(row["event_peaktime"]),  tz="UTC"),
            "end_time":   pd.Timestamp(str(row["event_endtime"]),   tz="UTC"),
            "goes_class": str(row["fl_goescls"]).upper(),
            "goes_class_letter": letter,
            "goes_class_number": number,
            "noaa_ar": str(row.get("ar_noaanum", "")),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("peak_time").reset_index(drop=True)
    return df


# ── Fallback 1: NASA DONKI REST API ───────────────────────────────────────────

def _download_via_donki(start: str, end: str) -> pd.DataFrame:
    """
    Download GOES flare catalog from NASA CCMC DONKI REST API.
    Chunked into 6-month windows (API limit). Always publicly accessible.
    https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR
    """
    import urllib.request
    import json

    print("HEK unavailable — trying NASA DONKI API …")

    start_dt = datetime.strptime(start[:10], "%Y-%m-%d")
    end_dt   = datetime.strptime(end[:10],   "%Y-%m-%d")

    all_rows: list[dict] = []
    chunk_start = start_dt

    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=180), end_dt)
        url = (
            "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR"
            f"?startDate={chunk_start.strftime('%Y-%m-%d')}"
            f"&endDate={chunk_end.strftime('%Y-%m-%d')}"
        )
        try:
            print(f"  Fetching {chunk_start.date()} → {chunk_end.date()} …")
            req = urllib.request.Request(url, headers={"User-Agent": "ARUNA-Pipeline/2.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                events = json.loads(resp.read().decode())
            if not events:
                chunk_start = chunk_end + timedelta(days=1)
                continue
            for ev in events:
                cls_raw = (ev.get("classType") or "").strip()
                letter, number = parse_goes_class(cls_raw)
                if not letter:
                    continue
                try:
                    def _ts(s: str) -> pd.Timestamp:
                        return pd.Timestamp(s.replace("Z", "+00:00"))
                    peak_ts  = _ts(ev["peakTime"])
                    start_ts = _ts(ev.get("beginTime") or ev["peakTime"])
                    end_ts   = _ts(ev.get("endTime")   or ev["peakTime"])
                    all_rows.append({
                        "start_time":        start_ts,
                        "peak_time":         peak_ts,
                        "end_time":          end_ts,
                        "goes_class":        cls_raw.upper(),
                        "goes_class_letter": letter,
                        "goes_class_number": number,
                        "noaa_ar":           str(ev.get("activeRegionNum") or ""),
                    })
                except Exception:
                    continue
        except Exception as exc:
            print(f"  DONKI fetch failed ({chunk_start.date()}–{chunk_end.date()}): {exc}")

        chunk_start = chunk_end + timedelta(days=1)

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df = df.sort_values("peak_time").reset_index(drop=True)
    return df


# ── Fallback 2: NOAA SWPC event text files ────────────────────────────────────

def _download_via_swpc(start: str, end: str) -> pd.DataFrame:
    """
    Download GOES flare list from NOAA SWPC monthly event files.
    URL pattern: https://www.swpc.noaa.gov/pub/indices/events/YYYYMM_events.txt
    """
    import urllib.request
    from calendar import monthrange

    print("DONKI unavailable — trying NOAA SWPC monthly event files …")

    start_dt = datetime.strptime(start[:10], "%Y-%m-%d")
    end_dt   = datetime.strptime(end[:10],   "%Y-%m-%d")

    all_rows: list[dict] = []
    year, month = start_dt.year, start_dt.month

    while (year, month) <= (end_dt.year, end_dt.month):
        ym = f"{year}{month:02d}"
        url = f"https://www.swpc.noaa.gov/pub/indices/events/{ym}_events.txt"
        try:
            print(f"  Fetching {ym} …")
            req = urllib.request.Request(url, headers={"User-Agent": "ARUNA-Pipeline/2.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                text = resp.read().decode("latin-1")
            for line in text.splitlines():
                line = line.strip()
                # SWPC event line format (fixed-width):
                # date(YYMM)  start  max   end  obs  Q  type  loc  freq  class  ...
                # Flare lines have "XRA" or "FLA" in the type column
                if not line or line.startswith("#") or len(line) < 50:
                    continue
                if "XRA" not in line and "FLA" not in line:
                    continue
                try:
                    parts = line.split()
                    # Parts[0] = YYMM, parts[1] = start HHMM, parts[2] = max HHMM, parts[3] = end HHMM
                    yymm   = parts[0]
                    yr2    = int(yymm[:2])
                    yr4    = 2000 + yr2 if yr2 < 50 else 1900 + yr2
                    mo_str = int(yymm[2:4])
                    # Day is the line's day field — SWPC includes it as the numeric date prefix
                    # Actually SWPC format has date as YYYYMMDD in some versions; try parsing
                    if len(parts[0]) == 8:  # YYYYMMDD
                        yr4  = int(parts[0][:4])
                        mo_str = int(parts[0][4:6])
                        dy   = int(parts[0][6:8])
                    else:
                        # Fallback: use the month being fetched
                        dy = 1
                    hh_s, mm_s = int(parts[1][:2]), int(parts[1][2:4])
                    hh_p, mm_p = int(parts[2][:2]), int(parts[2][2:4])
                    hh_e, mm_e = int(parts[3][:2]), int(parts[3][2:4])
                    # Find GOES class in parts — look for letter+number pattern
                    cls_raw = ""
                    for p in parts:
                        l, n = parse_goes_class(p)
                        if l:
                            cls_raw = p
                            break
                    letter, number = parse_goes_class(cls_raw)
                    if not letter:
                        continue
                    all_rows.append({
                        "start_time":        pd.Timestamp(yr4, mo_str, dy, hh_s, mm_s, tz="UTC"),
                        "peak_time":         pd.Timestamp(yr4, mo_str, dy, hh_p, mm_p, tz="UTC"),
                        "end_time":          pd.Timestamp(yr4, mo_str, dy, hh_e, mm_e, tz="UTC"),
                        "goes_class":        cls_raw.upper(),
                        "goes_class_letter": letter,
                        "goes_class_number": number,
                        "noaa_ar":           "",
                    })
                except Exception:
                    continue
        except Exception as exc:
            print(f"  Could not fetch {ym}: {exc}")

        month += 1
        if month > 12:
            month = 1
            year += 1

    df = pd.DataFrame(all_rows)
    if not df.empty:
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end,   tz="UTC")
        df = df[(df["peak_time"] >= s) & (df["peak_time"] <= e)]
        df = df.sort_values("peak_time").reset_index(drop=True)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def download(start: str, end: str, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Try HEK → DONKI → SWPC in order
    df = pd.DataFrame()
    try:
        df = _download_via_hek(start, end)
    except Exception as exc:
        print(f"HEK query failed ({exc}) — trying DONKI fallback …")

    if df.empty:
        try:
            df = _download_via_donki(start, end)
        except Exception as exc:
            print(f"DONKI failed ({exc}) — trying SWPC fallback …")

    if df.empty:
        df = _download_via_swpc(start, end)

    if df.empty:
        print("ERROR: Could not retrieve any GOES events. Check network connectivity.")
        sys.exit(1)

    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df)} events → {out}")

    # Print summary
    print("\nEvent count by class:")
    counts = df["goes_class_letter"].value_counts().sort_index()
    for letter, count in counts.items():
        label = class_to_label(letter)
        print(f"  {letter} (label={label}): {count:>5} events")

    print(f"\nDate range: {df['peak_time'].min()} → {df['peak_time'].max()}")
    print("\nNext step: python python/scripts/batch_l1_l2.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download NOAA GOES flare event catalog")
    parser.add_argument("--start", default="2023-09-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end",   default="2026-06-20",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--out",   default="data/goes_events.parquet",
                        help="Output Parquet path")
    args = parser.parse_args()
    download(args.start, args.end, args.out)


if __name__ == "__main__":
    main()
