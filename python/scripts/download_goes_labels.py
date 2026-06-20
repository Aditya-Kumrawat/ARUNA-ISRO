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
from datetime import datetime
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


# ── Fallback: NOAA NCEI CSV download ──────────────────────────────────────────

def _download_via_ncei(start: str, end: str) -> pd.DataFrame:
    """
    Fallback: download GOES flare list from NOAA NCEI directly.
    Uses the legacy solar events CSV — always publicly accessible.
    """
    import urllib.request
    import io

    print("HEK unavailable — trying NOAA NCEI CSV fallback …")

    start_year = int(start[:4])
    end_year   = int(end[:4])
    all_rows   = []

    for year in range(start_year, end_year + 1):
        url = (
            f"https://www.ngdc.noaa.gov/stp/space-weather/solar-data/"
            f"solar-features/solar-flares/x-rays/goes/xrs/"
            f"goes-xrs-report_{year}.txt"
        )
        try:
            print(f"  Fetching {year} from NCEI …")
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode("latin-1")
            # Parse fixed-width NCEI format
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or len(line) < 40:
                    continue
                try:
                    # NCEI format: YYMMDDHHMM HHMM HHMM CLASS ...
                    parts = line.split()
                    if len(parts) < 4:
                        continue
                    yymmddhhmm = parts[0]
                    yr = int(yymmddhhmm[:2])
                    yr4 = 2000 + yr if yr < 50 else 1900 + yr
                    mo = int(yymmddhhmm[2:4])
                    dy = int(yymmddhhmm[4:6])
                    hh_start = int(yymmddhhmm[6:8])
                    mm_start = int(yymmddhhmm[8:10])

                    peak_str = parts[1] if len(parts) > 1 else "0000"
                    hh_peak = int(peak_str[:2])
                    mm_peak = int(peak_str[2:4])

                    end_str = parts[2] if len(parts) > 2 else "0000"
                    hh_end = int(end_str[:2])
                    mm_end = int(end_str[2:4])

                    cls_raw = parts[3] if len(parts) > 3 else ""
                    letter, number = parse_goes_class(cls_raw)
                    if not letter:
                        continue

                    start_ts = pd.Timestamp(yr4, mo, dy, hh_start, mm_start, tz="UTC")
                    peak_ts  = pd.Timestamp(yr4, mo, dy, hh_peak,  mm_peak,  tz="UTC")
                    end_ts   = pd.Timestamp(yr4, mo, dy, hh_end,   mm_end,   tz="UTC")

                    all_rows.append({
                        "start_time": start_ts,
                        "peak_time":  peak_ts,
                        "end_time":   end_ts,
                        "goes_class": cls_raw.upper(),
                        "goes_class_letter": letter,
                        "goes_class_number": number,
                        "noaa_ar": "",
                    })
                except (ValueError, IndexError):
                    continue
        except Exception as exc:
            print(f"  Could not fetch {year}: {exc}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts   = pd.Timestamp(end,   tz="UTC")
        df = df[(df["peak_time"] >= start_ts) & (df["peak_time"] <= end_ts)]
        df = df.sort_values("peak_time").reset_index(drop=True)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def download(start: str, end: str, out_path: str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Try HEK first, fall back to NCEI CSV
    try:
        df = _download_via_hek(start, end)
    except Exception as exc:
        print(f"HEK query failed ({exc}) — trying NCEI fallback …")
        df = _download_via_ncei(start, end)

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
