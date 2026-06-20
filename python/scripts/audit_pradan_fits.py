#!/usr/bin/env python3
"""
audit_pradan_fits.py — Scan downloaded PRADAN FITS directories and print a
date-coverage table showing which days have SoLEXS, HEL1OS, or both.

Usage:
    python python/scripts/audit_pradan_fits.py \
        --solexs-dir data/solexs/ \
        --hel1os-dir data/hel1os/

Output:
    - Date coverage table (YYYY-MM-DD | SoLEXS | HEL1OS | Status)
    - Summary: total days, aligned days, gap days
    - List of dates missing one or both instruments (cannot be used for training)
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Regex to extract date from PRADAN filenames:
# AL1_SLX_L1_20240222_v1.0  →  20240222
# AL1_HXS_L1_20240222_001_v1.0  →  20240222
_DATE_RE = re.compile(r'(\d{8})')


def _extract_date(path: Path) -> str | None:
    """Pull YYYYMMDD from a PRADAN filename."""
    m = _DATE_RE.search(path.name)
    return m.group(1) if m else None


def _scan_dir(directory: Path, instrument_key: str) -> dict[str, list[Path]]:
    """
    Scan a directory (flat or one-subdir-per-day) for FITS files.
    Returns {date_str: [file_paths]}.
    """
    root = Path(directory)
    if not root.exists():
        print(f"  [WARN] directory not found: {root}")
        return {}

    # Collect all FITS-like files (plain or gzip-compressed)
    patterns = ("*.fits", "*.fit", "*.fits.gz", "**/*.fits", "**/*.fits.gz",
                "**/*.pi.gz", "**/*.lc.gz", "**/*.gti.gz")
    all_files: list[Path] = []
    for p in patterns:
        all_files.extend(root.glob(p))

    # Also include top-level directories whose name looks like a PRADAN bundle
    # e.g.  data/solexs/AL1_SLX_L1_20240222_v1.0/
    for entry in root.iterdir():
        if entry.is_dir() and _DATE_RE.search(entry.name):
            for p in ("*.fits", "*.fits.gz", "**/*.fits.gz", "**/*.pi.gz", "**/*.lc.gz"):
                all_files.extend(entry.glob(p))

    date_map: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(set(all_files)):
        d = _extract_date(f)
        if d:
            date_map[d].append(f)

    return dict(date_map)


def audit(solexs_dir: str, hel1os_dir: str) -> None:
    print("\n=== ARUNA PRADAN FITS Coverage Audit ===\n")

    sol_map = _scan_dir(Path(solexs_dir), "SoLEXS")
    hel_map = _scan_dir(Path(hel1os_dir), "HEL1OS")

    all_dates = sorted(set(sol_map) | set(hel_map))

    if not all_dates:
        print("No FITS files found. Check --solexs-dir and --hel1os-dir paths.")
        sys.exit(1)

    # Print table
    col = 14
    print(f"{'DATE':<12}  {'SoLEXS':<8}  {'HEL1OS':<8}  STATUS")
    print("-" * 50)

    aligned = 0
    sol_only = 0
    hel_only = 0
    missing = 0

    gap_dates: list[str] = []

    for d in all_dates:
        has_sol = d in sol_map
        has_hel = d in hel_map
        sol_count = len(sol_map.get(d, []))
        hel_count = len(hel_map.get(d, []))

        if has_sol and has_hel:
            status = "OK — dual-instrument"
            aligned += 1
        elif has_sol:
            status = "WARN — SoLEXS only (unusable for dual-stream TCN)"
            sol_only += 1
            gap_dates.append(f"{d}  [missing HEL1OS]")
        elif has_hel:
            status = "WARN — HEL1OS only (unusable for dual-stream TCN)"
            hel_only += 1
            gap_dates.append(f"{d}  [missing SoLEXS]")
        else:
            missing += 1

        date_fmt = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        sol_str = str(sol_count) if has_sol else "—"
        hel_str = str(hel_count) if has_hel else "—"
        print(f"{date_fmt:<12}  {sol_str:<8}  {hel_str:<8}  {status}")

    # Summary
    total = len(all_dates)
    print("\n" + "=" * 50)
    print(f"Total dates found : {total}")
    print(f"  Dual-instrument : {aligned}  ({aligned/total*100:.1f}%)  ← usable for training")
    print(f"  SoLEXS only     : {sol_only}")
    print(f"  HEL1OS only     : {hel_only}")
    print(f"  Total usable    : {aligned}")

    # Temporal span
    if all_dates:
        first = all_dates[0]
        last  = all_dates[-1]
        first_fmt = f"{first[:4]}-{first[4:6]}-{first[6:]}"
        last_fmt  = f"{last[:4]}-{last[4:6]}-{last[6:]}"
        print(f"\nDate range        : {first_fmt} → {last_fmt}")

    # Expected training split
    print("\nExpected temporal split (PRD §8.9):")
    train_days = [d for d in all_dates if d in sol_map and d in hel_map
                  and '20230902' <= d <= '20240630']
    val_days   = [d for d in all_dates if d in sol_map and d in hel_map
                  and '20240701' <= d <= '20241231']
    test_days  = [d for d in all_dates if d in sol_map and d in hel_map
                  and '20250101' <= d <= '20250630']
    print(f"  TRAIN (2023-09-02 → 2024-06-30): {len(train_days)} days")
    print(f"  VAL   (2024-07-01 → 2024-12-31): {len(val_days)} days")
    print(f"  TEST  (2025-01-01 → 2025-06-30): {len(test_days)} days")

    if gap_dates:
        print(f"\nDates with missing instrument ({len(gap_dates)}):")
        for g in gap_dates[:20]:
            print(f"  {g}")
        if len(gap_dates) > 20:
            print(f"  ... and {len(gap_dates) - 20} more")

    print("\nNext step: python python/scripts/download_goes_labels.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PRADAN FITS coverage")
    parser.add_argument("--solexs-dir", default="data/solexs",
                        help="Root directory containing SoLEXS FITS files/subdirs")
    parser.add_argument("--hel1os-dir", default="data/hel1os",
                        help="Root directory containing HEL1OS FITS files/subdirs")
    args = parser.parse_args()
    audit(args.solexs_dir, args.hel1os_dir)


if __name__ == "__main__":
    main()
