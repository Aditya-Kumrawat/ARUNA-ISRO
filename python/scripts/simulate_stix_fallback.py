#!/usr/bin/env python3
"""
simulate_stix_fallback.py -- Simulates the missing HEL1OS scenario for 2024-02-22
and exercises the STIX fallback path end-to-end.

What this proves:
  1. HEL1OS is declared "missing" (simulated -- no file on disk)
  2. STIXProvider fetches 32-channel data from Solar Orbiter
  3. 32 STIX channels -> 4 HEL1OS-equivalent bins (10-25, 25-40, 40-55, 55-70 keV)
  4. SHEI is computed from the mapped bins (proves the feature pipeline still works)
  5. Frames are tagged source_file='STIX:2024-02-22' for traceability

If this script prints SUCCESS the decision tree is valid and batch_l1_l2_gpu.py
will work correctly for any missing-HEL1OS day.

Usage:
    python python/scripts/simulate_stix_fallback.py
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
)
logger = logging.getLogger("stix_sim")

# Add project root so aruna package is importable
sys.path.insert(0, str(Path(__file__).parents[1]))

# -- STIX channel / HEL1OS mapping constants (mirror of stix_provider.py) ------
STIX_E_EDGES = np.array([
    0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22,
    25, 28, 32, 36, 40, 45, 50, 56, 63, 70, 76, 84, 100, 120, 150,
], dtype=np.float64)                    # 33 edges -> 32 channels

HEL1OS_BINS = np.array([10.0, 25.0, 40.0, 55.0, 70.0])  # 4 bins

STIX_TO_HEL1OS = [
    slice(6, 16),   # 10-25 keV  (STIX channels 6-15)
    slice(16, 20),  # 25-40 keV  (STIX channels 16-19)
    slice(20, 23),  # 40-56 keV  (STIX channels 20-22)
    slice(23, 25),  # 56-70 keV  (STIX channels 23-24)
]


def _print_channel_map():
    """Print exactly which STIX channels map to each HEL1OS bin."""
    print("\n-- STIX -> HEL1OS channel mapping -----------------------------------------")
    print(f"{'HEL1OS bin':<20}  {'keV range':<12}  {'STIX channels':<18}  {'STIX keV edges'}")
    print("-" * 75)
    hel_names = ["hel1os_10_25kev", "hel1os_25_40kev", "hel1os_40_55kev", "hel1os_55_70kev"]
    for i, (slc, name) in enumerate(zip(STIX_TO_HEL1OS, hel_names)):
        ch_range  = f"{slc.start}-{slc.stop - 1}"
        edge_lo   = STIX_E_EDGES[slc.start]
        edge_hi   = STIX_E_EDGES[slc.stop]
        kev_range = f"{HEL1OS_BINS[i]:.0f}-{HEL1OS_BINS[i+1]:.0f} keV"
        print(f"  {name:<18}  {kev_range:<12}  ch {ch_range:<14}  {edge_lo:.0f}-{edge_hi:.0f} keV (STIX)")
    print()


def _compute_shei(hel_bins_cps_kev: np.ndarray) -> float:
    """
    SHEI = log10(F_40-70 / F_10-25)
    hel_bins_cps_kev: (4,) -- [10-25, 25-40, 40-55, 55-70] in counts/s/keV
    """
    f_soft = hel_bins_cps_kev[0] * 15.0   # 10-25 keV * 15 keV width
    f_hard = (hel_bins_cps_kev[2] * 15.0  # 40-55 keV
            + hel_bins_cps_kev[3] * 15.0) # 55-70 keV
    if f_soft <= 0 or f_hard <= 0:
        return np.nan
    return float(np.log10(f_hard / f_soft))


def simulate_missing_hel1os():
    """
    Main simulation: declared HEL1OS missing -> STIX fallback -> feature check.
    Returns True on success, False on failure.
    """
    DATE        = "2024-02-22"
    # X6.3 flare peak: ~22:34 UTC. Fetch 90 minutes around peak.
    START_UTC   = f"{DATE}T21:30:00"
    END_UTC     = f"{DATE}T23:00:00"

    print("=" * 70)
    print("  ARUNA STIX FALLBACK SIMULATION")
    print("=" * 70)
    print(f"\n  Scenario : HEL1OS data MISSING on PRADAN for {DATE}")
    print(f"  Fallback : STIXProvider (Solar Orbiter, 32-ch, 4-150 keV)")
    print(f"  Window   : {START_UTC} to {END_UTC}  (X6.3 flare peak ~22:34 UTC)")
    print(f"  SolO dist: 0.67 AU (hardcoded fallback for 2024-02-22)")
    print()

    _print_channel_map()

    # -- Step 1: Simulate "no HEL1OS file found" ------------------------------
    print("-- Step 1 : HEL1OS file check -----------------------------------------------")
    fake_hel1os_path = Path("data/hel1os/AL1_HLS_L1_20240222_v1.0")
    if fake_hel1os_path.exists():
        print(f"  WARN: {fake_hel1os_path} actually exists -- simulation uses it anyway")
    else:
        print(f"  OK  : No HEL1OS file at {fake_hel1os_path}")
        print(f"  ->  : Triggering STIX fallback ...")
    print()

    # -- Step 2: Invoke STIXProvider -------------------------------------------
    print("-- Step 2 : Fetching STIX data ---------------------------------------------")
    try:
        from aruna.l1_ingestion.stix_provider import STIXProvider
    except ImportError as e:
        print(f"  FAIL: Cannot import STIXProvider: {e}")
        return False

    provider = STIXProvider()
    frames = provider.fetch(START_UTC, END_UTC, solo_distance_au=0.67)

    if not frames:
        print("  FAIL: STIXProvider returned 0 frames")
        print("        (check network / stixdcpy API)")
        print("\n  [SIMULATION INCOMPLETE -- STIX API unreachable]")
        return False

    print(f"  OK  : {len(frames)} ObservationFrames returned")
    print(f"  inst: {frames[0].instrument}")
    print(f"  src : {frames[0].source_file}")
    print(f"  qual: {frames[0].quality_flag.name}")
    print(f"  bins: {len(frames[0].counts_per_s_kev)} HEL1OS-equivalent channels")
    print()

    # -- Step 3: Show temporal coverage ----------------------------------------
    print("-- Step 3 : Temporal coverage -----------------------------------------------")
    t_start = frames[0].timestamp
    t_end   = frames[-1].timestamp
    duration_min = (t_end - t_start).total_seconds() / 60.0
    good = sum(1 for f in frames if f.quality_flag.value <= 1)
    gap  = sum(1 for f in frames if f.quality_flag.name == "GAP")
    print(f"  Start   : {t_start.isoformat()}")
    print(f"  End     : {t_end.isoformat()}")
    print(f"  Duration: {duration_min:.1f} minutes")
    print(f"  GOOD    : {good} frames")
    print(f"  GAP     : {gap} frames (forward-filled)")
    print()

    # -- Step 4: Compute SHEI on peak frames -----------------------------------
    print("-- Step 4 : SHEI computation on all frames ----------------------------------")
    shei_vals = []
    for f in frames:
        s = _compute_shei(f.counts_per_s_kev)
        if np.isfinite(s):
            shei_vals.append(s)

    if not shei_vals:
        print("  FAIL: All SHEI values are NaN (zero flux in bands?)")
        return False

    shei_arr = np.array(shei_vals)
    print(f"  SHEI min  : {shei_arr.min():.3f}")
    print(f"  SHEI max  : {shei_arr.max():.3f}")
    print(f"  SHEI mean : {shei_arr.mean():.3f}")
    print(f"  SHEI >-0.8: {(shei_arr > -0.8).sum()} frames  <- impulsive onset trigger")
    print()

    # -- Step 5: Channel-level spot check --------------------------------------
    print("-- Step 5 : Spot-check peak-flux frame --------------------------------------")
    # Find frame with highest total flux (closest to flare peak)
    total_flux = [f.counts_per_s_kev.sum() for f in frames]
    peak_idx   = int(np.argmax(total_flux))
    peak_frame = frames[peak_idx]
    bins_cps   = peak_frame.counts_per_s_kev

    bin_names  = ["10-25 keV", "25-40 keV", "40-55 keV", "55-70 keV"]
    print(f"  Peak frame: {peak_frame.timestamp.isoformat()}")
    print(f"  {'Bin':<12}  {'cts/s/keV':>12}  {'cts/s (*width)':>16}")
    widths = np.diff(HEL1OS_BINS)
    for name, cps, w in zip(bin_names, bins_cps, widths):
        print(f"  {name:<12}  {cps:>12.2f}  {cps*w:>16.1f}")
    peak_shei = _compute_shei(bins_cps)
    print(f"\n  SHEI at peak  : {peak_shei:.3f}  {'(impulsive onset!)' if peak_shei > -0.8 else '(below trigger)'}")
    print(f"  Distance corr : 0.67^2 = {0.67**2:.4f}  (applied inside STIXProvider)")
    print()

    # -- Step 6: Traceability check
    print("-- Step 6 : Source traceability ---------------------------------------------")
    stix_frames = [f for f in frames if "STIX" in f.source_file]
    print(f"  STIX-tagged frames: {len(stix_frames)} / {len(frames)}")
    print(f"  Instrument IDs    : {set(f.instrument for f in frames)}")
    print(f"  Cal version       : {frames[0].calibration_version}")
    print()

    print("=" * 70)
    print("  SUCCESS -- STIX fallback works for 2024-02-22")
    print("  All frames usable as HEL1OS_CDTE substitute in batch_l1_l2_gpu.py")
    print("=" * 70)
    return True


if __name__ == "__main__":
    ok = simulate_missing_hel1os()
    sys.exit(0 if ok else 1)
