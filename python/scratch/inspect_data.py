import numpy as np
from astropy.io import fits
from pathlib import Path

p_lc = Path("e:/ARUNA/data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.lc.gz")
p_pi = Path("e:/ARUNA/data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.pi.gz")

print("=== LIGHTCURVE INSPECTION ===")
with fits.open(p_lc) as hdul:
    data = hdul['RATE'].data
    times = data['TIME']
    counts = data['COUNTS']
    valid_idx = np.where(np.isfinite(counts))[0]
    print(f"Total rows: {len(times)}")
    print(f"Valid count rows: {len(valid_idx)}")
    if len(valid_idx) > 0:
        print(f"First valid index: {valid_idx[0]}, time: {times[valid_idx[0]]}, counts: {counts[valid_idx[0]]}")
        print(f"Last valid index: {valid_idx[-1]}, time: {times[valid_idx[-1]]}, counts: {counts[valid_idx[-1]]}")
        # Find peak counts
        max_idx = valid_idx[np.argmax(counts[valid_idx])]
        print(f"Max counts: {counts[max_idx]} at index {max_idx}, time {times[max_idx]}")
        # Let's print some stats around the peak
        print(f"Peak time datetime: {np.datetime64(int(times[max_idx]), 's')}")
    else:
        print("No valid count rows found in rate table!")

print("\n=== PI INSPECTION ===")
with fits.open(p_pi) as hdul:
    data = hdul['SPECTRUM'].data
    tstarts = data['TSTART']
    counts = data['COUNTS']
    exposures = data['EXPOSURE']
    
    # Check if there is any row where counts are finite
    # Note: counts is a 2D array of shape (86400, 340)
    # Check how many rows have all finite counts, or at least some finite counts
    has_finite = np.any(np.isfinite(counts), axis=1)
    finite_idx = np.where(has_finite)[0]
    print(f"Total spectrum rows: {len(tstarts)}")
    print(f"Rows with finite counts: {len(finite_idx)}")
    if len(finite_idx) > 0:
        print(f"First finite spectrum index: {finite_idx[0]}, tstart: {tstarts[finite_idx[0]]}")
        # Let's print the sum of counts per row
        row_sums = np.nansum(counts, axis=1)
        max_sum_idx = np.argmax(row_sums)
        print(f"Max spectrum sum: {row_sums[max_sum_idx]} at index {max_sum_idx}, tstart {tstarts[max_sum_idx]}")
        print(f"Spectrum shape: {counts.shape}")
        print(f"First row of non-nan spectrum: {counts[finite_idx[0]][:20]}...")
    else:
        print("No finite counts found in spectrum table!")
