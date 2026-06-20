"""Check actual PI spectrum data around the X6.3 peak."""
from astropy.io import fits
import numpy as np
from datetime import datetime, timezone

PEAK_IDX = 81154   # 2024-02-22 22:32:34 UTC

with fits.open('data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.pi.gz') as h:
    spec = h['SPECTRUM']
    data = spec.data

    # Check a few rows around the peak
    for i in [81000, 81100, PEAK_IDX, 81200]:
        row = data[i]
        tstart = float(row['TSTART'])
        counts = row['COUNTS']
        channels = row['CHANNEL']
        exposure = float(row['EXPOSURE'])
        dt = datetime.fromtimestamp(tstart, tz=timezone.utc)
        print(f"Row {i} | {dt} | TSTART={tstart:.0f} | EXPOSURE={exposure:.3f}")
        print(f"  CHANNEL[:10]: {channels[:10].tolist()}")
        print(f"  COUNTS[:10]:  {counts[:10].tolist()}")
        print(f"  total COUNTS: {np.nansum(counts):.1f}, max: {np.nanmax(counts):.1f} at ch {np.nanargmax(counts)}")

    # Quiet sun row (early morning)
    quiet_idx = 1000
    row = data[quiet_idx]
    tstart = float(row['TSTART'])
    counts = row['COUNTS']
    dt = datetime.fromtimestamp(tstart, tz=timezone.utc)
    print(f"\nQuiet sun row {quiet_idx} | {dt}")
    print(f"  total COUNTS: {np.nansum(counts):.1f}, max: {np.nanmax(counts):.1f} at ch {np.nanargmax(counts)}")
    print(f"  COUNTS[50:60]: {counts[50:60].tolist()}")

    # Check NaN rows (first few)
    nan_count = np.sum(np.isnan(data['COUNTS'].sum(axis=1)))
    print(f"\nTotal rows with NaN COUNTS: {nan_count}/{len(data)}")

    print(f"\nCHANNEL range (row 0): min={data[1000]['CHANNEL'].min()}, max={data[1000]['CHANNEL'].max()}")
