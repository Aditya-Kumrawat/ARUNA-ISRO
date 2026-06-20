from astropy.io import fits
import numpy as np
from datetime import datetime, timezone

with fits.open('data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.lc.gz') as h:
    rate = h['RATE']
    time = rate.data['TIME']
    counts = rate.data['COUNTS']

    valid = ~np.isnan(counts)
    print(f'Total rows: {len(counts)}, valid: {valid.sum()}')
    print(f'Max counts: {np.nanmax(counts):.1f} at index {np.nanargmax(counts)}')

    peak_idx = int(np.nanargmax(counts))
    peak_time = float(time[peak_idx])
    peak_dt = datetime.fromtimestamp(peak_time, tz=timezone.utc)
    print(f'Peak time: {peak_dt}')
    print(f'Peak counts: {counts[peak_idx]:.1f}')

    for i in range(max(0, peak_idx-5), min(len(time), peak_idx+6)):
        dt = datetime.fromtimestamp(float(time[i]), tz=timezone.utc)
        print(f'  [{i}] {dt.strftime("%H:%M:%S")} -> {counts[i]:.1f}')

print()
with fits.open('data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.gti.gz') as h:
    gti = h['GTI']
    for row in gti.data:
        s, e = float(row['START']), float(row['STOP'])
        ds = datetime.fromtimestamp(s, tz=timezone.utc)
        de = datetime.fromtimestamp(e, tz=timezone.utc)
        print(f'GTI: {ds} -> {de} ({e-s:.0f}s)')
