"""Quick test of real PRADAN data reading."""
import sys
sys.path.insert(0, 'python')
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)-8s %(name)s — %(message)s')

from pathlib import Path
from aruna.l1_ingestion.fits_reader import FitsReader

reader = FitsReader()

PI_PATH  = Path('data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.pi.gz')
GTI_PATH = Path('data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.gti.gz')

X63_PEAK_UNIX = 1_708_641_154.0   # 2024-02-22 22:32:34 UTC

result = reader.read_solexs_pradan_pi(
    PI_PATH,
    gti_path=GTI_PATH,
    detector='SDD2',
    window_center_unix=X63_PEAK_UNIX,
    window_sec=5400,   # 90 min: 45 min before + 45 min after
)

fd = result[0]
import numpy as np

print(f'\nFitsData:')
print(f'  instrument:   {fd.instrument}')
print(f'  n_frames:     {len(fd.timestamps)}')
print(f'  energy range: {fd.energy_bins_kev[0]:.2f} – {fd.energy_bins_kev[-1]:.2f} keV  ({len(fd.energy_bins_kev)-1} bins)')
print(f'  counts shape: {fd.counts_raw.shape}')
print(f'  counts range: {np.nanmin(fd.counts_raw):.1f} – {np.nanmax(fd.counts_raw):.1f} counts/s')
print(f'  timestamps[0]: {fd.timestamps[0]}')
print(f'  timestamps[-1]: {fd.timestamps[-1]}')

# Find peak
total_counts = np.nansum(fd.counts_raw, axis=1)
peak_idx = int(np.nanargmax(total_counts))
print(f'\n  Peak at frame {peak_idx}: {fd.timestamps[peak_idx]}')
print(f'  Peak total counts/s: {total_counts[peak_idx]:.1f}')
print(f'  Peak channel spectrum (ch 30-50): {fd.counts_raw[peak_idx, 30:50].astype(int).tolist()}')
print(f'  deadtime_frac[peak_idx]: {fd.deadtime_frac[peak_idx]:.4f}')
