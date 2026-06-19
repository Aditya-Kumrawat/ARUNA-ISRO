import gzip
from pathlib import Path
from astropy.io import fits
import numpy as np

def main():
    path = Path("e:/ARUNA/data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.pi.gz")
    if not path.exists():
        print("Path does not exist!")
        return

    with fits.open(path) as hdul:
        hdul.info()
        hdr0 = hdul[0].header
        print("\nPrimary Header Keywords:")
        for k in ['MISSION', 'TELESCOP', 'INSTRUME', 'OBS_DATE', 'TSTART', 'TSTOP']:
            print(f"  {k} = {hdr0.get(k)}")

        hdu = hdul['SPECTRUM']
        print("\nSpectrum Header Keywords:")
        for k in ['EXTNAME', 'TIMEREF', 'TSTART', 'TSTOP', 'EXPOSURE']:
            print(f"  {k} = {hdu.header.get(k)}")

        print("\nData Columns:")
        print(hdu.columns.names)
        
        # Read a slice of data around the flare peak
        # In our summary, the peak was found at index 81154
        # Let's inspect rows around 81150-81160
        tstart = hdu.data['TSTART'][81150:81160]
        telapse = hdu.data['TELAPSE'][81150:81160]
        spec_num = hdu.data['SPEC_NUM'][81150:81160]
        counts = hdu.data['COUNTS'][81150:81160]
        exposure = hdu.data['EXPOSURE'][81150:81160]

        print(f"\nSample rows around peak (81150-81160):")
        for i in range(10):
            idx = 81150 + i
            # Sum of counts across all channels for this row
            sum_counts = np.nansum(counts[i])
            print(f"  Row {idx}: TSTART={tstart[i]}, TELAPSE={telapse[i]}, SPEC_NUM={spec_num[i]}, Sum(COUNTS)={sum_counts}, EXPOSURE={exposure[i]}")

        # Let's also check channel mapping
        channel = hdu.data['CHANNEL'][0]
        print(f"\nChannel array (first 10): {channel[:10]}")
        print(f"Channel array length: {len(channel)}")
        print(f"Counts array shape for row 0: {counts[0].shape}")

if __name__ == "__main__":
    main()
