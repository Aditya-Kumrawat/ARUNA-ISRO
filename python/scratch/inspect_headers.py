from astropy.io import fits
from pathlib import Path

p = Path("e:/ARUNA/data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.pi.gz")
if p.exists():
    with fits.open(p) as hdul:
        for i, hdu in enumerate(hdul):
            print(f"=== HDU {i}: {hdu.name} ===")
            # print first 60 header cards to avoid flooding, or print all if short
            for k, v in list(hdu.header.items()):
                print(f"  {k} = {v}")
            if hasattr(hdu, "columns"):
                print("Columns:", hdu.columns.names)
                for col in hdu.columns:
                    print(f"  {col.name}: {col.format} {col.unit}")
else:
    print("File not found")
