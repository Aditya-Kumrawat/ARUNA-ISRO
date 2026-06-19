import gzip
from pathlib import Path
from astropy.io import fits

def inspect(path):
    print(f"\nInspecting: {path.name}")
    try:
        with fits.open(path) as hdul:
            hdul.info()
            for hdu in hdul:
                print(f"HDU: {hdu.name}")
                print("Header sample:")
                for k, v in list(hdu.header.items())[:15]:
                    print(f"  {k} = {v}")
                if hasattr(hdu, "columns"):
                    print("Columns:", hdu.columns.names)
                    if len(hdu.data) > 0:
                        print("Data sample (first row):")
                        for col in hdu.columns.names[:5]:
                            print(f"  {col}: {hdu.data[col][0]}")
    except Exception as e:
        print(f"Error reading fits: {e}")

def main():
    base = Path("e:/ARUNA/data/solexs_20240222/AL1_SLX_L1_20240222_v1.0")
    for p in base.rglob("*.gz"):
        inspect(p)

if __name__ == "__main__":
    main()
