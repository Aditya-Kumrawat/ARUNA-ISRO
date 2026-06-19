from astropy.io import fits

path = "e:/ARUNA/data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.gti.gz"
with fits.open(path) as hdul:
    hdul.info()
    print("=== PRIMARY HEADER ===")
    print(hdul[0].header.tostring('\n')[:500])
    print("\n=== GTI HEADER ===")
    print(hdul[1].header.tostring('\n')[:1000])
    print("\n=== Columns ===")
    print(hdul[1].columns.names)
    print("\nData:")
    print(hdul[1].data)
