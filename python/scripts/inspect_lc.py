from astropy.io import fits

path = "e:/ARUNA/data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.lc.gz"
with fits.open(path) as hdul:
    hdul.info()
    print("=== PRIMARY HEADER ===")
    print(hdul[0].header.tostring('\n')[:500])
    print("\n=== LIGHTCURVE HEADER ===")
    print(hdul[1].header.tostring('\n')[:1000])
    print("\n=== Columns ===")
    print(hdul[1].columns.names)
    print("\nSample Data:")
    for name in hdul[1].columns.names:
        print(f"  {name}: {hdul[1].data[name][:5]}")
