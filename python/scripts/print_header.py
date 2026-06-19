from astropy.io import fits

path = "e:/ARUNA/data/solexs_20240222/AL1_SLX_L1_20240222_v1.0/SDD2/AL1_SOLEXS_20240222_SDD2_L1.pi.gz"
with fits.open(path) as hdul:
    print("=== PRIMARY HEADER ===")
    print(hdul[0].header.tostring('\n'))
    print("\n=== SPECTRUM HEADER ===")
    print(hdul[1].header.tostring('\n'))
