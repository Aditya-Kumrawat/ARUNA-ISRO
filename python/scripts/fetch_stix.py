"""Get STIX ELUT energy channel boundaries and ephemeris distance for X6.3."""
import sys, json
sys.path.insert(0, 'python')
from stixdcpy.net import Request
import numpy as np

r = Request()

# Get ELUT
print("=== ELUT data ===")
elut = r.fetch_elut('2024-02-22T22:00:00')
elut_data = elut.get('data', None)
print(type(elut_data))
if isinstance(elut_data, dict):
    print("Keys:", list(elut_data.keys()))
    # Look for energy channel data
    for k, v in elut_data.items():
        if hasattr(v, '__len__') and len(v) > 5:
            print(f"  {k}: len={len(v)}  first={v[:5]}")
        else:
            print(f"  {k}: {v}")
elif isinstance(elut_data, list):
    print(f"List len: {len(elut_data)}")
    for item in elut_data[:3]:
        print(" ", item)

# Get ephemeris for Solar Orbiter distance
print()
print("=== Ephemeris ===")
eph = r.request_ephemeris('2024-02-22T22:00:00', '2024-02-22T23:00:00')
eph_data = eph.get('data', eph) if isinstance(eph, dict) else eph
print(type(eph_data))
if isinstance(eph_data, dict):
    print("Keys:", list(eph_data.keys()))
    for k, v in eph_data.items():
        if hasattr(v, '__len__'):
            print(f"  {k}: len={len(v)}  sample={v[:3] if len(v)>3 else v}")
        else:
            print(f"  {k}: {v}")
elif isinstance(eph_data, list):
    for item in eph_data[:3]:
        print(" ", item)

# Standard STIX energy channel boundaries (ELUT)
# Per the STIX calibration documentation, the 32 science energy bins are:
STIX_E_LOWER = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 70, 76, 84, 100, 120, 150, 150])
STIX_E_UPPER = np.array([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 25, 28, 32, 36, 40, 45, 50, 56, 63, 70, 76, 84, 100, 120, 150, 300, 300])
print()
print("=== Standard STIX energy bins (32 channels) ===")
for i, (lo, hi) in enumerate(zip(STIX_E_LOWER, STIX_E_UPPER)):
    # Mark channels that overlap with HEL1OS bins
    he_bin = ""
    if 10 <= lo < 25 or (lo < 10 and hi > 10): he_bin = " [HEL1OS 10-25]"
    if 25 <= lo < 40 or (lo < 25 and hi > 25): he_bin = " [HEL1OS 25-40]"
    if 40 <= lo < 55 or (lo < 40 and hi > 40): he_bin = " [HEL1OS 40-55]"
    if 55 <= lo < 70 or (lo < 55 and hi > 55): he_bin = " [HEL1OS 55-70]"
    print(f"  ch{i:2d}: {lo:4.0f}-{hi:4.0f} keV{he_bin}")
