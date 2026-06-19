"""
Probe the PRADAN portal to find SoLEXS/HEL1OS Level-1 FITS paths
for the 2024-02-22 X6.3 event.
"""
import re
import sys
import requests

COOKIES = (
    "primefaces.download=null;"
    "FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C8994D;"
    "JSESSIONID=c49fc69aa9270109cbe197a7cbd5;"
    "JSESSIONID=b23c2313ee431426842bff96fad8;"
    "OAuth_Token_Request_State=02033acb-a8c6-488b-b2be-3c2ff7f60e36;"
)
BASE = "https://pradan1.issdc.gov.in"
HEADERS = {"Cookie": COOKIES}


def get_page(path):
    r = requests.get(BASE + path, headers=HEADERS, allow_redirects=False, timeout=30)
    print(f"  {path} -> HTTP {r.status_code}, {len(r.text)} bytes")
    return r


def try_direct_download(url, local_name):
    """Try to download a file and report whether it works."""
    r = requests.get(url, headers=HEADERS, allow_redirects=False, timeout=30, stream=True)
    print(f"  GET {url}")
    print(f"  -> HTTP {r.status_code}")
    if r.status_code == 200:
        ct = r.headers.get("Content-Type", "")
        cl = r.headers.get("Content-Length", "?")
        print(f"  -> Content-Type: {ct}, Content-Length: {cl}")
        return True
    return False


# ── Step 1: Parse the main payload page for download paths ────────────────────
print("\n=== Step 1: Parse main portal page for existing paths ===")
r = get_page("/al1/protected/payload.xhtml")
html = r.text

# Extract all /downloadData paths
paths = re.findall(r"/al1/protected/downloadData[^\s\"'<>&]+", html)
print(f"\nAll download paths found ({len(paths)} total):")
for p in paths:
    print(" ", p)

print("\nSoLEXS / HEL1OS / SUIT specific paths:")
for p in paths:
    pl = p.lower()
    if "solexs" in pl or "hel1os" in pl or "suit" in pl:
        print(" MATCH:", p)

# ── Step 2: Probe known URL patterns for 2024-02-22 SoLEXS ───────────────────
print("\n=== Step 2: Probe SoLEXS Level-1 data directory for 2024-02-22 ===")
# PRADAN path structure: /downloadData/{instrument}/level1/{YYYY}/{MM}/{obs_dir}/
# We know the date is 2024-02-22. Observation directories are like T24_XXXX.
# Try to browse the 2024/02 directory listing.

probe_paths = [
    "/al1/protected/downloadData/solexs/level1/2024/02/",
    "/al1/protected/downloadData/hel1os/level1/2024/02/",
    "/al1/protected/downloadData/solexs/level1/2024/02",
    "/al1/protected/downloadData/hel1os/level1/2024/02",
]

for pp in probe_paths:
    rr = requests.get(BASE + pp, headers=HEADERS, allow_redirects=False, timeout=30)
    print(f"\n  {pp}")
    print(f"  -> HTTP {rr.status_code}, {len(rr.text)} bytes")
    if rr.status_code == 200:
        print("  Content preview:", rr.text[:600])
        # Look for observation directory names
        obs_dirs = re.findall(r"[TN]\d{2}_\d{4}", rr.text)
        if obs_dirs:
            print("  Observation dirs found:", set(obs_dirs))

# ── Step 3: Try to find a data catalog / search API ───────────────────────────
print("\n=== Step 3: Look for search/catalog API endpoints ===")
api_paths = [
    "/al1/protected/searchData.xhtml",
    "/al1/protected/dataSearch.xhtml",
    "/al1/protected/catalog.xhtml",
    "/al1/protected/browse.xhtml",
    "/al1/api/data/",
]
for ap in api_paths:
    rr = requests.get(BASE + ap, headers=HEADERS, allow_redirects=False, timeout=10)
    print(f"  {ap} -> {rr.status_code}")

print("\nDone.")
