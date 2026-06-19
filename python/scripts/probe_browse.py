"""
Probe the PRADAN browse page and try to find SoLEXS/HEL1OS data for 2024-02-22.
"""
import re
import requests
from pathlib import Path

COOKIES = (
    "primefaces.download=null;"
    "FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C8994D;"
    "JSESSIONID=c49fc69aa9270109cbe197a7cbd5;"
    "JSESSIONID=b23c2313ee431426842bff96fad8;"
    "OAuth_Token_Request_State=02033acb-a8c6-488b-b2be-3c2ff7f60e36;"
)
BASE = "https://pradan1.issdc.gov.in"
HEADERS = {"Cookie": COOKIES}


def get_and_save(path, label):
    r = requests.get(BASE + path, headers=HEADERS, allow_redirects=True, timeout=30)
    print(f"\n=== {label} ===")
    print(f"  URL: {BASE+path}")
    print(f"  HTTP {r.status_code}, {len(r.text)} bytes")
    if r.status_code == 200:
        out = Path(f"probe_{label.replace(' ', '_')}.html")
        out.write_text(r.text, encoding="utf-8")
        print(f"  Saved to {out}")
        # Print first 2000 chars
        print(r.text[:2000])
        # Look for instrument / date mentions
        text_lower = r.text.lower()
        for kw in ["solexs", "hel1os", "2024-02", "2024/02", "level1", "fits"]:
            if kw in text_lower:
                print(f"  KEYWORD FOUND: '{kw}'")
    return r


# ── Browse page ───────────────────────────────────────────────────────────────
r_browse = get_and_save("/al1/protected/browse.xhtml", "browse_page")

# ── Try to POST to browse with instrument=solexs date=2024-02-22 ──────────────
print("\n=== Try POST to browse for SoLEXS 2024-02-22 ===")
# Extract viewstate from browse page if PrimeFaces
vs_match = re.search(r'javax\.faces\.ViewState[^>]+value="([^"]+)"', r_browse.text)
view_state = vs_match.group(1) if vs_match else ""
print(f"  ViewState: {view_state[:40] if view_state else 'NOT FOUND'}")

# Common PrimeFaces form elements
form_data = {
    "javax.faces.ViewState": view_state,
    "j_idt12:instrument": "solexs",
    "j_idt12:startDate": "2024-02-22",
    "j_idt12:endDate": "2024-02-22",
    "j_idt12:level": "level1",
}
post_headers = dict(HEADERS)
post_headers["Content-Type"] = "application/x-www-form-urlencoded"
post_headers["Referer"] = BASE + "/al1/protected/browse.xhtml"

r_post = requests.post(
    BASE + "/al1/protected/browse.xhtml",
    headers=post_headers,
    data=form_data,
    allow_redirects=True,
    timeout=30,
)
print(f"  POST -> HTTP {r_post.status_code}, {len(r_post.text)} bytes")
if r_post.status_code == 200:
    # Look for download paths
    paths = re.findall(r"/al1/protected/downloadData[^\s\"'<>&?]+\.fits", r_post.text)
    print(f"  Found {len(paths)} FITS paths")
    for p in paths[:20]:
        print(f"    {p}")

print("\nDone.")
