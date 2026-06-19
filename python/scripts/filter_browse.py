"""
Navigate PRADAN browse.xhtml with ?id=solexs and ?id=hel1os,
then filter for Level-1 data on 2024-02-22 to get the actual FITS filenames.
"""
import re
import json
import time
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
BROWSE = BASE + "/al1/protected/browse.xhtml"

sess = requests.Session()
sess.headers.update({
    "Cookie": COOKIES,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": BASE + "/al1/protected/payload.xhtml",
})


def get_vs(html):
    m = re.search(r'name="javax\.faces\.ViewState"[^>]+value="([^"]+)"', html)
    return m.group(1) if m else ""


def extract_fits_paths(html, instrument):
    """Extract all FITS download paths from the browse results table."""
    # Pattern: href="/al1/protected/downloadData/solexs/..."
    paths = re.findall(
        rf'/al1/protected/downloadData/{instrument}/[^\s"\'<>&]+\.fits(?:\?[^\s"\'<>&]*)?',
        html, re.IGNORECASE
    )
    return paths


def filter_browse(instrument_id, start_dt, end_dt, level="level1"):
    """
    Navigate to browse.xhtml?id={instrument} and apply date filter.
    Returns HTML of results page.
    """
    print(f"\n{'='*60}")
    print(f"Browsing: {instrument_id.upper()} — {start_dt} to {end_dt}")
    print('='*60)

    # Step 1: GET the instrument page to get ViewState
    url = f"{BROWSE}?id={instrument_id}"
    r = sess.get(url, timeout=30)
    print(f"  GET {url} -> {r.status_code}, {len(r.text)} bytes")
    if r.status_code != 200:
        print(f"  ERROR: Got {r.status_code}")
        return None

    html = r.text
    vs = get_vs(html)
    print(f"  ViewState: {vs[:30]}...")

    # Save for inspection
    Path(f"browse_{instrument_id}.html").write_text(html, encoding="utf-8")

    # Check what the table shows already (no filter)
    paths_no_filter = extract_fits_paths(html, instrument_id)
    print(f"  Without filter: {len(paths_no_filter)} FITS paths found")
    for p in paths_no_filter[:5]:
        print(f"    {p}")

    # Step 2: POST with the date filter
    # Form field names from earlier analysis:
    # filterForm:filterTable:0:attr = ObservationTime;TIME_DURATION;...
    # filterForm:filterTable:0:opr = TIME_DURATION,TimeDurationIn
    # filterForm:filterTable:0:datetime1_input = "2024-02-22 00:00:00"
    # filterForm:filterTable:0:datetime2_input = "2024-02-22 23:59:59"
    # filterForm:filterButton = Filter (submit)

    obs_time_value = (
        "ObservationTime;TIME_DURATION;"
        "label.Product_Details.Observation_Details.Time_Coordinates.start_date_time, "
        "label.Product_Details.Observation_Details.Time_Coordinates.stop_date_time"
    )

    filter_data = {
        "filterForm": "filterForm",
        "filterForm:j_idt53": "DOWNLOAD",
        "filterForm:filterTable:0:attr": obs_time_value,
        "filterForm:filterTable:0:opr": "TIME_DURATION,TimeDurationIn",
        "filterForm:filterTable:0:datetime1_input": f"{start_dt} 00:00:00",
        "filterForm:filterTable:0:datetime2_input": f"{end_dt} 23:59:59",
        "filterForm:filterButton": "Filter",
        "javax.faces.ViewState": vs,
    }

    post_headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer": url,
        "Cookie": COOKIES,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }

    print(f"\n  POSTing filter: {start_dt} to {end_dt}")
    r2 = sess.post(BROWSE, data=filter_data, headers=post_headers, timeout=60, allow_redirects=True)
    print(f"  POST -> {r2.status_code}, {len(r2.text)} bytes")

    html2 = r2.text
    Path(f"browse_{instrument_id}_filtered.html").write_text(html2, encoding="utf-8")

    # Extract FITS paths from filtered results
    paths = extract_fits_paths(html2, instrument_id)
    print(f"\n  FITS paths found after filter: {len(paths)}")
    for p in paths[:50]:
        print(f"    {p}")

    # Also look for the row count in paginator
    row_count = re.search(r"rowCount:(\d+)", html2)
    if row_count:
        print(f"\n  Total rows matching filter: {row_count.group(1)}")

    # Look for any error messages
    errors = re.findall(r'class="[^"]*error[^"]*"[^>]*>([^<]+)', html2, re.IGNORECASE)
    for e in errors[:5]:
        print(f"  Error: {e.strip()}")

    return html2, paths


# ── Run for SoLEXS ────────────────────────────────────────────────────────────
result_solexs = filter_browse("solexs", "2024-02-22", "2024-02-22")
time.sleep(2)

# ── Run for HEL1OS ───────────────────────────────────────────────────────────
result_hel1os = filter_browse("hel1os", "2024-02-22", "2024-02-22")

print("\n\n=== SUMMARY ===")
if result_solexs:
    _, sp = result_solexs
    print(f"SoLEXS  2024-02-22: {len(sp)} files")
if result_hel1os:
    _, hp = result_hel1os
    print(f"HEL1OS  2024-02-22: {len(hp)} files")
