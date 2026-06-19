"""
Analyse the PRADAN browse page HTML to understand the form structure
and find the correct form field names for instrument/date search.
Then try to search for SoLEXS/HEL1OS data on 2024-02-22.
"""
import re
import json
import sys
from pathlib import Path
import requests
from html.parser import HTMLParser

COOKIES = (
    "primefaces.download=null;"
    "FGTServer=03DE191863F4388C06A7AAAF7E0136FBD15060DF21FA637D82A675307CD5BF28BF8658CAFD950178C8994D;"
    "JSESSIONID=c49fc69aa9270109cbe197a7cbd5;"
    "JSESSIONID=b23c2313ee431426842bff96fad8;"
    "OAuth_Token_Request_State=02033acb-a8c6-488b-b2be-3c2ff7f60e36;"
)
BASE = "https://pradan1.issdc.gov.in"
HEADERS = {"Cookie": COOKIES}

# ── Load the saved browse HTML ─────────────────────────────────────────────────
html_path = Path("probe_browse_page.html")
if not html_path.exists():
    r = requests.get(BASE + "/al1/protected/browse.xhtml", headers=HEADERS, timeout=30)
    html_path.write_text(r.text, encoding="utf-8")

html = html_path.read_text(encoding="utf-8")

# ── Extract all form inputs ────────────────────────────────────────────────────
print("=== All <input> elements ===")
inputs = re.findall(r'<input[^>]+>', html, re.IGNORECASE)
for inp in inputs:
    name = re.search(r'name="([^"]+)"', inp)
    id_ = re.search(r'id="([^"]+)"', inp)
    val = re.search(r'value="([^"]*)"', inp)
    typ = re.search(r'type="([^"]+)"', inp)
    if name:
        print(f"  name={name.group(1)!r:50s}  type={typ.group(1) if typ else '?':10s}  value={val.group(1)[:40] if val else ''}...")

# ── Extract all <select> elements ─────────────────────────────────────────────
print("\n=== All <select> elements ===")
selects = re.findall(r'<select[^>]+>.*?</select>', html, re.IGNORECASE | re.DOTALL)
for sel in selects:
    name = re.search(r'name="([^"]+)"', sel)
    print(f"\n  SELECT name={name.group(1) if name else '?'!r}")
    options = re.findall(r'<option[^>]*value="([^"]*)"[^>]*>([^<]*)</option>', sel)
    for val, label in options[:20]:
        print(f"    option value={val!r} label={label.strip()!r}")

# ── Extract ViewState ──────────────────────────────────────────────────────────
vs = re.search(r'id="javax\.faces\.ViewState"[^>]+value="([^"]+)"', html)
if vs:
    print(f"\n=== ViewState ===\n  {vs.group(1)}")

# ── Look for instrument-related text ──────────────────────────────────────────
print("\n=== Instrument/date related content ===")
# Find all text around instrument names
for kw in ["solexs", "hel1os", "SoLEXS", "HEL1OS", "instrument", "Instrument"]:
    matches = [(m.start(), html[max(0,m.start()-100):m.end()+200]) for m in re.finditer(kw, html)]
    if matches:
        print(f"\n  '{kw}' found {len(matches)} times, first context:")
        print("  " + matches[0][1][:300].replace('\n', ' '))

# ── Look for AJAX/PrimeFaces command buttons ───────────────────────────────────
print("\n=== PrimeFaces command buttons / links ===")
buttons = re.findall(r'<[^>]*(button|commandButton|commandLink)[^>]*>', html, re.IGNORECASE)
for b in buttons[:20]:
    print(" ", b[:200])

print("\nDone.")
