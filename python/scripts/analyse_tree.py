"""
Find the PrimeFaces tree AJAX call structure and navigate to SoLEXS / HEL1OS
nodes to get the Level-1 file list for 2024-02-22.
"""
import re
import json
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
BROWSE_URL = BASE + "/al1/protected/browse.xhtml"

sess = requests.Session()
sess.headers.update({
    "Cookie": COOKIES,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
})

# Load saved browse_full.html
html = Path("browse_full.html").read_text(encoding="utf-8")

# ── Find all <script> blocks with PrimeFaces widget config ─────────────────────
print("=== PrimeFaces widget configs ===")
scripts = re.findall(r'PrimeFaces\.widget\.[A-Za-z]+\((\{.*?\})\)', html, re.DOTALL)
for s in scripts[:10]:
    print(s[:300])
    print("---")

# ── Find tree-specific JS ──────────────────────────────────────────────────────
print("\n=== Tree JS init ===")
tree_js = re.findall(r"PrimeFaces\s*\.\s*cw\s*\(\s*'Tree'[^;]+;", html, re.DOTALL)
for tj in tree_js:
    print(tj[:500])

# ── Find all PrimeFaces.cw() calls ─────────────────────────────────────────────
print("\n=== All PrimeFaces.cw() calls ===")
cws = re.findall(r"PrimeFaces\.cw\s*\([^;]+;", html, re.DOTALL)
for cw in cws[:20]:
    print(cw[:300])
    print("---")

# ── Find node keys in tree ──────────────────────────────────────────────────────
print("\n=== Tree node data ===")
# Look for data-rowkey or rowKey patterns
node_keys = re.findall(r'data-rowkey="([^"]+)"', html)
print(f"data-rowkey values: {node_keys}")

node_types = re.findall(r'data-nodetype="([^"]+)"', html)
print(f"data-nodetype values: {node_types}")

# ── Find the form ID for navigator ─────────────────────────────────────────────
print("\n=== Navigator form content ===")
# Find the navigatorForm section
nav_form_match = re.search(r'<form[^>]+id="navigatorForm".*?</form>', html, re.DOTALL | re.IGNORECASE)
if nav_form_match:
    nav_html = nav_form_match.group(0)
    print(f"Navigator form found, {len(nav_html)} bytes")
    print("Content preview:")
    print(nav_html[:2000])

    # Extract all links/anchors in nav form
    anchors = re.findall(r'<a[^>]+>(.*?)</a>', nav_html, re.DOTALL)
    for a in anchors:
        clean = re.sub(r'<[^>]+>', '', a).strip()
        if clean:
            print(f"  Link: {clean!r}")
else:
    print("Navigator form NOT found. Looking for browsTree...")
    # Try finding the tree by widget name
    brows_tree = re.search(r"'browsTree'.*?(?=PrimeFaces\.cw|</script>)", html, re.DOTALL)
    if brows_tree:
        print(brows_tree.group(0)[:1000])

# ── Find any onclick handlers that reference tree actions ─────────────────────
print("\n=== Tree-related onclick handlers ===")
onclicks = re.findall(r'onclick="([^"]*[Tt]ree[Nn]ode[^"]*|[^"]*browse[Tt]ree[^"]*)"', html)
for oc in onclicks[:20]:
    print(f"  {oc[:200]}")

print("\nDone.")
