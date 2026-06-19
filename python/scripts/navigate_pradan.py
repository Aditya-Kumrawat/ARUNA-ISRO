"""
Use the PRADAN browse.xhtml filter form to search for SoLEXS + HEL1OS 
Level-1 data on 2024-02-22 (the X6.3 flare day).

The form uses:
  - navigatorForm: selects the instrument node in the tree
  - filterForm: applies date filters
  - tableForm: shows the results with checkboxes
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
sess = requests.Session()
sess.headers["Cookie"] = COOKIES
sess.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

def get_browse(follow=True):
    r = sess.get(BASE + "/al1/protected/browse.xhtml", allow_redirects=follow, timeout=30)
    print(f"GET browse -> {r.status_code}, {len(r.text)} bytes")
    return r

def extract_vs(html):
    m = re.search(r'name="javax\.faces\.ViewState"[^>]+value="([^"]+)"', html)
    return m.group(1) if m else ""

def extract_nav_tree(html):
    """Find tree node labels and their node IDs."""
    # PrimeFaces tree nodes have data-rowkey attributes or onclick with nodeClick
    nodes = re.findall(r'onclick="[^"]*nodeClick[^"]*[,\(](\d+)[,\)][^"]*"[^>]*><[^>]+>\s*([^<]+)', html)
    # Also look for text content near tree buttons
    tree_items = re.findall(r'PF\(\'browsTree\'\)[^;]*\d+[^;]*;.*?<span[^>]*>([^<]+)</span>', html, re.DOTALL)
    return nodes

# ── Step 1: Get the browse page ───────────────────────────────────────────────
print("=== Step 1: Load browse page ===")
r0 = get_browse()
html0 = r0.text
vs = extract_vs(html0)
print(f"ViewState: {vs[:40]}")

# Extract the navigator tree structure
print("\n=== Navigator tree content ===")
# Find the tree panel
tree_section = re.search(r'id="navigatorForm.*?</form>', html0, re.DOTALL)
if tree_section:
    ts = tree_section.group(0)
    # Find all anchor/link text in the tree
    links = re.findall(r'<a[^>]+>(.*?)</a>', ts, re.DOTALL)
    for lk in links[:40]:
        clean = re.sub(r'<[^>]+>', '', lk).strip()
        if clean:
            print(f"  Tree link: {clean!r}")
    # Find onclick handlers with node IDs
    onclicks = re.findall(r'onclick="([^"]+)"', ts)
    for oc in onclicks[:20]:
        print(f"  onclick: {oc[:120]}")

# ── Step 2: Look for instrument navigation in the page ────────────────────────
print("\n=== Instrument navigation nodes ===")
# PrimeFaces tree: look for data-nodetype or node labels
spans_in_tree = re.findall(r'<span[^>]*class="[^"]*tree[^"]*"[^>]*>([^<]+)</span>', html0, re.IGNORECASE)
print(f"Tree spans: {spans_in_tree[:30]}")

# Look for any mentions of SoLEXS or level-1 in navigation-related sections
for kw in ["solexs", "SoLEXS", "hel1os", "HEL1OS", "Level-1", "level1", "Payload", "Instrument"]:
    idxs = [m.start() for m in re.finditer(kw, html0, re.IGNORECASE)]
    if idxs:
        ctx = html0[max(0, idxs[0]-150):idxs[0]+300]
        print(f"\n'{kw}' @ {idxs[0]}, ctx:\n{ctx[:400]}")

# ── Step 3: Save the full HTML for manual inspection ─────────────────────────
Path("browse_full.html").write_text(html0, encoding="utf-8")
print("\n\nFull HTML saved to browse_full.html")

# ── Step 4: Try AJAX call to the navigator tree for SoLEXS ───────────────────
# PrimeFaces Tree AJAX: POST to browse.xhtml with the tree click event
print("\n=== Step 4: Try PrimeFaces AJAX tree node click ===")
# First, look for any j_idt IDs related to the navigator tree
nav_ids = re.findall(r'id="(navigatorForm[^"]+)"', html0)
print(f"Navigator form element IDs: {nav_ids[:20]}")

# The tree widget variable name
tree_wid = re.findall(r"PrimeFaces\.widget\.Tree\s*,\s*'([^']+)'", html0)
print(f"Tree widget IDs: {tree_wid}")

# Look for the tree id
tree_ids = re.findall(r'id="(j_idt\d+:tree[^"]*|[^"]*Tree[^"]*)"', html0, re.IGNORECASE)
print(f"Tree element IDs: {tree_ids[:10]}")

print("\nDone.")
