import re
import os
import requests
from pathlib import Path

# Parse cookies from Pradhan.sh or use them directly if known
def get_cookies():
    sh_path = Path("e:/ARUNA/Pradhan.sh")
    if sh_path.exists():
        content = sh_path.read_text(encoding="utf-8")
        m = re.search(r'cookies="([^"]+)"', content)
        if m:
            return m.group(1)
    # Check alternate file
    sh_path2 = Path("e:/ARUNA/velc_2026Jun18T144203940.sh")
    if sh_path2.exists():
        content = sh_path2.read_text(encoding="utf-8")
        m = re.search(r'cookies="([^"]+)"', content)
        if m:
            return m.group(1)
    return None

def main():
    cookies = get_cookies()
    if not cookies:
        print("Error: Could not find cookies in Pradhan.sh")
        return
    
    print(f"Using cookies: {cookies[:50]}...")
    
    url = "https://pradan1.issdc.gov.in/al1/protected/downloadData/solexs/level1/2024/02/N00_0000/AL1_SLX_L1_20240222_v1.0.zip?solexs"
    
    headers = {
        "Cookie": cookies,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "*/*",
    }
    
    print(f"Downloading {url}...")
    r = requests.get(url, headers=headers, stream=True)
    print(f"Response status: {r.status_code}")
    print(f"Response headers: {r.headers}")
    
    if r.status_code == 200:
        out_dir = Path("e:/ARUNA/data")
        out_dir.mkdir(exist_ok=True)
        zip_path = out_dir / "AL1_SLX_L1_20240222_v1.0.zip"
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded ZIP to {zip_path}, size: {zip_path.stat().st_size} bytes")
        
        # Unzip
        import zipfile
        extract_dir = out_dir / "solexs_20240222"
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to {extract_dir}")
        for p in extract_dir.rglob("*"):
            if p.is_file():
                print(f"  Extracted file: {p} ({p.stat().st_size} bytes)")
    else:
        print(f"Failed to download: status {r.status_code}")
        print(r.text[:500])

if __name__ == "__main__":
    main()
