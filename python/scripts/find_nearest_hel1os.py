import re
import requests
from pathlib import Path

def get_cookies():
    sh_path = Path("e:/ARUNA/Pradhan.sh")
    if sh_path.exists():
        content = sh_path.read_text(encoding="utf-8")
        m = re.search(r'cookies="([^"]+)"', content)
        if m:
            return m.group(1)
    return None

def main():
    cookies = get_cookies()
    if not cookies:
        print("Error: cookies not found")
        return
        
    sess = requests.Session()
    sess.headers.update({
        "Cookie": cookies,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    })
    
    url = "https://pradan1.issdc.gov.in/al1/protected/browse.xhtml?id=hel1os"
    r = sess.get(url, timeout=30)
    vs_match = re.search(r'name="javax\.faces\.ViewState"[^>]+value="([^"]+)"', r.text)
    vs = vs_match.group(1) if vs_match else ""
    
    obs_time_value = (
        "ObservationTime;TIME_DURATION;"
        "label.Product_Details.Observation_Details.Time_Coordinates.start_date_time, "
        "label.Product_Details.Observation_Details.Time_Coordinates.stop_date_time"
    )
    
    # Let's search month-by-month in 2024 to find when HEL1OS data starts being available
    months = [
        ("2024-01-01", "2024-01-31"),
        ("2024-02-01", "2024-02-29"),
        ("2024-03-01", "2024-03-31"),
        ("2024-04-01", "2024-04-30"),
        ("2024-05-01", "2024-05-31"),
        ("2024-06-01", "2024-06-30"),
        ("2024-07-01", "2024-07-31"),
    ]
    
    for start_dt, end_dt in months:
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
        r2 = sess.post(
            "https://pradan1.issdc.gov.in/al1/protected/browse.xhtml",
            data=filter_data,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Referer": url},
            timeout=30
        )
        row_count_match = re.search(r"rowCount:(\d+)", r2.text)
        rc = int(row_count_match.group(1)) if row_count_match else 0
        print(f"Month {start_dt[:7]}: {rc} files")
        if rc > 0:
            zips = re.findall(r'/al1/protected/downloadData/hel1os/[^\s"\'<>&]+\.zip(?:\?[^\s"\'<>&]*)?', r2.text)
            for z in zips[:3]:
                print(f"  {z}")

if __name__ == "__main__":
    main()
