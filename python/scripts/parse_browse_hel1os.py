import re
from pathlib import Path

def main():
    content = Path("e:/ARUNA/browse_hel1os.html").read_text(encoding="utf-8")
    print("Length of browse_hel1os.html:", len(content))
    
    # Let's find any occurrences of 'downloadData/hel1os' or 'downloadData'
    for m in re.finditer(r'downloadData', content, re.IGNORECASE):
        start = max(0, m.start() - 100)
        end = min(len(content), m.end() + 200)
        print(f"Match at {m.start()}:\n{content[start:end]}\n{'-'*50}")

    # Let's also print all tree links/payload items
    print("\n--- Tree Links in browse_hel1os.html ---")
    tree_section = re.search(r'id="navigatorForm.*?</form>', content, re.DOTALL)
    if tree_section:
        ts = tree_section.group(0)
        links = re.findall(r'<a[^>]+>(.*?)</a>', ts, re.DOTALL)
        for lk in links[:30]:
            clean = re.sub(r'<[^>]+>', '', lk).strip()
            if clean:
                print(f"  {clean}")

if __name__ == "__main__":
    main()
