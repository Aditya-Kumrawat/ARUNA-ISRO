import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class ManifestEntry:
    url: str
    instrument: str
    query_param: str


@dataclass
class ManifestResult:
    cookies: str
    url_prefix: str
    entries: List[ManifestEntry] = field(default_factory=list)


def _infer_instrument(relative_url: str) -> str:
    parts = relative_url.split("/")
    try:
        idx = parts.index("downloadData")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return "unknown"


class ManifestParser:
    def parse(self, path: Path) -> ManifestResult:
        text = Path(path).read_text()

        cookies_match = re.search(r'cookies="([^"]+)"', text)
        if not cookies_match:
            raise ValueError(f"No cookies found in {path}")

        prefix_match = re.search(r'urlPrefix="([^"]+)"', text)
        if not prefix_match:
            raise ValueError(f"No urlPrefix found in {path}")

        url_prefix = prefix_match.group(1).rstrip("/")
        cookies = cookies_match.group(1)

        paths_match = re.search(r'dataFilePaths=\(([^)]+)\)', text, re.DOTALL)
        if not paths_match:
            raise ValueError(f"No dataFilePaths found in {path}")

        raw_paths = re.findall(r'"([^"]+)"', paths_match.group(1))

        entries = []
        for rp in raw_paths:
            if "?" in rp:
                rel_path, param = rp.split("?", 1)
            else:
                rel_path, param = rp, "all"
            entries.append(ManifestEntry(
                url=url_prefix + rel_path,
                instrument=_infer_instrument(rel_path),
                query_param=param,
            ))

        return ManifestResult(cookies=cookies, url_prefix=url_prefix, entries=entries)
