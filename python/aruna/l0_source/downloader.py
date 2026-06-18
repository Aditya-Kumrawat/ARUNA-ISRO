import logging
import time
from pathlib import Path
from typing import List, Optional

from .cache import CacheManager
from .manifest import ManifestEntry
from .session import PradanSession, SessionExpiredError

logger = logging.getLogger(__name__)


class Downloader:
    def __init__(self, session: PradanSession, cache: CacheManager, delay_sec: float = 2.0):
        self._session = session
        self._cache = cache
        self._delay_sec = delay_sec

    def download(self, url: str, instrument: str, date: str, filename: str) -> Optional[Path]:
        """Download a single file to cache. Returns cache path or None on failure."""
        if not self._session.is_active:
            logger.warning("Session expired — skipping %s", filename)
            return None
        try:
            resp = self._session.get(url, stream=False)
            resp.raise_for_status()
            path = self._cache.put(instrument, date, filename, resp.content)
            logger.info("Downloaded %s → %s", filename, path)
            if self._delay_sec > 0:
                time.sleep(self._delay_sec)
            return path
        except SessionExpiredError:
            logger.error("Session expired while downloading %s", url)
            return None
        except Exception as exc:
            logger.error("Download failed for %s: %s", url, exc)
            return None

    def download_manifest(self, entries: List[ManifestEntry]) -> List[Path]:
        """Download all entries from a parsed manifest, skipping cache hits."""
        results = []
        for entry in entries:
            instrument, date, filename = self._parse_entry(entry)
            if instrument is None:
                continue
            cached = self._cache.get(instrument, date, filename)
            if cached:
                results.append(cached)
                continue
            path = self.download(entry.url, instrument, date, filename)
            if path:
                results.append(path)
        return results

    @staticmethod
    def _parse_entry(entry: ManifestEntry):
        # URL: .../downloadData/{inst}/level1/{year}/{month}/{obs_folder}/{filename}
        parts = entry.url.split("/")
        try:
            level1_idx = parts.index("level1")
            year = parts[level1_idx + 1]
            month = parts[level1_idx + 2]
            filename = parts[-1].split("?")[0]
            date = f"{year}/{month}"
            return entry.instrument, date, filename
        except (ValueError, IndexError):
            logger.error("Cannot parse date/filename from URL: %s", entry.url)
            return None, None, None
