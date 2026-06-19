"""
XL-1 SHARP Cache — file-based LRU cache for downloaded SHARP parameter rows.

Cache is keyed by (harpnum, date_str). Entries expire after cache_ttl_hours.
Designed to work without Redis; compatible with air-gapped deployments.
"""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import SHARPConfig, SHARPRecord

logger = logging.getLogger(__name__)


class SHARPCache:
    """
    Simple filesystem cache for SHARPRecord lists.

    Cache key: SHA-256 of (harpnum, date_str).
    Each entry is a pickle of (fetched_at_unix, List[SHARPRecord]).
    """

    def __init__(self, config: Optional[SHARPConfig] = None) -> None:
        self._cfg = config or SHARPConfig()
        self._dir = Path(self._cfg.cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl_sec = self._cfg.cache_ttl_hours * 3600

    def _key(self, harpnum: int, date_str: str) -> Path:
        raw = f"{harpnum}:{date_str}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return self._dir / f"sharp_{harpnum}_{digest}.pkl"

    def get(self, harpnum: int, date_str: str) -> Optional[List[SHARPRecord]]:
        """Return cached records or None if absent / expired."""
        path = self._key(harpnum, date_str)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as fh:
                fetched_at, records = pickle.load(fh)
            age = time.time() - fetched_at
            if age > self._ttl_sec:
                path.unlink(missing_ok=True)
                return None
            return records
        except Exception as exc:
            logger.debug("Cache read error (%s): %s", path, exc)
            path.unlink(missing_ok=True)
            return None

    def put(self, harpnum: int, date_str: str, records: List[SHARPRecord]) -> None:
        """Write records to cache. Silently ignores write errors."""
        path = self._key(harpnum, date_str)
        try:
            with open(path, "wb") as fh:
                pickle.dump((time.time(), records), fh, protocol=4)
        except Exception as exc:
            logger.debug("Cache write error (%s): %s", path, exc)

    def evict_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        removed = 0
        cutoff = time.time() - self._ttl_sec
        for pkl in self._dir.glob("sharp_*.pkl"):
            try:
                with open(pkl, "rb") as fh:
                    fetched_at, _ = pickle.load(fh)
                if fetched_at < cutoff:
                    pkl.unlink(missing_ok=True)
                    removed += 1
            except Exception:
                pkl.unlink(missing_ok=True)
                removed += 1
        return removed
