import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CacheManager:
    def __init__(self, cache_dir: Path, max_gb: float = 50.0, ttl_hours: float = 48.0):
        self._dir = Path(cache_dir)
        self._max_bytes = int(max_gb * 1024**3)
        self._ttl_sec = ttl_hours * 3600.0
        self._index_path = self._dir / ".index.json"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index: dict = self._load_index()
        self._cleanup_tmp()

    # ── index persistence ─────────────────────────────────────────────────────

    def _load_index(self) -> dict:
        tmp = self._index_path.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        if not self._index_path.exists():
            return {}
        try:
            return json.loads(self._index_path.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Cache index corrupt — starting empty")
            return {}

    def _save_index(self):
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._index, indent=2))
        tmp.replace(self._index_path)

    def _cleanup_tmp(self):
        for tmp in self._dir.rglob("*.tmp"):
            tmp.unlink(missing_ok=True)

    # ── size accounting ───────────────────────────────────────────────────────

    def total_size_bytes(self) -> int:
        return sum(e["size_bytes"] for e in self._index.values())

    def _evict_lru(self):
        while self.total_size_bytes() > self._max_bytes and self._index:
            oldest = min(self._index, key=lambda k: self._index[k]["last_accessed"])
            Path(oldest).unlink(missing_ok=True)
            del self._index[oldest]
            logger.debug("Evicted LRU entry: %s", oldest)
        self._save_index()

    def _evict_key(self, key: str):
        self._index.pop(key, None)
        self._save_index()

    # ── public API ────────────────────────────────────────────────────────────

    def put(self, instrument: str, date: str, filename: str, data: bytes) -> Path:
        """Write data to cache atomically. Returns the final cache path."""
        dest_dir = self._dir / instrument / date
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / filename
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(data)
        sha256 = hashlib.sha256(data).hexdigest()
        tmp.replace(dest)
        now = time.time()
        self._index[str(dest)] = {
            "sha256": sha256,
            "size_bytes": len(data),
            "inserted_at": now,
            "last_accessed": now,
        }
        self._evict_lru()
        return dest

    def get(self, instrument: str, date: str, filename: str) -> Optional[Path]:
        """Return verified cache path or None (miss / stale / corrupt)."""
        dest = self._dir / instrument / date / filename
        key = str(dest)
        if key not in self._index:
            return None
        entry = self._index[key]
        if time.time() - entry["inserted_at"] > self._ttl_sec:
            self._evict_key(key)
            return None
        if not dest.exists():
            self._evict_key(key)
            return None
        actual_sha = hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual_sha != entry["sha256"]:
            logger.warning("SHA-256 mismatch for %s — evicting", dest)
            dest.unlink(missing_ok=True)
            self._evict_key(key)
            return None
        entry["last_accessed"] = time.time()
        self._save_index()
        return dest
