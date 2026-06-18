# L0 Source Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the L0 source layer — PRADAN session management, manifest parsing, file download, LRU cache, supplementary source health checks, and a Kafka-publishing health monitor.

**Architecture:** `PradanSession` holds the live cookie and runs a keepalive coroutine; `ManifestParser` reads PRADAN-generated `.sh` scripts to obtain download URLs; `Downloader` fetches sequentially into `CacheManager`'s SHA-256-verified LRU filesystem cache; `SourceHealthMonitor` polls every 30s and publishes `SourceStatus` to Kafka (`aruna.source.health`), falling back to a local JSON file if Kafka is down.

**Tech Stack:** Python 3.11, `requests`, `asyncio`, `hashlib`, `structlog`, `kafka-python`, `pytest`, `pytest-asyncio`

## Global Constraints

- Python 3.11 only — no `match` statements for compat, use `if/elif`
- All timestamps ISO 8601 UTC
- `schema_version = "1.0"`, `pipeline_version = "2.0.0"` on every published message
- Cookie string never logged, never written to disk, never sent to Kafka
- Inter-request delay ≥ 2s (configurable) — PRADAN rate-limit compliance
- All file writes atomic: write to `.tmp`, then `Path.replace()` — no partial writes in cache
- Tests must not make real network calls unless marked `@pytest.mark.integration`
- Config values read from `config/config.yaml` (falls back to `config.example.yaml` in tests)

---

## File Map

| File | Creates / Modifies | Responsibility |
|------|--------------------|----------------|
| `python/aruna/l0_source/models.py` | Create | `SourceStatus` dataclass |
| `python/aruna/l0_source/cache.py` | Create | LRU cache, SHA-256 integrity, eviction |
| `python/aruna/l0_source/manifest.py` | Create | PRADAN `.sh` script parser |
| `python/aruna/l0_source/session.py` | Create | Cookie manager, keepalive, SIGHUP reload |
| `python/aruna/l0_source/downloader.py` | Create | Sequential HTTPS fetcher |
| `python/aruna/l0_source/supplementary.py` | Create | 7 thin supplementary source clients |
| `python/aruna/l0_source/health.py` | Create | Asyncio health monitor, Kafka publisher |
| `python/aruna/l0_source/__init__.py` | Modify | Re-export public API |
| `python/tests/test_l0_source.py` | Modify | Replace xfail stubs with real tests |
| `config/config.example.yaml` | Modify | Add L0-specific config keys |

---

### Task 1: SourceStatus model + config keys

**Files:**
- Create: `python/aruna/l0_source/models.py`
- Modify: `config/config.example.yaml`

**Interfaces:**
- Produces: `SourceStatus` dataclass — used by Task 7 (health monitor) and tests

- [ ] **Step 1: Write the failing test**

Add to `python/tests/test_l0_source.py` (replace the first xfail stub):

```python
"""L0 Source Layer Tests — Phase 1."""
import pytest
from aruna.l0_source.models import SourceStatus


def test_source_status_valid():
    s = SourceStatus(
        source="pradan",
        status="ONLINE",
        latency_ms=42.5,
        last_success_utc="2024-02-22T10:00:00+00:00",
        fallback_active=False,
        pipeline_version="2.0.0",
    )
    assert s.schema_version == "1.0"
    assert s.source == "pradan"


def test_source_status_invalid_status():
    with pytest.raises(ValueError, match="Invalid status"):
        SourceStatus(
            source="pradan",
            status="BOGUS",
            latency_ms=0.0,
            last_success_utc="2024-02-22T10:00:00+00:00",
            fallback_active=False,
            pipeline_version="2.0.0",
        )


def test_source_status_invalid_source():
    with pytest.raises(ValueError, match="Invalid source"):
        SourceStatus(
            source="unknown_source",
            status="ONLINE",
            latency_ms=0.0,
            last_success_utc="2024-02-22T10:00:00+00:00",
            fallback_active=False,
            pipeline_version="2.0.0",
        )
```

- [ ] **Step 2: Run test to verify it fails**

```
cd E:\aruna
conda activate aruna
pytest python/tests/test_l0_source.py::test_source_status_valid -v
```
Expected: `ERROR` — `ModuleNotFoundError: No module named 'aruna.l0_source.models'`

- [ ] **Step 3: Create `python/aruna/l0_source/models.py`**

```python
from dataclasses import dataclass

VALID_STATUSES = frozenset({"ONLINE", "DEGRADED", "OFFLINE", "SESSION_EXPIRED"})
VALID_SOURCES = frozenset({
    "pradan", "noaa_swpc", "goes16_xrs", "hek",
    "jsoc_sharp", "cdaweb_ace", "cdaw_lasco", "gfz_kp",
})


@dataclass
class SourceStatus:
    source: str
    status: str
    latency_ms: float
    last_success_utc: str
    fallback_active: bool
    pipeline_version: str
    schema_version: str = "1.0"

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {self.status!r}. Must be one of {VALID_STATUSES}")
        if self.source not in VALID_SOURCES:
            raise ValueError(f"Invalid source: {self.source!r}. Must be one of {VALID_SOURCES}")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest python/tests/test_l0_source.py::test_source_status_valid python/tests/test_l0_source.py::test_source_status_invalid_status python/tests/test_l0_source.py::test_source_status_invalid_source -v
```
Expected: 3 PASSED

- [ ] **Step 5: Add L0 config keys to `config/config.example.yaml`**

Find the `data:` section and add these keys (they are already partially present — merge carefully):

```yaml
data:
  # PRADAN portal
  pradan_url: "https://pradan1.issdc.gov.in"
  pradan_keepalive_interval_min: 10
  pradan_keepalive_max_cycles: 144
  pradan_offline_fallback_sec: 300
  manifest_watch_dir: "data/manifests/"
  download_delay_sec: 2.0
  # Cache
  cache_dir: "data/cache/"
  cache_ttl_hours: 48
  cache_max_gb: 50
```

- [ ] **Step 6: Commit**

```
git add python/aruna/l0_source/models.py python/tests/test_l0_source.py config/config.example.yaml
git commit -m "feat(l0): SourceStatus model + L0 config keys"
```

---

### Task 2: CacheManager

**Files:**
- Create: `python/aruna/l0_source/cache.py`
- Modify: `python/tests/test_l0_source.py`

**Interfaces:**
- Produces:
  - `CacheManager(cache_dir, max_gb, ttl_hours)`
  - `.put(instrument: str, date: str, filename: str, data: bytes) -> Path`
  - `.get(instrument: str, date: str, filename: str) -> Optional[Path]`
  - `.total_size_bytes() -> int`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l0_source.py`:

```python
import hashlib
import time
from pathlib import Path
from aruna.l0_source.cache import CacheManager


def test_cache_put_get_roundtrip(tmp_path):
    cm = CacheManager(tmp_path / "cache", max_gb=1.0, ttl_hours=48.0)
    data = b"FITS_DATA_MOCK_BYTES_12345"
    cm.put("solexs", "2024/02", "test.fits", data)
    result = cm.get("solexs", "2024/02", "test.fits")
    assert result is not None
    assert result.read_bytes() == data


def test_cache_miss_returns_none(tmp_path):
    cm = CacheManager(tmp_path / "cache", max_gb=1.0, ttl_hours=48.0)
    assert cm.get("solexs", "2024/02", "missing.fits") is None


def test_cache_sha256_mismatch_evicts(tmp_path):
    cm = CacheManager(tmp_path / "cache", max_gb=1.0, ttl_hours=48.0)
    data = b"original data"
    path = cm.put("solexs", "2024/02", "corrupt.fits", data)
    # Corrupt the file on disk
    path.write_bytes(b"corrupted!")
    result = cm.get("solexs", "2024/02", "corrupt.fits")
    assert result is None
    # Index entry removed
    assert cm.get("solexs", "2024/02", "corrupt.fits") is None


def test_cache_ttl_expiry(tmp_path):
    cm = CacheManager(tmp_path / "cache", max_gb=1.0, ttl_hours=0.0001)  # ~0.36 seconds
    cm.put("solexs", "2024/02", "old.fits", b"data")
    time.sleep(0.5)
    assert cm.get("solexs", "2024/02", "old.fits") is None


def test_cache_evicts_at_size_cap(tmp_path):
    # Cap at ~50 bytes — enough for one 20-byte file but not two
    cm = CacheManager(tmp_path / "cache", max_gb=50 / 1024**3, ttl_hours=48.0)
    cm.put("solexs", "2024/02", "first.fits", b"A" * 20)
    cm.put("solexs", "2024/02", "second.fits", b"B" * 20)
    # Total would be 40 bytes; at 50B cap both fit; use smaller cap
    cm2 = CacheManager(tmp_path / "cache2", max_gb=25 / 1024**3, ttl_hours=48.0)
    cm2.put("solexs", "2024/02", "first.fits", b"A" * 20)
    cm2.put("solexs", "2024/02", "second.fits", b"B" * 20)
    # After eviction, total <= cap
    assert cm2.total_size_bytes() <= int(25)


def test_cache_no_tmp_after_put(tmp_path):
    cm = CacheManager(tmp_path / "cache", max_gb=1.0, ttl_hours=48.0)
    cm.put("solexs", "2024/02", "test.fits", b"data")
    tmps = list((tmp_path / "cache").rglob("*.tmp"))
    assert tmps == []


def test_cache_index_atomic_write_survives_if_tmp_exists(tmp_path):
    """Index .tmp left from prior crash is cleaned and index loads correctly."""
    cache_dir = tmp_path / "cache"
    cm = CacheManager(cache_dir, max_gb=1.0, ttl_hours=48.0)
    cm.put("solexs", "2024/02", "test.fits", b"data")
    # Simulate crash: leave a stale .tmp index
    index_tmp = cache_dir / ".index.tmp"
    index_tmp.write_text("CORRUPT JSON{{{")
    # Reload cache — should not crash, should still find existing entry
    cm2 = CacheManager(cache_dir, max_gb=1.0, ttl_hours=48.0)
    result = cm2.get("solexs", "2024/02", "test.fits")
    assert result is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest python/tests/test_l0_source.py::test_cache_put_get_roundtrip -v
```
Expected: `ERROR — ModuleNotFoundError: No module named 'aruna.l0_source.cache'`

- [ ] **Step 3: Create `python/aruna/l0_source/cache.py`**

```python
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

    # ── index persistence ────────────────────────────────────────────────

    def _load_index(self) -> dict:
        # Ignore any stale .tmp — if it exists, it is from a prior crash
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

    # ── integrity ────────────────────────────────────────────────────────

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    # ── size accounting ──────────────────────────────────────────────────

    def total_size_bytes(self) -> int:
        return sum(e["size_bytes"] for e in self._index.values())

    def _evict_lru(self):
        while self.total_size_bytes() > self._max_bytes and self._index:
            oldest = min(self._index, key=lambda k: self._index[k]["last_accessed"])
            Path(oldest).unlink(missing_ok=True)
            del self._index[oldest]
            logger.debug(f"Evicted LRU entry: {oldest}")
        self._save_index()

    # ── public API ───────────────────────────────────────────────────────

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
        if self._sha256(dest) != entry["sha256"]:
            logger.warning(f"SHA-256 mismatch for {dest} — evicting")
            dest.unlink(missing_ok=True)
            self._evict_key(key)
            return None
        entry["last_accessed"] = time.time()
        self._save_index()
        return dest

    def _evict_key(self, key: str):
        self._index.pop(key, None)
        self._save_index()
```

- [ ] **Step 4: Run all cache tests**

```
pytest python/tests/test_l0_source.py -k "cache" -v
```
Expected: 7 PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l0_source/cache.py python/tests/test_l0_source.py
git commit -m "feat(l0): CacheManager — LRU, SHA-256, atomic writes"
```

---

### Task 3: ManifestParser

**Files:**
- Create: `python/aruna/l0_source/manifest.py`
- Modify: `python/tests/test_l0_source.py`

**Interfaces:**
- Produces:
  - `ManifestEntry(url: str, instrument: str, query_param: str)`
  - `ManifestResult(cookies: str, url_prefix: str, entries: list[ManifestEntry])`
  - `ManifestParser().parse(path: Path) -> ManifestResult`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l0_source.py`:

```python
import textwrap
from aruna.l0_source.manifest import ManifestParser, ManifestResult


SAMPLE_SCRIPT = textwrap.dedent("""\
    #! /bin/bash
    cookies="FGTServer=ABC123;JSESSIONID=b23e5074;OAuth_Token_Request_State=5431e0d8;"
    urlPrefix="https://pradan1.issdc.gov.in"
    proxyOptions=""
    counter=144;while [ $counter -gt 0 ]; do sleep 10m; wget $proxyOptions -N \\
      --content-disposition --tries=1 --no-cookies \\
      --header "Cookie: $cookies" $urlPrefix"/al1/protected/payload.xhtml"; \\
      counter=$(($counter-1)); done &
    bdpid=$!
    dataFilePaths=("/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135938_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/solexs/level1/2026/03/T26_0100/SL1_T26_0100_20260314_135938_lev1.fits?all" )
    i=0;
    for file in ${dataFilePaths[@]}; do
        echo $file;
        i=$(($i+1));
        wget $proxyOptions -x --max-redirect=0 --content-disposition --tries=1 \\
          --no-cookies --header "Cookie: $cookies" $urlPrefix$file;
    done
    kill -9 $bdpid
""")


def test_manifest_parser_extracts_cookies(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    assert "FGTServer=ABC123" in result.cookies
    assert "JSESSIONID=b23e5074" in result.cookies


def test_manifest_parser_extracts_url_prefix(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    assert result.url_prefix == "https://pradan1.issdc.gov.in"


def test_manifest_parser_extracts_entries(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    assert len(result.entries) == 2


def test_manifest_parser_infers_instrument(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    instruments = {e.instrument for e in result.entries}
    assert "velc" in instruments
    assert "solexs" in instruments


def test_manifest_parser_extracts_query_params(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    params = {e.query_param for e in result.entries}
    assert "velc" in params
    assert "all" in params


def test_manifest_parser_full_urls(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    for entry in result.entries:
        assert entry.url.startswith("https://pradan1.issdc.gov.in")
        assert "/level1/" in entry.url


def test_manifest_parser_real_pradhan_sh():
    """Parse the actual Pradhan.sh from repo root."""
    from pathlib import Path
    pradhan = Path("Pradhan.sh")
    if not pradhan.exists():
        pytest.skip("Pradhan.sh not present")
    result = ManifestParser().parse(pradhan)
    assert result.cookies
    assert result.url_prefix == "https://pradan1.issdc.gov.in"
    assert len(result.entries) > 0


def test_manifest_parser_missing_cookies_raises(tmp_path):
    bad_script = tmp_path / "bad.sh"
    bad_script.write_text('urlPrefix="https://example.com"\ndataFilePaths=("/foo")\n')
    with pytest.raises(ValueError, match="No cookies"):
        ManifestParser().parse(bad_script)
```

- [ ] **Step 2: Run to verify failure**

```
pytest python/tests/test_l0_source.py::test_manifest_parser_extracts_cookies -v
```
Expected: `ERROR — ModuleNotFoundError`

- [ ] **Step 3: Create `python/aruna/l0_source/manifest.py`**

```python
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
```

- [ ] **Step 4: Run all manifest tests**

```
pytest python/tests/test_l0_source.py -k "manifest" -v
```
Expected: all PASSED (real Pradhan.sh test may skip if file not in cwd)

- [ ] **Step 5: Commit**

```
git add python/aruna/l0_source/manifest.py python/tests/test_l0_source.py
git commit -m "feat(l0): ManifestParser — PRADAN .sh script parsing"
```

---

### Task 4: PradanSession

**Files:**
- Create: `python/aruna/l0_source/session.py`
- Modify: `python/tests/test_l0_source.py`

**Interfaces:**
- Produces:
  - `SessionExpiredError(Exception)`
  - `SessionState(Enum)` — `ACTIVE`, `EXPIRED`
  - `PradanSession(base_url, keepalive_interval_min, max_cycles)`
    - `.state -> SessionState`
    - `.is_active -> bool`
    - `.get(url, **kwargs) -> requests.Response` — raises `SessionExpiredError` on 302
    - `.reload()` — re-reads `PRADAN_COOKIES` from env
    - `.set_expired_callback(fn)` — called when session expires
    - `async .run_keepalive()` — asyncio coroutine

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l0_source.py`:

```python
import os
import signal
from unittest.mock import MagicMock, patch
from aruna.l0_source.session import PradanSession, SessionExpiredError, SessionState


def test_session_active_when_cookies_set(monkeypatch):
    monkeypatch.setenv("PRADAN_COOKIES", "JSESSIONID=test123;")
    s = PradanSession("https://pradan1.issdc.gov.in")
    assert s.is_active
    assert s.state == SessionState.ACTIVE


def test_session_expired_when_no_cookies(monkeypatch):
    monkeypatch.delenv("PRADAN_COOKIES", raising=False)
    s = PradanSession("https://pradan1.issdc.gov.in")
    assert not s.is_active
    assert s.state == SessionState.EXPIRED


def test_session_302_raises_session_expired(monkeypatch):
    monkeypatch.setenv("PRADAN_COOKIES", "JSESSIONID=test123;")
    s = PradanSession("https://pradan1.issdc.gov.in")
    mock_resp = MagicMock()
    mock_resp.status_code = 302
    with patch.object(s._http, "get", return_value=mock_resp):
        with pytest.raises(SessionExpiredError):
            s.get("https://pradan1.issdc.gov.in/al1/protected/downloadData/solexs/test.fits")
    assert s.state == SessionState.EXPIRED


def test_session_302_triggers_callback(monkeypatch):
    monkeypatch.setenv("PRADAN_COOKIES", "JSESSIONID=test123;")
    s = PradanSession("https://pradan1.issdc.gov.in")
    called = []
    s.set_expired_callback(lambda: called.append(True))
    mock_resp = MagicMock()
    mock_resp.status_code = 302
    with patch.object(s._http, "get", return_value=mock_resp):
        with pytest.raises(SessionExpiredError):
            s.get("https://pradan1.issdc.gov.in/test")
    assert called == [True]


def test_session_reload_restores_active(monkeypatch):
    monkeypatch.delenv("PRADAN_COOKIES", raising=False)
    s = PradanSession("https://pradan1.issdc.gov.in")
    assert s.state == SessionState.EXPIRED
    monkeypatch.setenv("PRADAN_COOKIES", "JSESSIONID=new123;")
    s.reload()
    assert s.state == SessionState.ACTIVE


def test_session_get_raises_when_expired(monkeypatch):
    monkeypatch.delenv("PRADAN_COOKIES", raising=False)
    s = PradanSession("https://pradan1.issdc.gov.in")
    with pytest.raises(SessionExpiredError):
        s.get("https://pradan1.issdc.gov.in/anything")


def test_session_keepalive_counter_exhaustion(monkeypatch):
    import asyncio
    monkeypatch.setenv("PRADAN_COOKIES", "JSESSIONID=test123;")
    s = PradanSession("https://pradan1.issdc.gov.in", keepalive_interval_min=0.001, max_cycles=1)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch.object(s._http, "get", return_value=mock_resp):
        asyncio.run(s._do_keepalive_tick())  # first tick — cycles to 1
        asyncio.run(s._do_keepalive_tick())  # second tick — max reached
    assert s.state == SessionState.EXPIRED
```

- [ ] **Step 2: Run to verify failure**

```
pytest python/tests/test_l0_source.py::test_session_active_when_cookies_set -v
```
Expected: `ERROR — ModuleNotFoundError`

- [ ] **Step 3: Create `python/aruna/l0_source/session.py`**

```python
import asyncio
import logging
import os
import signal
from enum import Enum
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

KEEPALIVE_PATH = "/al1/protected/payload.xhtml"


class SessionState(Enum):
    ACTIVE = "ACTIVE"
    EXPIRED = "EXPIRED"


class SessionExpiredError(Exception):
    pass


class PradanSession:
    def __init__(
        self,
        base_url: str,
        keepalive_interval_min: float = 10.0,
        max_cycles: int = 144,
    ):
        self._base_url = base_url.rstrip("/")
        self._keepalive_interval_sec = keepalive_interval_min * 60.0
        self._max_cycles = max_cycles
        self._state = SessionState.EXPIRED
        self._cookies: str = ""
        self._cycle_count: int = 0
        self._http = requests.Session()
        self._on_expired: Optional[Callable] = None
        self._reload()
        try:
            signal.signal(signal.SIGHUP, lambda sig, frame: self.reload())
        except (OSError, ValueError):
            pass  # SIGHUP not available on Windows in some contexts

    # ── public API ───────────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state == SessionState.ACTIVE

    def set_expired_callback(self, callback: Callable):
        self._on_expired = callback

    def reload(self):
        """Re-read PRADAN_COOKIES from env. Called on SIGHUP."""
        cookies = os.environ.get("PRADAN_COOKIES", "").strip()
        if cookies:
            self._cookies = cookies
            self._state = SessionState.ACTIVE
            self._cycle_count = 0
            self._http = requests.Session()
            logger.info("PRADAN session reloaded from PRADAN_COOKIES")
        else:
            self._state = SessionState.EXPIRED
            logger.warning("PRADAN_COOKIES not set — session EXPIRED")

    def get(self, url: str, **kwargs) -> requests.Response:
        if not self.is_active:
            raise SessionExpiredError("Session is EXPIRED — reload cookies")
        kwargs.setdefault("allow_redirects", False)
        kwargs.setdefault("timeout", 120)
        resp = self._http.get(url, headers={"Cookie": self._cookies}, **kwargs)
        if resp.status_code == 302:
            self._mark_expired()
            raise SessionExpiredError(f"HTTP 302 on {url} — session expired")
        return resp

    # ── keepalive ────────────────────────────────────────────────────────

    async def run_keepalive(self):
        while True:
            await asyncio.sleep(self._keepalive_interval_sec)
            await self._do_keepalive_tick()

    async def _do_keepalive_tick(self):
        if not self.is_active:
            return
        if self._cycle_count >= self._max_cycles:
            logger.warning("Keepalive max cycles (%d) reached — proactive EXPIRED", self._max_cycles)
            self._mark_expired()
            return
        try:
            url = self._base_url + KEEPALIVE_PATH
            resp = self._http.get(
                url, headers={"Cookie": self._cookies},
                allow_redirects=False, timeout=30,
            )
            if resp.status_code == 302:
                self._mark_expired()
            else:
                self._cycle_count += 1
                logger.debug("Keepalive cycle %d/%d", self._cycle_count, self._max_cycles)
        except Exception as exc:
            logger.warning("Keepalive request failed: %s", exc)

    # ── internal ─────────────────────────────────────────────────────────

    def _reload(self):
        self.reload()

    def _mark_expired(self):
        self._state = SessionState.EXPIRED
        if self._on_expired:
            self._on_expired()
```

- [ ] **Step 4: Run all session tests**

```
pytest python/tests/test_l0_source.py -k "session" -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l0_source/session.py python/tests/test_l0_source.py
git commit -m "feat(l0): PradanSession — cookie manager, keepalive, SIGHUP reload"
```

---

### Task 5: Downloader

**Files:**
- Create: `python/aruna/l0_source/downloader.py`
- Modify: `python/tests/test_l0_source.py`

**Interfaces:**
- Consumes: `PradanSession` (Task 4), `CacheManager` (Task 2), `ManifestEntry` (Task 3)
- Produces:
  - `Downloader(session, cache, delay_sec)`
  - `.download(url, instrument, date, filename) -> Optional[Path]`
  - `.download_manifest(entries: list[ManifestEntry]) -> list[Path]`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l0_source.py`:

```python
from unittest.mock import patch, MagicMock
from aruna.l0_source.downloader import Downloader
from aruna.l0_source.manifest import ManifestEntry


def _make_downloader(tmp_path, monkeypatch, cookies="JSESSIONID=test;"):
    from aruna.l0_source.cache import CacheManager
    from aruna.l0_source.session import PradanSession
    monkeypatch.setenv("PRADAN_COOKIES", cookies)
    session = PradanSession("https://pradan1.issdc.gov.in")
    cache = CacheManager(tmp_path / "cache", max_gb=1.0, ttl_hours=48.0)
    return Downloader(session=session, cache=cache, delay_sec=0.0)


def test_downloader_stores_file_in_cache(tmp_path, monkeypatch):
    dl = _make_downloader(tmp_path, monkeypatch)
    fake_fits = b"SIMPLE  =   T / fits file\n" + b"\x00" * 100
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = fake_fits
    with patch.object(dl._session._http, "get", return_value=mock_resp):
        result = dl.download(
            "https://pradan1.issdc.gov.in/al1/protected/downloadData/solexs/level1/2024/02/T001/test.fits",
            "solexs", "2024/02", "test.fits",
        )
    assert result is not None
    assert result.read_bytes() == fake_fits


def test_downloader_302_returns_none(tmp_path, monkeypatch):
    dl = _make_downloader(tmp_path, monkeypatch)
    mock_resp = MagicMock()
    mock_resp.status_code = 302
    with patch.object(dl._session._http, "get", return_value=mock_resp):
        result = dl.download(
            "https://pradan1.issdc.gov.in/protected/test.fits",
            "solexs", "2024/02", "test.fits",
        )
    assert result is None


def test_downloader_uses_cache_hit(tmp_path, monkeypatch):
    dl = _make_downloader(tmp_path, monkeypatch)
    # Pre-populate cache
    dl._cache.put("solexs", "2024/02", "cached.fits", b"cached data")
    # If cache is hit, no HTTP call should be made
    entry = ManifestEntry(
        url="https://pradan1.issdc.gov.in/al1/protected/downloadData/solexs/level1/2024/02/T001/cached.fits",
        instrument="solexs",
        query_param="all",
    )
    call_count = []
    original_get = dl._session.get
    def counting_get(*a, **kw):
        call_count.append(1)
        return original_get(*a, **kw)
    with patch.object(dl._session, "get", side_effect=counting_get):
        results = dl.download_manifest([entry])
    assert call_count == []  # cache hit — no HTTP
    assert len(results) == 1


def test_downloader_no_tmp_in_cache(tmp_path, monkeypatch):
    dl = _make_downloader(tmp_path, monkeypatch)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"fits data"
    with patch.object(dl._session._http, "get", return_value=mock_resp):
        dl.download(
            "https://pradan1.issdc.gov.in/test.fits",
            "solexs", "2024/02", "notemps.fits",
        )
    tmps = list((tmp_path / "cache").rglob("*.tmp"))
    assert tmps == []
```

- [ ] **Step 2: Run to verify failure**

```
pytest python/tests/test_l0_source.py::test_downloader_stores_file_in_cache -v
```
Expected: `ERROR — ModuleNotFoundError`

- [ ] **Step 3: Create `python/aruna/l0_source/downloader.py`**

```python
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
```

- [ ] **Step 4: Run all downloader tests**

```
pytest python/tests/test_l0_source.py -k "downloader" -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l0_source/downloader.py python/tests/test_l0_source.py
git commit -m "feat(l0): Downloader — sequential fetch, cache-first, rate-limit safe"
```

---

### Task 6: SupplementaryFetcher

**Files:**
- Create: `python/aruna/l0_source/supplementary.py`
- Modify: `python/tests/test_l0_source.py`

**Interfaces:**
- Produces:
  - `ALL_SUPPLEMENTARY_CLIENTS: dict[str, SupplementaryClient]`
  - Each client: `.ping() -> tuple[str, float]` — `(status_str, latency_ms)`
  - status_str ∈ `{"ONLINE", "DEGRADED", "OFFLINE"}`

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l0_source.py`:

```python
from unittest.mock import patch, MagicMock
from aruna.l0_source.supplementary import ALL_SUPPLEMENTARY_CLIENTS


EXPECTED_SOURCES = {
    "noaa_swpc", "goes16_xrs", "hek",
    "jsoc_sharp", "cdaweb_ace", "cdaw_lasco", "gfz_kp",
}


def test_supplementary_all_seven_clients_present():
    assert set(ALL_SUPPLEMENTARY_CLIENTS.keys()) == EXPECTED_SOURCES


def test_supplementary_ping_returns_online_on_200():
    for name, client in ALL_SUPPLEMENTARY_CLIENTS.items():
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("aruna.l0_source.supplementary.requests.get", return_value=mock_resp):
            status, latency_ms = client.ping()
        assert status == "ONLINE", f"{name} should be ONLINE on 200"
        assert latency_ms >= 0


def test_supplementary_ping_returns_offline_on_exception():
    for name, client in ALL_SUPPLEMENTARY_CLIENTS.items():
        with patch("aruna.l0_source.supplementary.requests.get", side_effect=Exception("timeout")):
            status, latency_ms = client.ping()
        assert status == "OFFLINE", f"{name} should be OFFLINE on exception"


def test_supplementary_ping_returns_degraded_on_5xx():
    for name, client in ALL_SUPPLEMENTARY_CLIENTS.items():
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        with patch("aruna.l0_source.supplementary.requests.get", return_value=mock_resp):
            status, latency_ms = client.ping()
        assert status == "DEGRADED", f"{name} should be DEGRADED on 503"
```

- [ ] **Step 2: Run to verify failure**

```
pytest python/tests/test_l0_source.py::test_supplementary_all_seven_clients_present -v
```
Expected: `ERROR — ModuleNotFoundError`

- [ ] **Step 3: Create `python/aruna/l0_source/supplementary.py`**

```python
import time
from typing import Tuple

import requests

_TIMEOUT = 10


def _ping(url: str) -> Tuple[str, float]:
    t0 = time.monotonic()
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        latency_ms = (time.monotonic() - t0) * 1000
        if resp.status_code < 400:
            return "ONLINE", latency_ms
        return "DEGRADED", latency_ms
    except Exception:
        latency_ms = (time.monotonic() - t0) * 1000
        return "OFFLINE", latency_ms


class _SupplementaryClient:
    _url: str

    def ping(self) -> Tuple[str, float]:
        return _ping(self._url)


class _NoaaSwpc(_SupplementaryClient):
    _url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json"


class _Goes16Xrs(_SupplementaryClient):
    _url = "https://noaa-goes16.s3.amazonaws.com"


class _Hek(_SupplementaryClient):
    _url = "https://www.heliophysicsapi.com/hek/v0/event/search?event_type=fl&result_limit=1&cosec=2"


class _JsocSharp(_SupplementaryClient):
    _url = "http://jsoc.stanford.edu/cgi-bin/ajax/jsoc_info?ds=hmi.sharp_720s&op=series_struct"


class _CdawebAce(_SupplementaryClient):
    _url = "https://cdaweb.gsfc.nasa.gov/WS/cdasr/1/dataviews/sp_phys/"


class _CdawLasco(_SupplementaryClient):
    _url = "https://cdaw.gsfc.nasa.gov/CME_list/"


class _GfzKp(_SupplementaryClient):
    _url = "https://kp.gfz-potsdam.de/app/json/?start=2024-01-01T00:00Z&end=2024-01-02T00:00Z&index=Kp"


ALL_SUPPLEMENTARY_CLIENTS: dict = {
    "noaa_swpc": _NoaaSwpc(),
    "goes16_xrs": _Goes16Xrs(),
    "hek": _Hek(),
    "jsoc_sharp": _JsocSharp(),
    "cdaweb_ace": _CdawebAce(),
    "cdaw_lasco": _CdawLasco(),
    "gfz_kp": _GfzKp(),
}
```

- [ ] **Step 4: Run all supplementary tests**

```
pytest python/tests/test_l0_source.py -k "supplementary" -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l0_source/supplementary.py python/tests/test_l0_source.py
git commit -m "feat(l0): SupplementaryFetcher — 7 source clients with ping()"
```

---

### Task 7: SourceHealthMonitor

**Files:**
- Create: `python/aruna/l0_source/health.py`
- Modify: `python/tests/test_l0_source.py`

**Interfaces:**
- Consumes: `PradanSession` (Task 4), `SourceStatus` (Task 1), `ALL_SUPPLEMENTARY_CLIENTS` (Task 6)
- Produces:
  - `SourceHealthMonitor(session, kafka_producer, kafka_topic, poll_interval_sec, health_file, offline_fallback_sec)`
  - `async .run()` — continuous poll loop
  - `async ._poll()` — single poll cycle (testable)

- [ ] **Step 1: Write the failing tests**

Append to `python/tests/test_l0_source.py`:

```python
import asyncio
import json
from unittest.mock import MagicMock, patch, AsyncMock
from aruna.l0_source.health import SourceHealthMonitor


def _make_monitor(tmp_path, monkeypatch, cookies="JSESSIONID=test;"):
    from aruna.l0_source.session import PradanSession
    monkeypatch.setenv("PRADAN_COOKIES", cookies)
    session = PradanSession("https://pradan1.issdc.gov.in")
    producer = MagicMock()
    monitor = SourceHealthMonitor(
        session=session,
        kafka_producer=producer,
        kafka_topic="aruna.source.health",
        poll_interval_sec=0.01,
        health_file=tmp_path / "data" / "health.json",
        offline_fallback_sec=300.0,
    )
    return monitor, producer


def test_health_monitor_publishes_pradan_status(tmp_path, monkeypatch):
    monitor, producer = _make_monitor(tmp_path, monkeypatch)
    with patch("aruna.l0_source.supplementary.requests.get", return_value=MagicMock(status_code=200)):
        asyncio.run(monitor._poll())
    assert producer.send.called
    call_args = producer.send.call_args_list[0]
    assert call_args[0][0] == "aruna.source.health"
    payload = call_args[1]["value"]
    assert payload["source"] == "pradan"
    assert payload["status"] == "ONLINE"


def test_health_monitor_session_expired_status(tmp_path, monkeypatch):
    monkeypatch.delenv("PRADAN_COOKIES", raising=False)
    from aruna.l0_source.session import PradanSession
    session = PradanSession("https://pradan1.issdc.gov.in")
    producer = MagicMock()
    monitor = SourceHealthMonitor(
        session=session,
        kafka_producer=producer,
        kafka_topic="aruna.source.health",
        poll_interval_sec=0.01,
        health_file=tmp_path / "health.json",
        offline_fallback_sec=300.0,
    )
    with patch("aruna.l0_source.supplementary.requests.get", return_value=MagicMock(status_code=200)):
        asyncio.run(monitor._poll())
    pradan_call = producer.send.call_args_list[0]
    payload = pradan_call[1]["value"]
    assert payload["status"] == "SESSION_EXPIRED"


def test_health_monitor_writes_to_file_when_kafka_down(tmp_path, monkeypatch):
    monitor, producer = _make_monitor(tmp_path, monkeypatch)
    producer.send.side_effect = Exception("Kafka connection refused")
    with patch("aruna.l0_source.supplementary.requests.get", return_value=MagicMock(status_code=200)):
        asyncio.run(monitor._poll())
    health_file = tmp_path / "data" / "health.json"
    assert health_file.exists()
    records = json.loads(health_file.read_text())
    assert isinstance(records, list)
    assert len(records) > 0
    assert records[0]["source"] == "pradan"


def test_health_monitor_schema_version_in_every_publish(tmp_path, monkeypatch):
    monitor, producer = _make_monitor(tmp_path, monkeypatch)
    with patch("aruna.l0_source.supplementary.requests.get", return_value=MagicMock(status_code=200)):
        asyncio.run(monitor._poll())
    for call in producer.send.call_args_list:
        assert call[1]["value"]["schema_version"] == "1.0"
        assert call[1]["value"]["pipeline_version"] == "2.0.0"


def test_health_monitor_fallback_activates_after_5min(tmp_path, monkeypatch):
    monkeypatch.delenv("PRADAN_COOKIES", raising=False)
    from aruna.l0_source.session import PradanSession
    session = PradanSession("https://pradan1.issdc.gov.in")
    producer = MagicMock()
    monitor = SourceHealthMonitor(
        session=session,
        kafka_producer=producer,
        kafka_topic="aruna.source.health",
        poll_interval_sec=0.01,
        health_file=tmp_path / "health.json",
        offline_fallback_sec=0.0,  # threshold=0 so first poll triggers fallback
    )
    with patch("aruna.l0_source.supplementary.requests.get", return_value=MagicMock(status_code=200)):
        asyncio.run(monitor._poll())
        asyncio.run(monitor._poll())  # second poll crosses 0s threshold
    # Find the pradan call where fallback_active=True
    pradan_calls = [
        c[1]["value"] for c in producer.send.call_args_list
        if c[1]["value"]["source"] == "pradan"
    ]
    assert any(c["fallback_active"] for c in pradan_calls)
```

- [ ] **Step 2: Run to verify failure**

```
pytest python/tests/test_l0_source.py::test_health_monitor_publishes_pradan_status -v
```
Expected: `ERROR — ModuleNotFoundError`

- [ ] **Step 3: Create `python/aruna/l0_source/health.py`**

```python
import asyncio
import dataclasses
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .models import SourceStatus
from .session import PradanSession, SessionState
from .supplementary import ALL_SUPPLEMENTARY_CLIENTS

logger = logging.getLogger(__name__)

_PIPELINE_VERSION = "2.0.0"


class SourceHealthMonitor:
    def __init__(
        self,
        session: PradanSession,
        kafka_producer=None,
        kafka_topic: str = "aruna.source.health",
        poll_interval_sec: float = 30.0,
        health_file: Path = Path("data/health.json"),
        offline_fallback_sec: float = 300.0,
    ):
        self._session = session
        self._producer = kafka_producer
        self._topic = kafka_topic
        self._interval = poll_interval_sec
        self._health_file = Path(health_file)
        self._offline_fallback_sec = offline_fallback_sec
        self._pradan_offline_since: Optional[float] = None
        self._fallback_active = False
        self._pending: List[SourceStatus] = []

    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _check_pradan(self) -> SourceStatus:
        if self._session.state == SessionState.EXPIRED:
            status_str = "SESSION_EXPIRED"
        else:
            status_str = "ONLINE"

        if status_str in ("SESSION_EXPIRED", "OFFLINE"):
            if self._pradan_offline_since is None:
                self._pradan_offline_since = time.monotonic()
            elapsed = time.monotonic() - self._pradan_offline_since
            if elapsed >= self._offline_fallback_sec:
                self._fallback_active = True
                logger.warning("PRADAN offline %.0fs — activating NOAA fallback", elapsed)
        else:
            self._pradan_offline_since = None
            self._fallback_active = False

        return SourceStatus(
            source="pradan",
            status=status_str,
            latency_ms=0.0,
            last_success_utc=self._now_utc(),
            fallback_active=self._fallback_active,
            pipeline_version=_PIPELINE_VERSION,
        )

    def _publish(self, status: SourceStatus):
        payload = dataclasses.asdict(status)
        if self._producer is not None:
            try:
                self._producer.send(self._topic, value=payload)
                for pending in self._pending:
                    self._producer.send(self._topic, value=dataclasses.asdict(pending))
                self._pending.clear()
                return
            except Exception as exc:
                logger.warning("Kafka publish failed: %s — buffering to %s", exc, self._health_file)

        self._pending.append(status)
        self._health_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._health_file.with_suffix(".tmp")
        records = [dataclasses.asdict(s) for s in self._pending]
        tmp.write_text(json.dumps(records, indent=2))
        tmp.replace(self._health_file)

    async def _poll(self):
        self._publish(self._check_pradan())
        for name, client in ALL_SUPPLEMENTARY_CLIENTS.items():
            status_str, latency_ms = client.ping()
            self._publish(SourceStatus(
                source=name,
                status=status_str,
                latency_ms=latency_ms,
                last_success_utc=self._now_utc(),
                fallback_active=False,
                pipeline_version=_PIPELINE_VERSION,
            ))

    async def run(self):
        while True:
            await self._poll()
            await asyncio.sleep(self._interval)
```

- [ ] **Step 4: Run all health monitor tests**

```
pytest python/tests/test_l0_source.py -k "health_monitor" -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```
git add python/aruna/l0_source/health.py python/tests/test_l0_source.py
git commit -m "feat(l0): SourceHealthMonitor — Kafka publish, health.json fallback"
```

---

### Task 8: Public `__init__`, full test suite, acceptance gate

**Files:**
- Modify: `python/aruna/l0_source/__init__.py`
- Modify: `python/tests/test_l0_source.py`

**Interfaces:**
- Consumes: all Task 1–7 exports
- Produces: clean public API at `aruna.l0_source.*`

- [ ] **Step 1: Write acceptance tests (final gate)**

Append to `python/tests/test_l0_source.py`:

```python
# ── Acceptance gate (PRD §4.6 + brainstorm) ──────────────────────────────────


def test_acceptance_cache_directory_writable(tmp_path):
    from aruna.l0_source.cache import CacheManager
    cm = CacheManager(tmp_path / "cache")
    cm.put("solexs", "2024/02", "test.fits", b"data")
    assert cm.get("solexs", "2024/02", "test.fits") is not None


def test_acceptance_manifest_parses_real_pradhan_sh():
    """PRADAN connection established and authenticated (manifest + cookie extraction)."""
    from pathlib import Path
    from aruna.l0_source.manifest import ManifestParser
    pradhan = Path("Pradhan.sh")
    if not pradhan.exists():
        pytest.skip("Pradhan.sh not present")
    result = ManifestParser().parse(pradhan)
    assert result.cookies, "Cookies must not be empty"
    assert len(result.entries) > 0, "Must have at least one download entry"


def test_acceptance_source_status_schema():
    """Every SourceStatus has required schema_version and pipeline_version."""
    from aruna.l0_source.models import SourceStatus
    s = SourceStatus("pradan", "ONLINE", 10.0, "2024-02-22T00:00:00+00:00", False, "2.0.0")
    assert s.schema_version == "1.0"
    assert s.pipeline_version == "2.0.0"


def test_acceptance_health_monitor_publishes_all_8_sources(tmp_path, monkeypatch):
    """Health monitor publishes status for pradan + all 7 supplementary sources."""
    import asyncio
    from aruna.l0_source.session import PradanSession
    from aruna.l0_source.health import SourceHealthMonitor
    monkeypatch.setenv("PRADAN_COOKIES", "JSESSIONID=test;")
    session = PradanSession("https://pradan1.issdc.gov.in")
    published_sources = []
    producer = MagicMock()
    producer.send.side_effect = lambda topic, value: published_sources.append(value["source"])
    monitor = SourceHealthMonitor(
        session=session,
        kafka_producer=producer,
        kafka_topic="aruna.source.health",
        poll_interval_sec=0.01,
        health_file=tmp_path / "health.json",
        offline_fallback_sec=300.0,
    )
    with patch("aruna.l0_source.supplementary.requests.get", return_value=MagicMock(status_code=200)):
        asyncio.run(monitor._poll())
    expected = {"pradan", "noaa_swpc", "goes16_xrs", "hek", "jsoc_sharp", "cdaweb_ace", "cdaw_lasco", "gfz_kp"}
    assert set(published_sources) == expected


def test_acceptance_downloader_delay_respected(tmp_path, monkeypatch):
    """Inter-request delay ≥ configured value (rate-limit compliance)."""
    import time
    from aruna.l0_source.cache import CacheManager
    from aruna.l0_source.session import PradanSession
    from aruna.l0_source.downloader import Downloader
    monkeypatch.setenv("PRADAN_COOKIES", "JSESSIONID=test;")
    session = PradanSession("https://pradan1.issdc.gov.in")
    cache = CacheManager(tmp_path / "cache")
    dl = Downloader(session=session, cache=cache, delay_sec=0.1)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"fits"
    timestamps = []
    original_sleep = time.sleep
    def recording_sleep(t):
        timestamps.append(t)
        original_sleep(t)
    with patch("aruna.l0_source.downloader.time.sleep", side_effect=recording_sleep):
        with patch.object(session._http, "get", return_value=mock_resp):
            dl.download("https://pradan1.issdc.gov.in/test.fits", "solexs", "2024/02", "f1.fits")
            dl.download("https://pradan1.issdc.gov.in/test2.fits", "solexs", "2024/02", "f2.fits")
    assert len(timestamps) == 2
    assert all(t >= 0.1 for t in timestamps)


@pytest.mark.integration
def test_acceptance_noaa_swpc_reachable():
    """NOAA SWPC JSON endpoint reachable (fallback)."""
    from aruna.l0_source.supplementary import ALL_SUPPLEMENTARY_CLIENTS
    status, latency_ms = ALL_SUPPLEMENTARY_CLIENTS["noaa_swpc"].ping()
    assert status in ("ONLINE", "DEGRADED"), f"NOAA SWPC returned {status}"
```

- [ ] **Step 2: Run the full test suite**

```
pytest python/tests/test_l0_source.py -v --ignore-glob="*integration*" -m "not integration"
```
Expected: all tests PASSED (integration test skipped unless `-m integration` is added)

- [ ] **Step 3: Update `python/aruna/l0_source/__init__.py`**

```python
from .cache import CacheManager
from .downloader import Downloader
from .health import SourceHealthMonitor
from .manifest import ManifestEntry, ManifestParser, ManifestResult
from .models import SourceStatus
from .session import PradanSession, SessionExpiredError, SessionState
from .supplementary import ALL_SUPPLEMENTARY_CLIENTS

__all__ = [
    "CacheManager",
    "Downloader",
    "ManifestEntry",
    "ManifestParser",
    "ManifestResult",
    "PradanSession",
    "SessionExpiredError",
    "SessionState",
    "SourceHealthMonitor",
    "SourceStatus",
    "ALL_SUPPLEMENTARY_CLIENTS",
]
```

- [ ] **Step 4: Run full suite one final time**

```
pytest python/tests/test_l0_source.py -v -m "not integration"
```
Expected: all PASSED, 0 failures, integration tests skipped

- [ ] **Step 5: Final commit**

```
git add python/aruna/l0_source/__init__.py python/tests/test_l0_source.py
git commit -m "feat(l0): complete L0 source layer — all acceptance tests green"
```

---

## Self-Review

**Spec coverage check:**
- ✓ §3.1 PradanSession — Task 4
- ✓ §3.2 ManifestParser — Task 3
- ✓ §3.3 Downloader + manifest watch — Task 5 (manifest watch dir is operational, not in-process)
- ✓ §3.4 CacheManager — Task 2
- ✓ §3.5 SupplementaryFetcher (7 clients) — Task 6
- ✓ §3.6 SourceHealthMonitor — Task 7
- ✓ §4 Fallback cascade — tested in Task 7 (SESSION_EXPIRED, fallback_active, health.json)
- ✓ §6 Config keys — Task 1 step 5
- ✓ §7 PRADAN_COOKIES env var — Task 4
- ✓ §8 All acceptance tests — Task 8

**Placeholder scan:** None found.

**Type consistency:**
- `CacheManager.put/get` signatures consistent between Tasks 2, 5, and 8 ✓
- `SourceStatus` fields consistent between Tasks 1, 7, and 8 ✓
- `ManifestEntry.instrument` used in Task 5 matches Task 3 definition ✓
- `PradanSession._http` accessed in test mocks — matches implementation field name ✓
