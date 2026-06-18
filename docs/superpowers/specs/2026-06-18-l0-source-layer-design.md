# L0 Source Layer — Design Spec
**Date:** 2026-06-18  
**PRD chapter:** §4  
**Phase:** 1  
**Status:** Approved

---

## 1. Objective

L0 is the boundary between the external world and the ARUNA pipeline. It is **passive** — it never processes scientific data, only provides authenticated access to raw telemetry. All external I/O lives here; every other layer talks to L0 through `CacheManager.get()` or a supplementary-source client.

---

## 2. Scope

### Primary source (PRADAN)
Session-cookie authenticated HTTPS portal at `https://pradan1.issdc.gov.in`. Provides:
- SoLEXS L1 FITS (1s cadence)
- HEL1OS L1 FITS (1s cadence)
- VELC L1 FITS (used by XL modules)
- SUIT L1 FITS (60s cadence)
- Housekeeping (HK) FITS (1s cadence)

### Supplementary sources (health-checked, fetched on-demand)
| Source | Role | Auth |
|--------|------|------|
| NOAA SWPC JSON | SoLEXS fallback + real-time cross-ref | None |
| GOES-16 XRS (AWS S3) | Cross-validation | Anonymous |
| SunPy HEK | GOES flare labels (L1 join) | None |
| JSOC SHARP (drms) | Magnetic parameters (XL-1) | None |
| CDAWeb ACE | Solar wind (XL-5) | None |
| CDAW LASCO | CME catalog (XL-5) | None |
| GFZ Potsdam Kp | Kp/Ap index (L6 SEP classification) | None |

---

## 3. Components

### 3.1 `PradanSession`
Holds the live PRADAN cookie string. Single instance shared across all instruments (SoLEXS, HEL1OS, VELC, SUIT — all authenticate to the same portal).

**State machine:**
```
ACTIVE ──(HTTP 302 on any protected URL)──→ EXPIRED
  │                                              │
  │                                              │ publishes SESSION_EXPIRED
  │                                              │ to aruna.source.health
  │                                        operator updates PRADAN_COOKIES
  │                                        in .env → sends SIGHUP
  └──────────────(reload())────────────────────→ ACTIVE
```

**Behaviour:**
- Reads `PRADAN_COOKIES` env var on init and on SIGHUP (hot-reload, no restart)
- Keepalive coroutine: `GET /al1/protected/payload.xhtml` every 10 minutes
- Keepalive counter: max 144 cycles (24h); on exhaustion, moves to `EXPIRED` proactively
- Any HTTP 302 on a download URL → `SessionExpiredError` (never followed; redirect = expiry)
- Cookie string held in memory only — never logged, never written to disk, never sent to Kafka

### 3.2 `ManifestParser`
Parses PRADAN-generated `.sh` download scripts (the format produced by the portal's "download script" button).

**Extracts:**
- Cookie string (`cookies="..."` line)
- URL prefix (`urlPrefix="..."` line)
- List of `(relative_url, instrument, query_param)` tuples from `dataFilePaths` array
- Query param suffix (`?all`, `?velc`, etc.) identifies instrument context

**API:** `ManifestParser.parse(path: Path) -> ManifestResult`  
Can ingest multiple `.sh` files in one session — all share the same `PradanSession` cookie.

### 3.3 `Downloader`
Sequential HTTPS fetcher. Never parallelises requests (PRADAN rate-limit compliance).

**Behaviour:**
- Uses `requests.Session` with `allow_redirects=False` — 302 → `SessionExpiredError`
- Downloads to `{destination}.tmp` first; on completion computes SHA-256; atomically renames to final path
- Configurable inter-request delay: `data.download_delay_sec` in `config.yaml` (default 2s)
- On `SessionExpiredError`: notifies `SourceHealthMonitor`, pauses queue, waits for `PradanSession` to return to `ACTIVE`
- Incomplete `.tmp` files cleaned up on startup
- Watches `data/manifests/` for new `.sh` files (polling, 30s interval); auto-ingests via `ManifestParser` and appends URLs to download queue
- **Real-time operation model**: operator generates a PRADAN download script covering the desired time window, drops it into `data/manifests/`; L0 picks it up and downloads to cache; L1 reads from cache. This is the PRADAN-imposed constraint — no programmatic file discovery exists.

### 3.4 `CacheManager`
LRU filesystem cache at `data/cache/`. L1 and all other layers call `get()` here — they never touch PRADAN directly.

**Layout:**
```
data/cache/
  solexs/YYYY/MM/DD/{filename}.fits
  hel1os/YYYY/MM/DD/{filename}.fits
  velc/YYYY/MM/DD/{filename}.fits
  suit/YYYY/MM/DD/{filename}.fits
  hk/YYYY/MM/DD/{filename}.fits
  .index.json    ← {path: {sha256, size_bytes, inserted_at, last_accessed}}
```

**Eviction rules:**
- Hard cap: 50 GB; on every `put()` evict LRU entries until under cap
- TTL: 48h; `get()` returns `None` for entries older than 48h (stale → triggers fallback)
- `.index.json` written atomically (write-to-tmp + rename) — crash-safe

**Integrity:**
- SHA-256 computed from downloaded bytes on write, stored in index
- Every `get()` re-verifies SHA-256 before returning path
- Mismatch → evict entry, return `None` (triggers re-download)
- No PRADAN manifest to compare against — SHA-256 is self-referential (detects bit-rot)

### 3.5 `SupplementaryFetcher`
One thin client per supplementary source. Each implements a `ping() -> SourceStatus` method used by the health monitor. Data fetching is done on-demand by the consuming layer (L1, L6, XL modules) — L0 only confirms reachability.

Sources: `NoaaSwpcClient`, `Goes16XrsClient`, `HekClient`, `JsocSharpClient`, `CdawebAceClient`, `CdawLascoClient`, `GfzKpClient`.

### 3.6 `SourceHealthMonitor`
Asyncio background task. Polls every 30s.

**Publishes to `aruna.source.health`:**
```python
@dataclass
class SourceStatus:
    source: str              # "pradan" | "noaa_swpc" | "goes16_xrs" | ...
    status: str              # "ONLINE" | "DEGRADED" | "OFFLINE" | "SESSION_EXPIRED"
    latency_ms: float
    last_success_utc: str    # ISO 8601 UTC
    fallback_active: bool
    pipeline_version: str
    schema_version: str = "1.0"
```

**Kafka resilience:** if Kafka is unreachable, status written to `data/health.json` (atomic write). Flushed to Kafka on recovery. Health monitoring never depends on Kafka being live.

---

## 4. Fallback Cascade

```
PRADAN ONLINE                → normal operation
PRADAN SESSION_EXPIRED       → DATA_DEGRADED, pipeline continues on cache
PRADAN OFFLINE > 5 minutes   → trigger NOAA SWPC fallback,
                               publish DATA_FALLBACK alert,
                               set fallback_active=True on SourceStatus
Cache entry > 48h old        → NOAA fallback only, DATA_STALE flag on ObservationFrame
All sources offline           → publish CRITICAL_NO_DATA,
                               pipeline halts ingestion (does not crash),
                               retries every 60s
```

---

## 5. File Structure

```
python/aruna/l0_source/
  __init__.py
  session.py        # PradanSession — cookie manager, keepalive, SIGHUP reload
  manifest.py       # ManifestParser — PRADAN .sh script parser
  downloader.py     # Downloader — sequential HTTPS fetcher
  cache.py          # CacheManager — LRU cache, SHA-256, eviction
  supplementary.py  # SupplementaryFetcher — 7 thin source clients
  health.py         # SourceHealthMonitor — asyncio poller, Kafka publisher
  models.py         # SourceStatus dataclass
```

---

## 6. Configuration (`config.yaml` keys)

```yaml
data:
  cache_dir: data/cache/
  cache_ttl_hours: 48
  cache_max_gb: 50
  download_delay_sec: 2.0
  pradan_url: https://pradan1.issdc.gov.in
  pradan_keepalive_interval_min: 10
  pradan_keepalive_max_cycles: 144
  pradan_offline_fallback_sec: 300   # 5 minutes
  manifest_watch_dir: data/manifests/
```

---

## 7. Environment Variables

```
PRADAN_COOKIES   — full cookie string from PRADAN download script (hot-reloadable via SIGHUP)
```
Never stored in `config.yaml`. Never committed to git. Rotated every ~24h by operator.

---

## 8. Acceptance Tests

### From PRD §4.6
```
✓ PRADAN connection established and authenticated
✓ SoLEXS FITS readable with astropy.io.fits
✓ HEL1OS FITS readable with astropy.io.fits
✓ NOAA SWPC JSON endpoint reachable
✓ Cache directory writable
✓ Health monitor publishing to aruna.source.health
```

### Added from brainstorm
```
✓ ManifestParser extracts cookies + URL list from real Pradhan.sh format
✓ SessionExpiredError fires on HTTP 302 (injected mock redirect)
✓ SIGHUP cookie hot-reload resumes downloads without pipeline restart
✓ SHA-256 mismatch on cache read → entry evicted, re-download triggered
✓ Cache eviction enforces 50 GB hard cap (synthetic oversized entries)
✓ LRU TTL: entry > 48h → get() returns None
✓ Keepalive counter exhaustion (144 cycles) → proactive EXPIRED state
✓ Kafka down → health status written to data/health.json, not lost
✓ All 7 supplementary sources return ONLINE or graceful OFFLINE status
✓ Inter-request delay ≥ 2s verified (rate-limit compliance)
✓ .tmp file never present in cache after completed download
✓ Atomic index write: simulated crash mid-write → index not corrupted on reload
✓ VELC and SUIT manifests parse correctly alongside SoLEXS/HEL1OS
✓ Multiple .sh manifests for same session → single PradanSession shared
```

---

## 9. Out of Scope for L0

- Dead-time correction → L1
- FITS header parsing beyond instrument/timestamp identification → L1
- GOES label joining → L1
- Scientific data validation → L1 (pandera schema)
- Kafka topic schemas beyond `aruna.source.health` → downstream layers
