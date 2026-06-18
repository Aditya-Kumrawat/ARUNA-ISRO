"""
L0 Source Layer Tests — Phase 1
PRD §4.6 acceptance criteria + brainstorm additions.
"""

import asyncio
import hashlib
import json
import textwrap
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Task 1: SourceStatus model ────────────────────────────────────────────────

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


# ── Task 2: CacheManager ──────────────────────────────────────────────────────

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
    path.write_bytes(b"corrupted!")
    result = cm.get("solexs", "2024/02", "corrupt.fits")
    assert result is None


def test_cache_ttl_expiry(tmp_path):
    cm = CacheManager(tmp_path / "cache", max_gb=1.0, ttl_hours=0.0001)  # ~0.36s
    cm.put("solexs", "2024/02", "old.fits", b"data")
    time.sleep(0.5)
    assert cm.get("solexs", "2024/02", "old.fits") is None


def test_cache_evicts_at_size_cap(tmp_path):
    # Cap at 25 bytes — first 20-byte put fits; second 20-byte put triggers eviction
    cm = CacheManager(tmp_path / "cache2", max_gb=25 / 1024**3, ttl_hours=48.0)
    cm.put("solexs", "2024/02", "first.fits", b"A" * 20)
    cm.put("solexs", "2024/02", "second.fits", b"B" * 20)
    assert cm.total_size_bytes() <= 25


def test_cache_no_tmp_after_put(tmp_path):
    cm = CacheManager(tmp_path / "cache", max_gb=1.0, ttl_hours=48.0)
    cm.put("solexs", "2024/02", "test.fits", b"data")
    tmps = list((tmp_path / "cache").rglob("*.tmp"))
    assert tmps == []


def test_cache_index_survives_stale_tmp(tmp_path):
    """Stale .tmp index from prior crash is cleaned; existing entries still readable."""
    cache_dir = tmp_path / "cache"
    cm = CacheManager(cache_dir, max_gb=1.0, ttl_hours=48.0)
    cm.put("solexs", "2024/02", "test.fits", b"data")
    # Simulate crash: leave a stale .tmp index
    (cache_dir / ".index.tmp").write_text("CORRUPT{{{")
    cm2 = CacheManager(cache_dir, max_gb=1.0, ttl_hours=48.0)
    assert cm2.get("solexs", "2024/02", "test.fits") is not None


# ── Task 3: ManifestParser ────────────────────────────────────────────────────

from aruna.l0_source.manifest import ManifestParser, ManifestResult

_SAMPLE_SCRIPT = textwrap.dedent("""\
    #! /bin/bash
    cookies="FGTServer=ABC123;JSESSIONID=b23e5074;OAuth_Token_Request_State=5431e0d8;"
    urlPrefix="https://pradan1.issdc.gov.in"
    proxyOptions=""
    counter=144;while [ $counter -gt 0 ]; do sleep 10m; wget $proxyOptions -N --content-disposition --tries=1 --no-cookies --header "Cookie: $cookies" $urlPrefix"/al1/protected/payload.xhtml"; counter=$(($counter-1)); done &
    bdpid=$!
    dataFilePaths=("/al1/protected/downloadData/velc/level1/2026/03/T26_0194/VS1_T26_0194_003417_20260314_135938_HG_lev1_V2_1.fits?velc" "/al1/protected/downloadData/solexs/level1/2026/03/T26_0100/SL1_T26_0100_20260314_135938_lev1.fits?all" )
    i=0;
    for file in ${dataFilePaths[@]}; do
        echo $file;
        wget $proxyOptions -x --max-redirect=0 --content-disposition --tries=1 --no-cookies --header "Cookie: $cookies" $urlPrefix$file;
    done
    kill -9 $bdpid
""")


def test_manifest_parser_extracts_cookies(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(_SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    assert "FGTServer=ABC123" in result.cookies
    assert "JSESSIONID=b23e5074" in result.cookies


def test_manifest_parser_extracts_url_prefix(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(_SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    assert result.url_prefix == "https://pradan1.issdc.gov.in"


def test_manifest_parser_extracts_two_entries(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(_SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    assert len(result.entries) == 2


def test_manifest_parser_infers_instruments(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(_SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    instruments = {e.instrument for e in result.entries}
    assert "velc" in instruments
    assert "solexs" in instruments


def test_manifest_parser_extracts_query_params(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(_SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    params = {e.query_param for e in result.entries}
    assert "velc" in params
    assert "all" in params


def test_manifest_parser_full_urls_include_prefix(tmp_path):
    script = tmp_path / "test.sh"
    script.write_text(_SAMPLE_SCRIPT)
    result = ManifestParser().parse(script)
    for entry in result.entries:
        assert entry.url.startswith("https://pradan1.issdc.gov.in")
        assert "/level1/" in entry.url


def test_manifest_parser_real_pradhan_sh():
    """Parse the actual Pradhan.sh from repo root."""
    pradhan = Path("Pradhan.sh")
    if not pradhan.exists():
        pytest.skip("Pradhan.sh not present in cwd")
    result = ManifestParser().parse(pradhan)
    assert result.cookies
    assert result.url_prefix == "https://pradan1.issdc.gov.in"
    assert len(result.entries) > 0


def test_manifest_parser_missing_cookies_raises(tmp_path):
    bad = tmp_path / "bad.sh"
    bad.write_text('urlPrefix="https://example.com"\ndataFilePaths=("/foo")\n')
    with pytest.raises(ValueError, match="No cookies"):
        ManifestParser().parse(bad)


# ── Task 4: PradanSession ─────────────────────────────────────────────────────

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
    monkeypatch.setenv("PRADAN_COOKIES", "JSESSIONID=test123;")
    s = PradanSession("https://pradan1.issdc.gov.in", keepalive_interval_min=0.001, max_cycles=1)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch.object(s._http, "get", return_value=mock_resp):
        asyncio.run(s._do_keepalive_tick())  # cycle_count → 1 (== max_cycles)
        asyncio.run(s._do_keepalive_tick())  # next tick: count >= max → EXPIRED
    assert s.state == SessionState.EXPIRED


# ── Task 5: Downloader ────────────────────────────────────────────────────────

from aruna.l0_source.downloader import Downloader
from aruna.l0_source.manifest import ManifestEntry


def _make_downloader(tmp_path, monkeypatch, cookies="JSESSIONID=test;"):
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
    mock_resp.raise_for_status = MagicMock()
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
    dl._cache.put("solexs", "2024/02", "cached.fits", b"cached data")
    entry = ManifestEntry(
        url="https://pradan1.issdc.gov.in/al1/protected/downloadData/solexs/level1/2024/02/T001/cached.fits",
        instrument="solexs",
        query_param="all",
    )
    http_calls = []
    original_get = dl._session.get
    with patch.object(dl._session, "get", side_effect=lambda *a, **kw: http_calls.append(1) or original_get(*a, **kw)):
        results = dl.download_manifest([entry])
    assert http_calls == []  # cache hit — no HTTP call
    assert len(results) == 1


def test_downloader_no_tmp_in_cache(tmp_path, monkeypatch):
    dl = _make_downloader(tmp_path, monkeypatch)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"fits data"
    mock_resp.raise_for_status = MagicMock()
    with patch.object(dl._session._http, "get", return_value=mock_resp):
        dl.download(
            "https://pradan1.issdc.gov.in/test.fits",
            "solexs", "2024/02", "notemps.fits",
        )
    tmps = list((tmp_path / "cache").rglob("*.tmp"))
    assert tmps == []


# ── Task 6: SupplementaryFetcher ─────────────────────────────────────────────

from aruna.l0_source.supplementary import ALL_SUPPLEMENTARY_CLIENTS

_EXPECTED_SOURCES = {
    "noaa_swpc", "goes16_xrs", "hek",
    "jsoc_sharp", "cdaweb_ace", "cdaw_lasco", "gfz_kp",
}


def test_supplementary_all_seven_clients_present():
    assert set(ALL_SUPPLEMENTARY_CLIENTS.keys()) == _EXPECTED_SOURCES


def test_supplementary_ping_returns_online_on_200():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    for name, client in ALL_SUPPLEMENTARY_CLIENTS.items():
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
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    for name, client in ALL_SUPPLEMENTARY_CLIENTS.items():
        with patch("aruna.l0_source.supplementary.requests.get", return_value=mock_resp):
            status, latency_ms = client.ping()
        assert status == "DEGRADED", f"{name} should be DEGRADED on 503"


# ── Task 7: SourceHealthMonitor ───────────────────────────────────────────────

from aruna.l0_source.health import SourceHealthMonitor


def _make_monitor(tmp_path, monkeypatch, cookies="JSESSIONID=test;"):
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
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("aruna.l0_source.supplementary.requests.get", return_value=mock_resp):
        asyncio.run(monitor._poll())
    assert producer.send.called
    first_call = producer.send.call_args_list[0]
    payload = first_call[1]["value"]
    assert payload["source"] == "pradan"
    assert payload["status"] == "ONLINE"


def test_health_monitor_session_expired_status(tmp_path, monkeypatch):
    monkeypatch.delenv("PRADAN_COOKIES", raising=False)
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
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("aruna.l0_source.supplementary.requests.get", return_value=mock_resp):
        asyncio.run(monitor._poll())
    payload = producer.send.call_args_list[0][1]["value"]
    assert payload["status"] == "SESSION_EXPIRED"


def test_health_monitor_writes_to_file_when_kafka_down(tmp_path, monkeypatch):
    monitor, producer = _make_monitor(tmp_path, monkeypatch)
    producer.send.side_effect = Exception("Kafka connection refused")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("aruna.l0_source.supplementary.requests.get", return_value=mock_resp):
        asyncio.run(monitor._poll())
    health_file = tmp_path / "data" / "health.json"
    assert health_file.exists()
    records = json.loads(health_file.read_text())
    assert isinstance(records, list)
    assert len(records) > 0
    assert records[0]["source"] == "pradan"


def test_health_monitor_schema_version_in_every_publish(tmp_path, monkeypatch):
    monitor, producer = _make_monitor(tmp_path, monkeypatch)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("aruna.l0_source.supplementary.requests.get", return_value=mock_resp):
        asyncio.run(monitor._poll())
    for call in producer.send.call_args_list:
        payload = call[1]["value"]
        assert payload["schema_version"] == "1.0"
        assert payload["pipeline_version"] == "2.0.0"


def test_health_monitor_fallback_activates_after_threshold(tmp_path, monkeypatch):
    monkeypatch.delenv("PRADAN_COOKIES", raising=False)
    session = PradanSession("https://pradan1.issdc.gov.in")
    producer = MagicMock()
    monitor = SourceHealthMonitor(
        session=session,
        kafka_producer=producer,
        kafka_topic="aruna.source.health",
        poll_interval_sec=0.01,
        health_file=tmp_path / "health.json",
        offline_fallback_sec=0.0,  # threshold=0 → first poll triggers fallback
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("aruna.l0_source.supplementary.requests.get", return_value=mock_resp):
        asyncio.run(monitor._poll())
        asyncio.run(monitor._poll())
    pradan_calls = [
        c[1]["value"] for c in producer.send.call_args_list
        if c[1]["value"]["source"] == "pradan"
    ]
    assert any(c["fallback_active"] for c in pradan_calls)


# ── Task 8: Acceptance gate ───────────────────────────────────────────────────


def test_acceptance_cache_directory_writable(tmp_path):
    cm = CacheManager(tmp_path / "cache")
    cm.put("solexs", "2024/02", "test.fits", b"data")
    assert cm.get("solexs", "2024/02", "test.fits") is not None


def test_acceptance_manifest_parses_real_pradhan_sh():
    """PRADAN connection established and authenticated (manifest + cookie extraction)."""
    pradhan = Path("Pradhan.sh")
    if not pradhan.exists():
        pytest.skip("Pradhan.sh not present in cwd")
    result = ManifestParser().parse(pradhan)
    assert result.cookies, "Cookies must not be empty"
    assert len(result.entries) > 0, "Must have at least one download entry"


def test_acceptance_source_status_schema():
    s = SourceStatus("pradan", "ONLINE", 10.0, "2024-02-22T00:00:00+00:00", False, "2.0.0")
    assert s.schema_version == "1.0"
    assert s.pipeline_version == "2.0.0"


def test_acceptance_health_monitor_publishes_all_8_sources(tmp_path, monkeypatch):
    """Health monitor publishes status for pradan + all 7 supplementary sources."""
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
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("aruna.l0_source.supplementary.requests.get", return_value=mock_resp):
        asyncio.run(monitor._poll())
    expected = {
        "pradan", "noaa_swpc", "goes16_xrs", "hek",
        "jsoc_sharp", "cdaweb_ace", "cdaw_lasco", "gfz_kp",
    }
    assert set(published_sources) == expected


def test_acceptance_downloader_delay_respected(tmp_path, monkeypatch):
    """Inter-request delay ≥ configured value (rate-limit compliance)."""
    monkeypatch.setenv("PRADAN_COOKIES", "JSESSIONID=test;")
    session = PradanSession("https://pradan1.issdc.gov.in")
    cache = CacheManager(tmp_path / "cache")
    dl = Downloader(session=session, cache=cache, delay_sec=0.1)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"fits"
    mock_resp.raise_for_status = MagicMock()
    sleep_calls = []
    with patch("aruna.l0_source.downloader.time.sleep", side_effect=lambda t: sleep_calls.append(t)):
        with patch.object(session._http, "get", return_value=mock_resp):
            dl.download("https://pradan1.issdc.gov.in/test.fits", "solexs", "2024/02", "f1.fits")
            dl.download("https://pradan1.issdc.gov.in/test2.fits", "solexs", "2024/02", "f2.fits")
    assert len(sleep_calls) == 2
    assert all(t >= 0.1 for t in sleep_calls)


@pytest.mark.integration
def test_acceptance_noaa_swpc_reachable():
    """NOAA SWPC JSON endpoint reachable (fallback)."""
    status, latency_ms = ALL_SUPPLEMENTARY_CLIENTS["noaa_swpc"].ping()
    assert status in ("ONLINE", "DEGRADED"), f"NOAA SWPC returned {status}"
