import asyncio
import logging
import os
import signal
from enum import Enum
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)

_KEEPALIVE_PATH = "/al1/protected/payload.xhtml"


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
        except (OSError, ValueError, AttributeError):
            pass  # SIGHUP not available on Windows

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> SessionState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state == SessionState.ACTIVE

    def set_expired_callback(self, callback: Callable):
        self._on_expired = callback

    def reload(self):
        """Re-read PRADAN_COOKIES from environment. Safe to call from SIGHUP handler."""
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
            raise SessionExpiredError("Session is EXPIRED — update PRADAN_COOKIES and send SIGHUP")
        kwargs.setdefault("allow_redirects", False)
        kwargs.setdefault("timeout", 120)
        resp = self._http.get(url, headers={"Cookie": self._cookies}, **kwargs)
        if resp.status_code == 302:
            self._mark_expired()
            raise SessionExpiredError(f"HTTP 302 on {url} — session expired")
        return resp

    # ── keepalive ─────────────────────────────────────────────────────────────

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
            url = self._base_url + _KEEPALIVE_PATH
            resp = self._http.get(
                url,
                headers={"Cookie": self._cookies},
                allow_redirects=False,
                timeout=30,
            )
            if resp.status_code == 302:
                self._mark_expired()
            else:
                self._cycle_count += 1
                logger.debug("Keepalive cycle %d/%d", self._cycle_count, self._max_cycles)
        except Exception as exc:
            logger.warning("Keepalive request failed: %s", exc)

    # ── internal ──────────────────────────────────────────────────────────────

    def _reload(self):
        self.reload()

    def _mark_expired(self):
        self._state = SessionState.EXPIRED
        if self._on_expired:
            self._on_expired()
