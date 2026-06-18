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
