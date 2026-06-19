"""
XL-1 — Magnetogram Integration (SDO/HMI SHARP parameters).

Provides 6–24 hour flare potential estimates from solar active region
magnetic field configuration. Operates as a soft prior on top of the
core Aditya-L1 (SoLEXS + HEL1OS) forecast output.

Design: always optional. Every downstream consumer must handle
MagneticContext = None without error.

Typical usage:
    from aruna.xl1_magnetograms import MagnetogramLayer

    layer = MagnetogramLayer()
    ctx = layer.get_context(harpnum=4977)        # None if JSOC unavailable
    frames = layer.fuse(forecast_frames, ctx)    # returns frames unchanged if ctx is None
"""
from .cache import SHARPCache
from .downloader import JSOCDownloader
from .fusion import SHARPFusionLayer
from .models import MagneticContext, SHARPConfig, SHARPRecord
from .parser import parse_rows, latest_record
from .sharp_features import build_magnetic_context, as_feature_vector

import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class MagnetogramLayer:
    """
    High-level façade for the XL-1 magnetogram integration.

    Wraps downloader + cache + parser + feature extraction + fusion into
    a single object. Inject into ForecastOrchestrator or call standalone.
    """

    def __init__(self, config: Optional[SHARPConfig] = None) -> None:
        self._cfg = config or SHARPConfig()
        self._dl    = JSOCDownloader(self._cfg)
        self._cache = SHARPCache(self._cfg)
        self._fuser = SHARPFusionLayer(self._cfg)

    def get_context(
        self,
        harpnum: int,
        timestamp: Optional[datetime] = None,
    ) -> Optional[MagneticContext]:
        """
        Fetch and process the most recent SHARP snapshot for a given HARP.

        Returns None on any failure — never raises.
        """
        try:
            date_str = (timestamp or datetime.utcnow()).strftime("%Y-%m-%d")
            records = self._cache.get(harpnum, date_str)
            if records is None:
                raw_rows = self._dl.fetch_latest(harpnum)
                records = parse_rows(raw_rows)
                self._cache.put(harpnum, date_str, records)
            return build_magnetic_context(records, self._cfg)
        except Exception as exc:
            logger.warning("MagnetogramLayer.get_context failed (HARP %d): %s", harpnum, exc)
            return None

    def fuse(self, frames, ctx: Optional[MagneticContext]):
        """Apply magnetic prior to ForecastFrames. Returns frames unchanged if ctx=None."""
        return self._fuser.apply(frames, ctx)

    def list_active_regions(self) -> List[int]:
        """Return HARP numbers of currently visible active regions. [] on failure."""
        try:
            return self._dl.list_active_harps()
        except Exception:
            return []


__all__ = [
    "MagnetogramLayer",
    "MagneticContext",
    "SHARPConfig",
    "SHARPRecord",
    "SHARPCache",
    "JSOCDownloader",
    "SHARPFusionLayer",
    "build_magnetic_context",
    "as_feature_vector",
    "parse_rows",
    "latest_record",
]
