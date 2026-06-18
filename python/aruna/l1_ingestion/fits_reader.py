from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
from astropy.io import fits

from .models import FitsData, HKData

logger = logging.getLogger(__name__)

SOLEXS_DETECTORS = {'SDD1': 'SOLEXS_SDD1', 'SDD2': 'SOLEXS_SDD2'}
HEL1OS_DETECTORS = {'CDTE': 'HEL1OS_CDTE', 'CZT': 'HEL1OS_CZT'}
HEL1OS_EBOUNDS   = {'CDTE': 'EBOUNDS_CDTE', 'CZT': 'EBOUNDS_CZT'}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _parse_timeref(header) -> datetime:
    s = header.get('TIMEREF', '2000-01-01T00:00:00')
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _times_to_dt64(times_sec: np.ndarray, timeref: datetime) -> np.ndarray:
    epoch = np.datetime64(timeref.replace(tzinfo=None).isoformat(), 'ns')
    offsets = (times_sec * 1e9).astype('int64').astype('timedelta64[ns]')
    return epoch + offsets


def _read_ebounds(hdul: fits.HDUList, ext_name: str) -> np.ndarray:
    try:
        eb = hdul[ext_name]
        e_min = eb.data['E_MIN'].astype(np.float64)
        e_max = eb.data['E_MAX'].astype(np.float64)
        return np.append(e_min, e_max[-1])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Missing or malformed {ext_name} extension: {exc}") from exc


class FitsReader:
    def read_solexs(self, path: Path) -> List[FitsData]:
        """Read SoLEXS FITS → [FitsData(SDD1), FitsData(SDD2)]."""
        path = Path(path)
        checksum = _sha256(path)
        result = []
        with fits.open(path) as hdul:
            energy_bins = _read_ebounds(hdul, 'EBOUNDS')
            for ext_name, instrument in SOLEXS_DETECTORS.items():
                hdu = hdul[ext_name]
                timeref = _parse_timeref(hdu.header)
                result.append(FitsData(
                    instrument=instrument,
                    timestamps=_times_to_dt64(hdu.data['TIME'], timeref),
                    counts_raw=hdu.data['COUNTS'].astype(np.float64),
                    energy_bins_kev=energy_bins,
                    livetime=hdu.data['LIVETIME'].astype(np.float64),
                    deadtime_frac=hdu.data['DEADTIME_FRAC'].astype(np.float64),
                    source_file=path.name,
                    checksum_sha256=checksum,
                ))
        return result

    def read_hel1os(self, path: Path) -> List[FitsData]:
        """Read HEL1OS FITS → [FitsData(CDTE), FitsData(CZT)]."""
        path = Path(path)
        checksum = _sha256(path)
        result = []
        with fits.open(path) as hdul:
            for ext_name, instrument in HEL1OS_DETECTORS.items():
                hdu = hdul[ext_name]
                timeref = _parse_timeref(hdu.header)
                energy_bins = _read_ebounds(hdul, HEL1OS_EBOUNDS[ext_name])
                result.append(FitsData(
                    instrument=instrument,
                    timestamps=_times_to_dt64(hdu.data['TIME'], timeref),
                    counts_raw=hdu.data['COUNTS'].astype(np.float64),
                    energy_bins_kev=energy_bins,
                    livetime=hdu.data['LIVETIME'].astype(np.float64),
                    deadtime_frac=hdu.data['DEADTIME_FRAC'].astype(np.float64),
                    source_file=path.name,
                    checksum_sha256=checksum,
                ))
        return result

    def read_hk(self, path: Path) -> HKData:
        """Read HK FITS → HKData."""
        path = Path(path)
        with fits.open(path) as hdul:
            hdu = hdul['HK']
            timeref = _parse_timeref(hdu.header)
            return HKData(
                timestamps=_times_to_dt64(hdu.data['TIME'], timeref),
                saa_flags=hdu.data['SAA_FLAG'].astype(bool),
                instrument_modes=list(hdu.data['INSTRUMENT_MODE']),
                solar_dist_au=hdu.data['SOLAR_DIST_AU'].astype(np.float64),
            )
