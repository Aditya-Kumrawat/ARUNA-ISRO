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

# SUIT uses UV filters — these are not keV; stored as a fake 1-bin "energy" axis
_SUIT_ENERGY_BINS = np.array([0.0, 0.001])   # placeholder; downstream uses total_flux only
# VELC Doppler proxy similarly stored as single scalar per timestamp
_VELC_ENERGY_BINS = np.array([0.0, 0.001])

# Known SUIT FITS column names for integrated UV flux (try in order)
_SUIT_FLUX_COLS = ('UV_FLUX', 'FLUX', 'COUNTS', 'RATE', 'INTENSITY')
# Known VELC FITS column names for Doppler velocity proxy
_VELC_VEL_COLS  = ('DOPPLER_VEL', 'VEL_PROXY', 'FLUX', 'COUNTS')


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

    def read_suit(self, path: Path) -> FitsData:
        """Read SUIT L1 FITS → FitsData with integrated UV flux per timestamp.

        SUIT captures UV light curves at 60s cadence in multiple filter bands.
        For ARUNA, we only need the total integrated UV flux across all active
        filters — stored as a 1-bin scalar FitsData (energy_bins is a fake axis).

        Column name search order: UV_FLUX, FLUX, COUNTS, RATE, INTENSITY.
        Extension search order: LIGHTCURVE, SPECTRUM, EVENTS, then HDU index 1.
        """
        path = Path(path)
        checksum = _sha256(path)
        with fits.open(path) as hdul:
            ext_names = [hdu.name for hdu in hdul]
            lc_name = next(
                (n for n in ('LIGHTCURVE', 'SPECTRUM', 'EVENTS') if n in ext_names),
                None,
            )
            hdu = hdul[lc_name] if lc_name else hdul[1]
            timeref = _parse_timeref(hdu.header)
            timestamps = _times_to_dt64(hdu.data['TIME'].astype(np.float64), timeref)

            uv_col = next((c for c in _SUIT_FLUX_COLS if c in hdu.data.names), None)
            if uv_col is None:
                raise ValueError(
                    f"SUIT FITS {path.name} has no recognised flux column. "
                    f"Available: {list(hdu.data.names)}"
                )
            uv_flux = hdu.data[uv_col].astype(np.float64)
            if uv_flux.ndim == 1:
                uv_flux = uv_flux[:, np.newaxis]   # (N_times,) → (N_times, 1)

        return FitsData(
            instrument='SUIT_FD',
            timestamps=timestamps,
            counts_raw=uv_flux,
            energy_bins_kev=_SUIT_ENERGY_BINS.copy(),
            livetime=np.ones(len(timestamps)),
            deadtime_frac=np.zeros(len(timestamps)),   # UV imager: no count-rate dead-time
            source_file=path.name,
            checksum_sha256=checksum,
        )

    def read_velc(self, path: Path) -> FitsData:
        """Read VELC L1 FITS → FitsData with Doppler-proxy scalar per timestamp.

        VELC measures plasma evaporation velocity. For XL-2 MPCI, we need a
        0–1 score derived from the Doppler velocity magnitude. We store the raw
        proxy value here; score normalisation happens in xl/mpci.py.
        """
        path = Path(path)
        checksum = _sha256(path)
        with fits.open(path) as hdul:
            ext_names = [hdu.name for hdu in hdul]
            lc_name = next(
                (n for n in ('LIGHTCURVE', 'SPECTRUM', 'DOPPLER') if n in ext_names),
                None,
            )
            hdu = hdul[lc_name] if lc_name else hdul[1]
            timeref = _parse_timeref(hdu.header)
            timestamps = _times_to_dt64(hdu.data['TIME'].astype(np.float64), timeref)

            vel_col = next((c for c in _VELC_VEL_COLS if c in hdu.data.names), None)
            if vel_col is None:
                raise ValueError(
                    f"VELC FITS {path.name} has no recognised velocity column. "
                    f"Available: {list(hdu.data.names)}"
                )
            vel_proxy = hdu.data[vel_col].astype(np.float64)
            if vel_proxy.ndim == 1:
                vel_proxy = vel_proxy[:, np.newaxis]

        return FitsData(
            instrument='VELC_DP',
            timestamps=timestamps,
            counts_raw=vel_proxy,
            energy_bins_kev=_VELC_ENERGY_BINS.copy(),
            livetime=np.ones(len(timestamps)),
            deadtime_frac=np.zeros(len(timestamps)),
            source_file=path.name,
            checksum_sha256=checksum,
        )

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
