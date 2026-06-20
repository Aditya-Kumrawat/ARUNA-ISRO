from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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

# PRADAN SoLEXS SDD energy calibration (linear approximation, 340 channels)
_PRADAN_E_MIN_KEV = 1.5
_PRADAN_E_MAX_KEV = 15.0
_PRADAN_N_CH      = 340

# PRADAN HEL1OS energy calibration fallbacks (used when EBOUNDS absent in file)
_HEL1OS_CDTE_E_MIN = 10.0
_HEL1OS_CDTE_E_MAX = 100.0
_HEL1OS_CZT_E_MIN  = 30.0
_HEL1OS_CZT_E_MAX  = 150.0
_HEL1OS_N_CH_DEFAULT = 256  # CdTe; CZT may differ

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

    def read_solexs_pradan_pi(
        self,
        pi_path: Path,
        gti_path: Optional[Path] = None,
        detector: str = "SDD2",
        window_center_unix: Optional[float] = None,
        window_sec: int = 5400,
    ) -> List[FitsData]:
        """
        Read a PRADAN SoLEXS .pi.gz file (per-detector PI spectrum).

        PRADAN level-1 format stores each detector as a separate file:
          - AL1_SOLEXS_YYYYMMDD_SDD{N}_L1.pi.gz  — SPECTRUM extension:
              TSTART (Unix s), TELAPSE (1.0), CHANNEL (0-339), COUNTS (340), EXPOSURE (1.0)
          - AL1_SOLEXS_YYYYMMDD_SDD{N}_L1.gti.gz — GTI extension: START, STOP (Unix s)

        Args:
            pi_path:              .pi.gz file path
            gti_path:             companion .gti.gz path (optional)
            detector:             "SDD1" or "SDD2"
            window_center_unix:   Unix timestamp to centre the extraction window on
            window_sec:           total window width in seconds (default 5400 = 90 min)

        Returns:
            List with a single FitsData entry for the detector.
        """
        pi_path = Path(pi_path)
        checksum = _sha256(pi_path)
        instrument = SOLEXS_DETECTORS.get(detector, f"SOLEXS_{detector}")

        with fits.open(pi_path) as hdul:
            spec = hdul['SPECTRUM']
            tstart_all  = spec.data['TSTART'].astype(np.float64)   # Unix seconds
            counts_all  = spec.data['COUNTS'].astype(np.float64)   # (N, 340)
            exposure_all = spec.data['EXPOSURE'].astype(np.float64) # seconds

        # Build mask: start with all valid
        mask = np.ones(len(tstart_all), dtype=bool)

        # GTI masking
        if gti_path is not None:
            gti_path = Path(gti_path)
            try:
                with fits.open(gti_path) as ghdul:
                    gti = ghdul['GTI']
                    if len(gti.data) > 0:
                        gti_mask = np.zeros(len(tstart_all), dtype=bool)
                        for row in gti.data:
                            s, e = float(row['START']), float(row['STOP'])
                            gti_mask |= (tstart_all >= s) & (tstart_all <= e)
                        mask &= gti_mask
                        logger.info("GTI: %d valid intervals, %d/%d frames selected",
                                    len(gti.data), gti_mask.sum(), len(tstart_all))
            except Exception as exc:
                logger.warning("GTI masking failed (%s) — using all rows", exc)

        # Time window
        if window_center_unix is not None:
            half = window_sec / 2.0
            win_mask = ((tstart_all >= window_center_unix - half) &
                        (tstart_all <= window_center_unix + half))
            mask &= win_mask
            logger.info("Time window: %d rows in [%.0f, %.0f]",
                        win_mask.sum(), window_center_unix - half, window_center_unix + half)

        # NaN filter
        with np.errstate(invalid='ignore'):
            nan_mask = ~np.isnan(counts_all.sum(axis=1))
        mask &= nan_mask

        # Apply combined mask
        tstart   = tstart_all[mask]
        counts_s = counts_all[mask]
        exposure = exposure_all[mask]

        # Convert to counts/s per channel (EXPOSURE is live time in seconds)
        exposure_safe = np.where(exposure > 0, exposure, 1.0)
        counts_rate   = counts_s / exposure_safe[:, np.newaxis]

        # Livetime fraction: PRADAN stores absolute exposure per 1s cadence
        livetime_frac = np.minimum(exposure_safe, 1.0)
        deadtime_frac = 1.0 - livetime_frac

        # Timestamps: TSTART is absolute Unix seconds → datetime64[ns]
        epoch = np.datetime64('1970-01-01T00:00:00', 'ns')
        timestamps = epoch + (tstart * 1e9).astype('int64').astype('timedelta64[ns]')

        # Energy bins: linear 1.5–15 keV across 340 channels (no EBOUNDS in file)
        energy_bins = np.linspace(_PRADAN_E_MIN_KEV, _PRADAN_E_MAX_KEV, _PRADAN_N_CH + 1)

        logger.info("read_solexs_pradan_pi: %d frames loaded from %s (detector=%s)",
                    len(timestamps), pi_path.name, detector)

        return [FitsData(
            instrument=instrument,
            timestamps=timestamps,
            counts_raw=counts_rate,
            energy_bins_kev=energy_bins,
            livetime=livetime_frac,
            deadtime_frac=deadtime_frac,
            source_file=pi_path.name,
            checksum_sha256=checksum,
        )]

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

    def read_hel1os_pradan_pi(
        self,
        pi_path: Path,
        gti_path: Optional[Path] = None,
        detector: str = "CDTE",
        window_center_unix: Optional[float] = None,
        window_sec: int = 5400,
    ) -> List[FitsData]:
        """
        Read a PRADAN HEL1OS .pi.gz file (per-detector PI spectrum).

        PRADAN level-1 HEL1OS format mirrors SoLEXS:
          SPECTRUM extension: TSTART (Unix s), COUNTS (N, n_ch), EXPOSURE (s)
          Optional EBOUNDS extension: E_MIN, E_MAX per channel

        Falls back to expected energy ranges when EBOUNDS is absent:
          CdTe: 10–100 keV (256 ch default), CZT: 30–150 keV

        Returns a list with a single FitsData entry.
        """
        pi_path  = Path(pi_path)
        checksum = _sha256(pi_path)
        det_up   = detector.upper()
        instrument = HEL1OS_DETECTORS.get(det_up, f"HEL1OS_{det_up}")

        with fits.open(pi_path) as hdul:
            spec        = hdul['SPECTRUM']
            tstart_all  = spec.data['TSTART'].astype(np.float64)
            counts_all  = spec.data['COUNTS'].astype(np.float64)
            exposure_all = spec.data['EXPOSURE'].astype(np.float64)

            # Energy bins — prefer EBOUNDS extension, fall back to nominal range
            ebounds_name = HEL1OS_EBOUNDS.get(det_up, f"EBOUNDS_{det_up}")
            if ebounds_name in [h.name for h in hdul]:
                energy_bins = _read_ebounds(hdul, ebounds_name)
            else:
                n_ch = counts_all.shape[1] if counts_all.ndim == 2 else _HEL1OS_N_CH_DEFAULT
                e_min = _HEL1OS_CZT_E_MIN if det_up == "CZT" else _HEL1OS_CDTE_E_MIN
                e_max = _HEL1OS_CZT_E_MAX if det_up == "CZT" else _HEL1OS_CDTE_E_MAX
                energy_bins = np.linspace(e_min, e_max, n_ch + 1)
                logger.info(
                    "HEL1OS %s: no EBOUNDS — using nominal %.0f–%.0f keV (%d ch)",
                    det_up, e_min, e_max, n_ch,
                )

        mask = np.ones(len(tstart_all), dtype=bool)

        if gti_path is not None:
            gti_path = Path(gti_path)
            try:
                with fits.open(gti_path) as ghdul:
                    gti = ghdul['GTI']
                    if len(gti.data) > 0:
                        gti_mask = np.zeros(len(tstart_all), dtype=bool)
                        for row in gti.data:
                            s, e = float(row['START']), float(row['STOP'])
                            gti_mask |= (tstart_all >= s) & (tstart_all <= e)
                        mask &= gti_mask
            except Exception as exc:
                logger.warning("HEL1OS GTI masking failed (%s) — using all rows", exc)

        if window_center_unix is not None:
            half     = window_sec / 2.0
            win_mask = (tstart_all >= window_center_unix - half) & (tstart_all <= window_center_unix + half)
            mask    &= win_mask

        with np.errstate(invalid='ignore'):
            mask &= ~np.isnan(counts_all.sum(axis=1))

        tstart   = tstart_all[mask]
        counts_s = counts_all[mask]
        exposure = exposure_all[mask]

        exposure_safe  = np.where(exposure > 0, exposure, 1.0)
        counts_rate    = counts_s / exposure_safe[:, np.newaxis]
        livetime_frac  = np.minimum(exposure_safe, 1.0)
        deadtime_frac  = 1.0 - livetime_frac

        epoch      = np.datetime64('1970-01-01T00:00:00', 'ns')
        timestamps = epoch + (tstart * 1e9).astype('int64').astype('timedelta64[ns]')

        logger.info(
            "read_hel1os_pradan_pi: %d frames from %s (detector=%s)",
            len(timestamps), pi_path.name, detector,
        )
        return [FitsData(
            instrument=instrument,
            timestamps=timestamps,
            counts_raw=counts_rate,
            energy_bins_kev=energy_bins,
            livetime=livetime_frac,
            deadtime_frac=deadtime_frac,
            source_file=pi_path.name,
            checksum_sha256=checksum,
        )]

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
