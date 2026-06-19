from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from ..models.observation import ObservationFrame, QualityFlag
from .alignment import TimeAligner
from .background import BackgroundSubtractor
from .deadtime import DeadTimeCorrector
from .fits_reader import FitsReader
from .goes_join import GoesLabelJoiner
from .models import FitsData, HKData, IngestionConfig
from .publisher import KafkaPublisher
from .validator import FrameValidator

logger = logging.getLogger(__name__)

_PIPELINE_VERSION = "2.0.0"
_CALIBRATION_VERSION = "sarwade_2025_v1"


def _dt64_to_datetime(t: np.datetime64) -> datetime:
    ts = float((t - np.datetime64('1970-01-01T00:00:00', 'ns')) / np.timedelta64(1, 's'))
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _nearest_hk(hk: HKData, ts: np.datetime64):
    """Find HK values nearest to ts."""
    idx = int(np.argmin(np.abs(hk.timestamps - ts)))
    return (
        bool(hk.saa_flags[idx]),
        str(hk.instrument_modes[idx]),
        float(hk.solar_dist_au[idx]),
    )


def _fits_to_frames(
    data: FitsData,
    hk: HKData,
    quality_flags: np.ndarray,
) -> List[ObservationFrame]:
    frames = []
    for i in range(len(data.timestamps)):
        saa, mode, dist = _nearest_hk(hk, data.timestamps[i])
        flag = QualityFlag(int(quality_flags[i]))
        if saa and flag == QualityFlag.GOOD:
            flag = QualityFlag.SAA_PASSAGE
        frames.append(ObservationFrame(
            instrument=data.instrument,
            timestamp=_dt64_to_datetime(data.timestamps[i]),
            energy_bins_kev=data.energy_bins_kev,
            counts_per_s_kev=data.counts_raw[i],
            quality_flag=flag,
            dead_time_fraction=float(data.deadtime_frac[i]),
            calibration_version=_CALIBRATION_VERSION,
            solar_distance_au=dist,
            spacecraft_mode=mode,
            saa_flag=saa,
            source_file=data.source_file,
            checksum_sha256=data.checksum_sha256,
            pipeline_version=_PIPELINE_VERSION,
        ))
    return frames


class IngestionPipeline:
    def __init__(
        self,
        cache,
        config: Optional[IngestionConfig] = None,
        kafka_producer=None,
    ):
        self._cache = cache
        self._cfg = config or IngestionConfig()
        self._reader = FitsReader()
        self._dtc = DeadTimeCorrector()
        self._bg = BackgroundSubtractor(chi2_dof_limit=self._cfg.chi2_dof_limit)
        self._aligner = TimeAligner(self._cfg.clock_drift_warn_sec, self._cfg.clock_drift_max_sec)
        self._joiner = GoesLabelJoiner(self._cfg.goes_match_window_min, self._cfg.goes_query_window_min)
        self._validator = FrameValidator(self._cfg)
        self._publisher = KafkaPublisher(
            kafka_producer,
            self._cfg.kafka_topic_solexs,
            self._cfg.kafka_topic_hel1os,
            self._cfg.kafka_topic_suit,
            self._cfg.kafka_topic_velc,
            self._cfg.kafka_topic_quality,
        ) if kafka_producer else None

    def ingest(
        self,
        solexs_path: Path,
        hel1os_path: Path,
        hk_path: Path,
        hek_results=None,
        suit_path: Optional[Path] = None,
        velc_path: Optional[Path] = None,
    ) -> List[ObservationFrame]:
        """
        Full L1 ingestion pipeline: FITS → corrected → aligned → labelled → validated → published.

        suit_path, velc_path are optional. When absent the pipeline runs in the
        original 2-instrument mode; no SUIT/VELC frames are emitted.
        """
        hk = self._reader.read_hk(hk_path)

        # ── SoLEXS: dead-time correction + SDD1/SDD2 selection ──────────────
        solexs_list = self._reader.read_solexs(solexs_path)
        sdd1 = next(fd for fd in solexs_list if fd.instrument == 'SOLEXS_SDD1')
        sdd2 = next(fd for fd in solexs_list if fd.instrument == 'SOLEXS_SDD2')
        corr_sdd1, flags_sdd1 = self._dtc.correct(sdd1)
        corr_sdd2, flags_sdd2 = self._dtc.correct(sdd2)
        solexs_data, solexs_flags = self._dtc.select_solexs_detector(
            corr_sdd1, flags_sdd1, corr_sdd2, flags_sdd2
        )

        # ── HEL1OS: dead-time correction + background subtraction ────────────
        hel1os_list = self._reader.read_hel1os(hel1os_path)
        cdte = next(fd for fd in hel1os_list if fd.instrument == 'HEL1OS_CDTE')
        corr_cdte, flags_cdte = self._dtc.correct(cdte)
        saa_flags_cdte = np.array([_nearest_hk(hk, ts)[0] for ts in cdte.timestamps])
        bg_cdte, bg_flags_cdte, _ = self._bg.subtract(corr_cdte, [], saa_flags_cdte)

        # ── SoLEXS / HEL1OS time alignment ──────────────────────────────────
        solexs_aligned, hel1os_aligned, _ = self._aligner.align(solexs_data, bg_cdte)
        n_sol = len(solexs_aligned.timestamps)
        n_hel = len(hel1os_aligned.timestamps)
        solexs_flags_trim = solexs_flags[:n_sol]
        if n_hel <= len(bg_flags_cdte):
            hel1os_flags_trim = bg_flags_cdte[:n_hel]
        else:
            hel1os_flags_trim = np.full(n_hel, int(QualityFlag.GOOD), dtype=np.int8)
            hel1os_flags_trim[:len(bg_flags_cdte)] = bg_flags_cdte

        # ── Build X-ray ObservationFrames ────────────────────────────────────
        all_frames: List[ObservationFrame] = []
        all_frames.extend(_fits_to_frames(solexs_aligned, hk, solexs_flags_trim))
        all_frames.extend(_fits_to_frames(hel1os_aligned, hk, hel1os_flags_trim))

        # ── SUIT: resample 60s cadence → 1s X-ray grid ─────────────────────
        if suit_path is not None:
            suit_path = Path(suit_path)
            try:
                suit_raw = self._reader.read_suit(suit_path)
                suit_resampled = self._aligner.resample_suit_to_xray(
                    suit_raw, solexs_aligned.timestamps
                )
                suit_flags = np.full(n_sol, int(QualityFlag.GOOD), dtype=np.int8)
                all_frames.extend(_fits_to_frames(suit_resampled, hk, suit_flags))
            except Exception as exc:
                logger.warning("SUIT ingestion failed (%s) — continuing without SUIT", exc)

        # ── VELC: resample to 1s X-ray grid (optional) ─────────────────────
        if velc_path is not None:
            velc_path = Path(velc_path)
            try:
                velc_raw = self._reader.read_velc(velc_path)
                velc_resampled = self._aligner.resample_suit_to_xray(
                    velc_raw, solexs_aligned.timestamps   # same logic as SUIT resampling
                )
                velc_flags = np.full(n_sol, int(QualityFlag.GOOD), dtype=np.int8)
                all_frames.extend(_fits_to_frames(velc_resampled, hk, velc_flags))
            except Exception as exc:
                logger.warning("VELC ingestion failed (%s) — continuing without VELC", exc)

        # ── GOES label join ──────────────────────────────────────────────────
        all_frames = self._joiner.join(all_frames, hek_results)

        # ── Schema validation ────────────────────────────────────────────────
        valid_frames, _ = self._validator.validate(all_frames)

        # ── Kafka publish ────────────────────────────────────────────────────
        if self._publisher:
            self._publisher.publish(valid_frames)

        return valid_frames
