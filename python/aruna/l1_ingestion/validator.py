from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pandera as pa

from ..models.observation import ObservationFrame, QualityFlag
from .models import IngestionConfig, QuarantineEntry

logger = logging.getLogger(__name__)

_SCHEMA = pa.DataFrameSchema({
    'instrument':         pa.Column(str,   nullable=False),
    'dead_time_fraction': pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False),
    'solar_distance_au':  pa.Column(float, pa.Check.greater_than(0.0), nullable=False),
    'quality_flag':       pa.Column(int,   pa.Check.isin([f.value for f in QualityFlag]), nullable=False),
}, coerce=True)


class FrameValidator:
    def __init__(self, config: IngestionConfig):
        self._cfg = config
        self._quarantine: List[QuarantineEntry] = []

    @property
    def quarantine(self) -> List[QuarantineEntry]:
        return list(self._quarantine)

    def validate(
        self,
        frames: List[ObservationFrame],
    ) -> Tuple[List[ObservationFrame], List[ObservationFrame]]:
        if not frames:
            return [], []

        records = [{
            'instrument':         f.instrument,
            'dead_time_fraction': f.dead_time_fraction,
            'solar_distance_au':  f.solar_distance_au,
            'quality_flag':       int(f.quality_flag),
        } for f in frames]
        try:
            _SCHEMA.validate(pd.DataFrame(records), lazy=True)
        except pa.errors.SchemaErrors as exc:
            logger.warning("pandera errors: %s", exc.failure_cases.shape)

        valid, quarantined = [], []
        now = datetime.now(timezone.utc)
        for frame in frames:
            if self._is_complete(frame):
                valid.append(frame)
            else:
                quarantined.append(frame)
                self._quarantine.append(QuarantineEntry(
                    fits_path=Path(frame.source_file),
                    instrument=frame.instrument,
                    quarantined_at=now,
                    retry_after=now + timedelta(seconds=self._cfg.quarantine_ttl_sec),
                    reason="completeness < 80%",
                ))
                logger.warning("Frame %s quarantined: low completeness", frame.frame_id)

        if frames:
            rate = len(valid) / len(frames)
            if rate < 0.95:
                logger.error("Validation pass rate %.1f%% < 95%%", rate * 100)
        return valid, quarantined

    def _is_complete(self, frame: ObservationFrame) -> bool:
        c = frame.counts_per_s_kev
        if c is None or len(c) == 0:
            return False
        return float(np.isfinite(c).sum()) / len(c) >= self._cfg.completeness_threshold
