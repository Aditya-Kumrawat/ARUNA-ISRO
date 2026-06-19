"""
ForecastOrchestrator — stateful L4 entry point.

Owns the 600-frame sliding buffer and coordinates all sub-components:
  FeatureExtractor → DualStreamTCN → MoiraiForecaster → EnsembleWeighter
  → MapieCalibrator → ForecastFrame assembly → Kafka publish

Not thread-safe. One instance per Kafka consumer partition.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime
from typing import List, Optional

import numpy as np

from aruna.models.forecast import ForecastFrame, CLASS_LABELS, FORECAST_HORIZONS
from aruna.models.nowcast import NowcastEvent
from aruna.models.physics import PhysicsFrame
from .calibrator import MapieCalibrator
from .ensemble import EnsembleWeighter
from .features import FeatureExtractor
from .models import ForecastConfig
from .moirai_model import MoiraiForecaster

# Lazy torch import — required for TCN forward pass
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def _build_forecast_frame(
    probs: np.ndarray,             # (5,) calibrated
    prediction_set: List[str],
    ece: float,
    horizon_minutes: int,
    tcn_weight: float,
    moirai_weight: float,
    buffer: deque,
    hope_flag_active: bool,
    physics_state: str,
    config: ForecastConfig,
    lead_time_est_sec: int,
    lead_time_unc_sec: int,
) -> ForecastFrame:
    """Assemble a single ForecastFrame from calibrated outputs."""
    argmax_idx = int(np.argmax(probs))
    predicted_class = CLASS_LABELS[argmax_idx]
    peak_class_prob = float(probs[argmax_idx])

    # Confidence interval: [low, high] based on prediction set span
    if prediction_set:
        low_idx = min(CLASS_LABELS.index(c) for c in prediction_set)
        high_idx = max(CLASS_LABELS.index(c) for c in prediction_set)
        ci_low = float(probs[low_idx])
        ci_high = float(probs[high_idx])
    else:
        ci_low = peak_class_prob
        ci_high = peak_class_prob

    # Input window provenance
    oldest_frame: PhysicsFrame = buffer[0]
    newest_frame: PhysicsFrame = buffer[-1]

    return ForecastFrame(
        timestamp=datetime.utcnow(),
        horizon_minutes=horizon_minutes,
        prob_quiet=float(probs[0]),
        prob_B=float(probs[1]),
        prob_C=float(probs[2]),
        prob_M=float(probs[3]),
        prob_X=float(probs[4]),
        predicted_class=predicted_class,
        peak_class_prob=peak_class_prob,
        confidence_interval_low=ci_low,
        confidence_interval_high=ci_high,
        prediction_set=prediction_set,
        calibration_ece=ece,
        lead_time_estimate_sec=lead_time_est_sec,
        lead_time_uncertainty_sec=lead_time_unc_sec,
        tcn_weight=tcn_weight,
        moirai_weight=moirai_weight,
        hope_flag_active=hope_flag_active,
        physics_state=physics_state,
        input_window_start=oldest_frame.timestamp,
        input_window_end=newest_frame.timestamp,
        model_version=config.model_version,
        schema_version="1.0",
        pipeline_version=config.pipeline_version,
    )


class ForecastOrchestrator:
    """
    Stateful L4 forecasting entry point.

    Call process(frame, nowcast_event) on each incoming PhysicsFrame.
    Returns a list[ForecastFrame] (one per horizon) when the buffer is full,
    or None during the 10-minute warm-up period.

    HOPE acceleration: when a NowcastEvent with alert_type=HOPE_PRECURSOR is
    provided, the forecast runs immediately regardless of buffer size.
    """

    def __init__(
        self,
        config: Optional[ForecastConfig] = None,
        kafka_producer=None,
    ) -> None:
        self._cfg = config or ForecastConfig()
        self._pub = kafka_producer

        # Sliding buffer — maxlen enforces automatic eviction of oldest frames
        self._buffer: deque[PhysicsFrame] = deque(maxlen=self._cfg.sequence_length)

        # Sub-components
        self._extractor = FeatureExtractor()
        self._moirai = MoiraiForecaster(context_length=self._cfg.moirai_context_length)
        self._ensemble = EnsembleWeighter(self._cfg)
        self._calibrator = MapieCalibrator(self._cfg)
        self._calibrator.load()   # no-op if file missing (passthrough mode)

        # TCN — loaded lazily (needs torch)
        self._tcn = None
        if _TORCH_AVAILABLE:
            from .tcn_model import load_tcn
            self._tcn = load_tcn(self._cfg.tcn_weights_path)

        self._frame_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def process(
        self,
        frame: PhysicsFrame,
        nowcast_event: Optional[NowcastEvent] = None,
    ) -> Optional[List[ForecastFrame]]:
        """
        Ingest one PhysicsFrame.

        Parameters
        ----------
        frame         : current L2 PhysicsFrame
        nowcast_event : L3 event (if any) — HOPE_PRECURSOR triggers immediate forecast

        Returns
        -------
        list[ForecastFrame] : one per horizon when buffer is ≥ sequence_length,
                              or immediately on HOPE_PRECURSOR
        None                : during warm-up (buffer < sequence_length, no HOPE)
        """
        self._buffer.append(frame)
        self._frame_count += 1

        hope_active = (
            nowcast_event is not None
            and nowcast_event.alert_type == "HOPE_PRECURSOR"
        )

        # Warm-up guard: do not forecast until buffer is full (unless HOPE)
        if len(self._buffer) < self._cfg.sequence_length and not hope_active:
            return None

        return self._run_forecast(hope_active=hope_active, nowcast_event=nowcast_event)

    def reset(self) -> None:
        """Clear the sliding buffer. Call between replay events."""
        self._buffer.clear()
        self._frame_count = 0
        logger.debug("ForecastOrchestrator: buffer cleared")

    def close(self) -> None:
        """Cleanup. Nothing persistent to release in current implementation."""
        logger.debug("ForecastOrchestrator: closed")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _run_forecast(
        self,
        hope_active: bool,
        nowcast_event: Optional[NowcastEvent],
    ) -> List[ForecastFrame]:
        """Execute the full inference pipeline and return list[ForecastFrame]."""
        t_start = time.monotonic()

        frames = list(self._buffer)

        # ── Feature extraction ────────────────────────────────────────────────
        t_thermal, t_nonthermal, t_shared = self._extractor.extract(frames)

        # ── TCN forward pass ──────────────────────────────────────────────────
        tcn_probs = self._tcn_infer(t_thermal, t_nonthermal, t_shared)

        # ── Moirai zero-shot forecast ─────────────────────────────────────────
        # Build a numpy feature array for Moirai from the thermal stream
        feature_history = t_thermal.squeeze(0).numpy()  # (T, 8)
        moirai_probs, context_clipped = self._moirai.predict(feature_history)

        # ── Ensemble ──────────────────────────────────────────────────────────
        ensemble_probs, w_tcn, w_moirai = self._ensemble.combine(
            tcn_probs, moirai_probs, context_was_clipped=context_clipped
        )

        # ── Calibration ───────────────────────────────────────────────────────
        cal_probs, pred_set, ece = self._calibrator.calibrate(ensemble_probs)

        # ── Lead time estimate ────────────────────────────────────────────────
        lead_time_sec, lead_time_unc_sec = self._estimate_lead_time(
            frames, nowcast_event, hope_active
        )

        # ── Physics state ─────────────────────────────────────────────────────
        latest = frames[-1]
        physics_state = latest.flare_phase

        # ── Assemble ForecastFrames (one per horizon) ─────────────────────────
        results: List[ForecastFrame] = []
        for horizon in self._cfg.forecast_horizons:
            ff = _build_forecast_frame(
                probs=cal_probs,
                prediction_set=pred_set,
                ece=ece,
                horizon_minutes=horizon,
                tcn_weight=w_tcn,
                moirai_weight=w_moirai,
                buffer=self._buffer,
                hope_flag_active=hope_active,
                physics_state=physics_state,
                config=self._cfg,
                lead_time_est_sec=lead_time_sec,
                lead_time_unc_sec=lead_time_unc_sec,
            )
            results.append(ff)
            if self._pub is not None:
                self._publish(ff)

        elapsed = time.monotonic() - t_start
        logger.info(
            "ForecastOrchestrator: %d frames → class=%s P(M+X)=%.3f "
            "HOPE=%s elapsed=%.2fs",
            len(frames),
            results[0].predicted_class,
            results[0].prob_major_flare(),
            hope_active,
            elapsed,
        )
        if elapsed > 10.0:
            logger.warning(
                "FORECAST_LATENCY_EXCEEDED: %.2fs > 10s threshold", elapsed
            )

        return results

    def _tcn_infer(
        self,
        t_thermal: "torch.Tensor",
        t_nonthermal: "torch.Tensor",
        t_shared: "torch.Tensor",
    ) -> np.ndarray:
        """Run TCN forward pass; return (5,) softmax probabilities."""
        if self._tcn is None or not _TORCH_AVAILABLE:
            # Fallback: uniform distribution if torch unavailable
            return np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)

        import torch
        with torch.no_grad():
            logits = self._tcn(t_thermal, t_nonthermal, t_shared)  # (1, 5)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
        return probs.astype(np.float32)

    @staticmethod
    def _estimate_lead_time(
        frames: List[PhysicsFrame],
        nowcast_event: Optional[NowcastEvent],
        hope_active: bool,
    ) -> tuple[int, int]:
        """
        Estimate lead time to peak in seconds.

        HOPE events carry a pre-computed estimate from the physics engine.
        Otherwise, use empirical estimate: 90s base + T_MK gradient proxy.
        Returns (estimate_sec, uncertainty_sec).
        """
        if hope_active and nowcast_event is not None and nowcast_event.lead_time_sec is not None:
            return nowcast_event.lead_time_sec, 30  # ± 30s uncertainty

        if frames:
            # Use latest HOPE lead time from the most recent frame
            latest = frames[-1]
            if latest.hope_lead_time_sec is not None:
                return latest.hope_lead_time_sec, 60

        # Generic fallback: 90-second base (empirical from Sarwade et al. 2025)
        return 90, 90

    def _publish(self, frame: ForecastFrame) -> None:
        """Publish ForecastFrame to Kafka topic."""
        payload = {
            k: v.isoformat() if isinstance(v, datetime) else v
            for k, v in frame.__dict__.items()
        }
        try:
            self._pub.send(
                self._cfg.kafka_topic_output,
                key=b"forecast",
                value=json.dumps(payload).encode(),
            )
        except Exception as exc:
            logger.error("ForecastOrchestrator: Kafka publish failed: %s", exc)
