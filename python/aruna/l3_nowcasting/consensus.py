from __future__ import annotations
import logging
import uuid
from datetime import datetime
from typing import List, Optional

from aruna.l2_physics.models import PhysicsFrame
from aruna.models.nowcast import NowcastEvent
from .goes_class import GOESClassifier
from .models import NowcastConfig

logger = logging.getLogger(__name__)


def _isnan(v: float) -> bool:
    return v != v


class ConsensusEngine:
    """4-state FSM: QUIET → HOPE → ONSET → PEAK → QUIET. Not thread-safe."""

    def __init__(self, config: Optional[NowcastConfig] = None) -> None:
        self._cfg = config or NowcastConfig()
        self._classifier = GOESClassifier(self._cfg.solexs_to_wm2_factor)
        self._state: str = "QUIET"
        self._sol_trigger_time: Optional[datetime] = None
        self._hel_trigger_time: Optional[datetime] = None
        self._onset_time: Optional[datetime] = None
        self._hope_time: Optional[datetime] = None
        self._onset_wall_time: Optional[datetime] = None
        self._peak_flux: float = 0.0
        self._peak_flux_sol: float = 0.0
        self._peak_flux_hel: float = 0.0
        self._peak_T_MK: float = 0.0
        self._peak_EM_cm3: float = 0.0
        self._peak_DNI: float = 0.0
        self._declining_count: int = 0
        self._prev_flux: float = 0.0
        self._current_event_id: Optional[str] = None
        self._sol_only_fired: bool = False

    # ── public ──────────────────────────────────────────────────────────────

    def ingest(
        self,
        sol_triggered: bool,
        hel_triggered: bool,
        hope_flag: bool,
        hope_reason: Optional[str],
        physics_frame: PhysicsFrame,
        flux_solexs: float,
        flux_hel1os: float,
        kp_index: Optional[float] = None,
    ) -> Optional[NowcastEvent]:
        ts = physics_frame.timestamp

        # SEP event override — fires immediately regardless of FSM state
        if (
            kp_index is not None
            and kp_index > self._cfg.sep_kp_threshold
            and not hope_flag
            and (sol_triggered or hel_triggered)
        ):
            return self._make_event(
                "FLARE_ONSET",
                ts,
                physics_frame,
                flux_solexs,
                flux_hel1os,
                detection_method="SEP_EVENT",
                confidence=0.3,
                instruments=(["SOLEXS"] if sol_triggered else ["HEL1OS"]),
                flare_type="STANDARD",
            )

        if self._state == "QUIET":
            return self._handle_quiet(
                sol_triggered,
                hel_triggered,
                hope_flag,
                hope_reason,
                physics_frame,
                flux_solexs,
                flux_hel1os,
                ts,
            )
        if self._state == "HOPE":
            return self._handle_hope(
                sol_triggered,
                hel_triggered,
                physics_frame,
                flux_solexs,
                flux_hel1os,
                ts,
            )
        if self._state == "ONSET":
            return self._handle_onset(physics_frame, flux_solexs, flux_hel1os, ts)
        if self._state == "PEAK":
            return self._handle_peak(
                sol_triggered,
                hel_triggered,
                physics_frame,
                flux_solexs,
                flux_hel1os,
                ts,
            )
        return None

    @property
    def state(self) -> str:
        return self._state

    # ── state handlers ───────────────────────────────────────────────────────

    def _handle_quiet(
        self,
        sol_t: bool,
        hel_t: bool,
        hope_flag: bool,
        hope_reason: Optional[str],
        frame: PhysicsFrame,
        flux_sol: float,
        flux_hel: float,
        ts: datetime,
    ) -> Optional[NowcastEvent]:
        if sol_t:
            self._sol_trigger_time = ts
        if hel_t:
            self._hel_trigger_time = ts
        if self._dual_trigger_met(ts):
            return self._transition_to_onset(frame, flux_sol, flux_hel, ts)
        # SOLEXS_ONLY degraded mode: emit advisory when sol_triggered but hel_triggered=False
        if sol_t and not hel_t and not self._sol_only_fired:
            self._sol_only_fired = True
            return self._make_event(
                "FLARE_ONSET",
                ts,
                frame,
                flux_sol,
                flux_hel,
                detection_method="SOLEXS_ONLY",
                confidence=0.5,
                instruments=["SOLEXS"],
                flare_type="STANDARD",
            )
        if hope_flag and self._hope_time is None:
            self._hope_time = ts
            self._state = "HOPE"
            self._current_event_id = str(uuid.uuid4())
            return self._make_event(
                "HOPE_PRECURSOR",
                ts,
                frame,
                flux_sol,
                flux_hel,
                detection_method="HOPE",
                confidence=0.6,
                instruments=["SOLEXS", "HEL1OS"],
                flare_type="STANDARD",
                hope_reason=hope_reason,
                lead_time_sec=frame.hope_lead_time_sec,
                event_id=self._current_event_id,
            )
        return None

    def _handle_hope(
        self,
        sol_t: bool,
        hel_t: bool,
        frame: PhysicsFrame,
        flux_sol: float,
        flux_hel: float,
        ts: datetime,
    ) -> Optional[NowcastEvent]:
        if sol_t:
            self._sol_trigger_time = ts
        if hel_t:
            self._hel_trigger_time = ts
        if self._dual_trigger_met(ts):
            return self._transition_to_onset(frame, flux_sol, flux_hel, ts)
        # SOLEXS_ONLY degraded mode: emit advisory when sol_triggered but hel_triggered=False
        if sol_t and not hel_t and not self._sol_only_fired:
            self._sol_only_fired = True
            return self._make_event(
                "FLARE_ONSET",
                ts,
                frame,
                flux_sol,
                flux_hel,
                detection_method="SOLEXS_ONLY",
                confidence=0.5,
                instruments=["SOLEXS"],
                flare_type="STANDARD",
            )
        return None

    def _handle_onset(
        self,
        frame: PhysicsFrame,
        flux_sol: float,
        flux_hel: float,
        ts: datetime,
    ) -> Optional[NowcastEvent]:
        combined = flux_sol + flux_hel
        if combined > self._peak_flux:
            self._peak_flux = combined
            self._peak_flux_sol = flux_sol
            self._peak_flux_hel = flux_hel
            self._peak_T_MK = (
                frame.T_MK if not _isnan(frame.T_MK) else self._peak_T_MK
            )
            self._peak_EM_cm3 = (
                frame.EM_cm3 if not _isnan(frame.EM_cm3) else self._peak_EM_cm3
            )
            self._peak_DNI = frame.DNI
        if combined < self._prev_flux:
            self._declining_count += 1
        else:
            self._declining_count = 0
        self._prev_flux = combined
        if self._declining_count >= 3:
            self._state = "PEAK"
            return self._make_event(
                "FLARE_PEAK",
                ts,
                frame,
                self._peak_flux_sol,
                self._peak_flux_hel,
                detection_method="DUAL_TRIGGER",
                confidence=0.9,
                instruments=["SOLEXS", "HEL1OS"],
                flare_type="STANDARD",
                event_id=self._current_event_id,
            )
        return None

    def _handle_peak(
        self,
        sol_t: bool,
        hel_t: bool,
        frame: PhysicsFrame,
        flux_sol: float,
        flux_hel: float,
        ts: datetime,
    ) -> Optional[NowcastEvent]:
        combined = flux_sol + flux_hel
        # Long-duration guard
        if self._onset_wall_time is not None and (
            (ts - self._onset_wall_time).total_seconds()
            > self._cfg.long_duration_lock_sec
        ):
            return self._transition_to_end(frame, flux_sol, flux_hel, ts)
        # Normal end: flux below configured fraction of peak
        if (
            self._peak_flux > 0
            and combined < self._peak_flux * self._cfg.peak_decay_threshold
        ):
            return self._transition_to_end(frame, flux_sol, flux_hel, ts)
        # Overlapping flare: new dual trigger while already in PEAK
        if sol_t:
            self._sol_trigger_time = ts
        if hel_t:
            self._hel_trigger_time = ts
        if self._dual_trigger_met(ts):
            new_id = str(uuid.uuid4())
            self._current_event_id = new_id
            self._onset_wall_time = ts
            self._peak_flux = 0.0
            self._declining_count = 0
            self._prev_flux = combined
            self._state = "ONSET"
            self._hope_time = None
            return self._make_event(
                "FLARE_ONSET",
                ts,
                frame,
                flux_sol,
                flux_hel,
                detection_method="DUAL_TRIGGER",
                confidence=0.85,
                instruments=["SOLEXS", "HEL1OS"],
                flare_type="OVERLAPPING",
                event_id=new_id,
            )
        return None

    # ── helpers ──────────────────────────────────────────────────────────────

    def _dual_trigger_met(self, ts: datetime) -> bool:
        if self._sol_trigger_time is None or self._hel_trigger_time is None:
            return False
        return (
            abs(
                (self._sol_trigger_time - self._hel_trigger_time).total_seconds()
            )
            <= self._cfg.dual_trigger_window_sec
        )

    def _transition_to_onset(
        self,
        frame: PhysicsFrame,
        flux_sol: float,
        flux_hel: float,
        ts: datetime,
    ) -> NowcastEvent:
        self._state = "ONSET"
        self._onset_wall_time = ts
        self._onset_time = ts
        self._declining_count = 0
        self._prev_flux = flux_sol + flux_hel
        self._peak_flux = 0.0
        # Reset trigger times so PEAK state requires fresh dual triggers for overlapping detection
        self._sol_trigger_time = None
        self._hel_trigger_time = None
        self._sol_only_fired = False
        if self._current_event_id is None:
            self._current_event_id = str(uuid.uuid4())
        lead_time = (
            int((ts - self._hope_time).total_seconds())
            if self._hope_time is not None
            else None
        )
        return self._make_event(
            "FLARE_ONSET",
            ts,
            frame,
            flux_sol,
            flux_hel,
            detection_method="DUAL_TRIGGER",
            confidence=0.85,
            instruments=["SOLEXS", "HEL1OS"],
            flare_type="STANDARD",
            lead_time_sec=lead_time,
            event_id=self._current_event_id,
        )

    def _transition_to_end(
        self,
        frame: PhysicsFrame,
        flux_sol: float,
        flux_hel: float,
        ts: datetime,
    ) -> NowcastEvent:
        self._state = "QUIET"
        self._sol_trigger_time = None
        self._hel_trigger_time = None
        self._hope_time = None
        self._peak_flux = 0.0
        self._onset_wall_time = None
        self._sol_only_fired = False
        event = self._make_event(
            "FLARE_END",
            ts,
            frame,
            flux_sol,
            flux_hel,
            detection_method="DUAL_TRIGGER",
            confidence=0.9,
            instruments=["SOLEXS", "HEL1OS"],
            flare_type="STANDARD",
            event_id=self._current_event_id,
        )
        self._current_event_id = None
        return event

    def _make_event(
        self,
        alert_type: str,
        ts: datetime,
        frame: PhysicsFrame,
        flux_sol: float,
        flux_hel: float,
        detection_method: str,
        confidence: float,
        instruments: List[str],
        flare_type: str,
        hope_reason: Optional[str] = None,
        lead_time_sec: Optional[int] = None,
        event_id: Optional[str] = None,
    ) -> NowcastEvent:
        goes_class, goes_subclass = self._classifier.classify(flux_sol)
        # SOLEXS_ONLY is always advisory — do not escalate confidence
        if (
            self._classifier.is_major_xclass(goes_class, goes_subclass)
            and self._cfg.x3_auto_critical
            and detection_method != "SOLEXS_ONLY"
        ):
            confidence = 1.0
        event = NowcastEvent(
            alert_type=alert_type,
            timestamp=ts,
            goes_class=goes_class,
            goes_subclass=goes_subclass,
            confidence=confidence,
            instruments=instruments,
            physics_frame_id=frame.frame_id,
            detection_method=detection_method,
            flare_type=flare_type,
            lead_time_sec=lead_time_sec,
            hope_reason=hope_reason,
            provenance_hash=frame.provenance_hash,
            pipeline_version=self._cfg.pipeline_version,
        )
        if event_id:
            event.event_id = event_id
        return event
