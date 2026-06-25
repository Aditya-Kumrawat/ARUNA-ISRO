from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from aruna.models.physics import PhysicsFrame

# Physical bounds for a solar flare (conservative)
_T_MAX_MK = 50.0          # hottest known solar flares ~50 MK
_T_MIN_MK = 0.5           # below this is pre-flare quiet sun
_EM_MAX = 1e52            # cm^-3, upper bound
_CHI_SUSPICIOUS_LOW = 0.1  # chi² < 0.1 is suspiciously perfect
_CHI_MAX = 10.0           # chi² > 10 means poor fit


@dataclass
class IntegrityFlag:
    timestamp: datetime
    anomaly_type: str
    severity: str          # INFO | WARNING | CRITICAL
    affected_stage: str
    explanation: str
    recommended_action: str


class IntegrityMonitor:
    """
    Observes PhysicsFrame for anomalies. Never modifies pipeline outputs.
    Emits IntegrityFlag list — empty list means clean frame.
    """

    def observe(self, frame: PhysicsFrame) -> List[IntegrityFlag]:
        flags: List[IntegrityFlag] = []
        now = frame.timestamp

        # 1. Temperature bounds
        if frame.T_MK > _T_MAX_MK:
            flags.append(IntegrityFlag(
                timestamp=now,
                anomaly_type="TEMPERATURE_OUT_OF_BOUNDS",
                severity="CRITICAL",
                affected_stage="L2_PHYSICS",
                explanation=f"Temperature {frame.T_MK:.1f} MK exceeds physical maximum {_T_MAX_MK} MK for solar flares.",
                recommended_action="Inspect isothermal fit parameters; check SoLEXS energy calibration.",
            ))
        elif frame.T_MK < _T_MIN_MK:
            flags.append(IntegrityFlag(
                timestamp=now,
                anomaly_type="TEMPERATURE_BELOW_MINIMUM",
                severity="WARNING",
                affected_stage="L2_PHYSICS",
                explanation=f"Temperature {frame.T_MK:.2f} MK is below quiet-sun minimum {_T_MIN_MK} MK.",
                recommended_action="Check dead-time correction and background subtraction.",
            ))

        # 2. Emission measure bounds
        if frame.EM_cm3 > _EM_MAX:
            flags.append(IntegrityFlag(
                timestamp=now,
                anomaly_type="EMISSION_MEASURE_OUT_OF_BOUNDS",
                severity="CRITICAL",
                affected_stage="L2_PHYSICS",
                explanation=f"Emission measure {frame.EM_cm3:.2e} cm⁻³ exceeds physical maximum.",
                recommended_action="Inspect isothermal fit convergence.",
            ))

        # 3. Chi-squared fit quality
        if 0 < frame.fit_chi_squared < _CHI_SUSPICIOUS_LOW:
            flags.append(IntegrityFlag(
                timestamp=now,
                anomaly_type="FIT_SUSPICIOUSLY_PERFECT",
                severity="INFO",
                affected_stage="L2_PHYSICS",
                explanation=f"χ²={frame.fit_chi_squared:.3f} is suspiciously low — may indicate overfitting.",
                recommended_action="Verify fit converged on physical solution.",
            ))
        elif frame.fit_chi_squared > _CHI_MAX:
            flags.append(IntegrityFlag(
                timestamp=now,
                anomaly_type="FIT_QUALITY_POOR",
                severity="WARNING",
                affected_stage="L2_PHYSICS",
                explanation=f"χ²={frame.fit_chi_squared:.1f} exceeds threshold {_CHI_MAX} — isothermal fit is unreliable.",
                recommended_action="Consider multi-temperature fit or check energy range.",
            ))

        # 4. HOPE flag / phase inconsistency
        if frame.hope_flag and frame.flare_phase == "QUIET":
            flags.append(IntegrityFlag(
                timestamp=now,
                anomaly_type="HOPE_PHASE_INCONSISTENCY",
                severity="WARNING",
                affected_stage="L2_PHYSICS",
                explanation="HOPE flag is True but flare_phase is QUIET — these states are mutually exclusive.",
                recommended_action="Check HOPE detection logic in L2 engine.",
            ))

        return flags
