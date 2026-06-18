"""HOPE precursor indicator evaluator.

HOPE fires when five conditions all hold: T rising, EM flat, HEL1OS quiet,
TNPSV in precursor quadrant, and DNI above threshold.
"""
from __future__ import annotations
import numpy as np
from .models import HOPEResult, PhysicsConfig


class HOPEEvaluator:
    """Evaluate five-condition HOPE precursor indicator.

    HOPE (Hierarchical Onset Prediction Engine) precursor fires ≥90 seconds
    before X-class flare onset in optimal cases. All five conditions must
    be true simultaneously for the flag to fire.
    """

    def evaluate(
        self,
        T: float,
        dT_dt: float,
        EM: float,
        dEM_dt: float,
        DNI: float,
        tnpsv_mag: float,
        tnpsv_angle: float,
        hel_flux: float,
        hel_quiet_baseline: float,
        streak_sec: float,
        config: PhysicsConfig,
    ) -> HOPEResult:
        """Evaluate HOPE precursor conditions.

        Args:
            T: Plasma temperature (MK).
            dT_dt: Temperature rate of change (MK/s).
            EM: Emission measure (cm^-3).
            dEM_dt: Emission measure rate of change (cm^-3/s).
            DNI: Dynamic Neupert Integral (dimensionless).
            tnpsv_mag: TNPSV velocity magnitude (dimensionless).
            tnpsv_angle: TNPSV angle in T-EM phase space (radians).
            hel_flux: HEL1OS 40–100 keV flux (counts/s).
            hel_quiet_baseline: HEL1OS quiet baseline (counts/s).
            streak_sec: Elapsed time since flare onset (seconds).
            config: Physics configuration with thresholds.

        Returns:
            HOPEResult with hope_flag, reason, and lead_time_sec.
        """
        # Condition 1: Temperature rising AND minimum streak duration
        cond_T = dT_dt > 0 and streak_sec >= 3.0

        # Condition 2: Emission measure flat (below rise threshold)
        cond_EM = abs(dEM_dt) < config.hope_em_rise_threshold

        # Condition 3: HEL1OS quiet (flux below baseline * multiplier)
        cond_HEL = hel_flux < hel_quiet_baseline * config.hope_hel_quiet_multiplier

        # Condition 4: TNPSV in precursor quadrant (60°–120°, with sufficient speed)
        cond_TNPSV = (
            tnpsv_mag > config.hope_tnpsv_speed_threshold
            and config.hope_tnpsv_angle_min < tnpsv_angle < config.hope_tnpsv_angle_max
        )

        # Condition 5: DNI above threshold
        cond_DNI = DNI > 0.2

        # Fire only when all five conditions are true
        hope_flag = all([cond_T, cond_EM, cond_HEL, cond_TNPSV, cond_DNI])

        # Log which conditions failed
        failed = [
            n
            for n, v in [
                ("T_rising", cond_T),
                ("EM_flat", cond_EM),
                ("HEL_quiet", cond_HEL),
                ("TNPSV", cond_TNPSV),
                ("DNI", cond_DNI),
            ]
            if not v
        ]

        # Build detailed reason string
        reason = (
            f"dT/dt={dT_dt:.3f} streak={streak_sec:.1f}s | "
            f"dEM/dt={dEM_dt:.2e} | HEL={hel_flux:.1f} base={hel_quiet_baseline:.1f} | "
            f"TNPSV mag={tnpsv_mag:.3f} ang={np.degrees(tnpsv_angle):.1f}° | "
            f"DNI={DNI:.2f}"
        )
        if failed:
            reason += f" | FAILED: {', '.join(failed)}"

        # Lead time scales with streak: base 90s + 2x elapsed time
        lead_time_sec = int(90 + streak_sec * 2)

        return HOPEResult(hope_flag=hope_flag, reason=reason, lead_time_sec=lead_time_sec)
