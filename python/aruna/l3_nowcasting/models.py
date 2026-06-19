from __future__ import annotations
from dataclasses import dataclass


@dataclass
class NowcastConfig:
    # Baseline
    baseline_window_sec: int = 10800
    baseline_percentile: float = 85.0
    baseline_recompute_interval: int = 60
    trigger_sigma_multiplier: float = 3.0

    # PELT
    pelt_penalty: float = 1.5
    pelt_min_size: int = 10
    pelt_model: str = "rbf"
    pelt_flux_window: int = 600

    # Consensus / state machine
    dual_trigger_window_sec: int = 180
    peak_decay_threshold: float = 0.5
    long_duration_lock_sec: int = 21600

    # GOES calibration
    solexs_to_wm2_factor: float = 1.0e-8

    # Escalation
    x3_auto_critical: bool = True
    sep_kp_threshold: int = 6

    # Kafka
    kafka_topic_input: str = "aruna.physics.frames"
    kafka_topic_output: str = "aruna.nowcast.events"

    # Database
    db_url: str = "sqlite:///aruna_catalog.db"
    pipeline_version: str = "2.0.0"
