// Core data types mirroring Python dataclasses exchanged on Kafka / SQLite

export type FlarePhase = 'QUIET' | 'HOPE' | 'IMPULSIVE' | 'GRADUAL' | 'DECAY'
export type AlertType = 'HOPE_PRECURSOR' | 'FLARE_ONSET' | 'FLARE_PEAK' | 'FLARE_END'
export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'EXTREME'

export interface PhysicsFrame {
  frame_id: string
  timestamp: string
  T_MK: number
  T_uncertainty_MK: number
  EM_cm3: number
  EM_uncertainty: number
  DNI: number
  tnpsv_magnitude: number
  tnpsv_angle_rad: number
  SHEI: number
  dSHEI_dt: number
  HFD: number | null
  QPP_period_sec: number | null
  QPP_amplitude: number | null
  FeKa_flux: number
  flare_phase: FlarePhase
  hope_flag: boolean
  hope_lead_time_sec: number | null
  hope_reason: string | null
  fit_chi_squared: number
  solexs_3_6kev: number
  solexs_6_10kev: number
  hel1os_10_25kev: number
  hel1os_25_50kev: number
  hel1os_50_70kev: number
  pipeline_version: string
}

export interface ForecastFrame {
  forecast_id: string
  timestamp: string
  horizon_minutes: number
  prob_quiet: number
  prob_B: number
  prob_C: number
  prob_M: number
  prob_X: number
  predicted_class: string
  peak_class_prob: number
  confidence_interval_low: number
  confidence_interval_high: number
  prediction_set: string[]
  calibration_ece: number
  lead_time_estimate_sec: number
  tcn_weight: number
  moirai_weight: number
  hope_flag_active: boolean
  physics_state: string
}

export interface NavicRegionRisk {
  level: RiskLevel
  probability: number
}

export interface ThreatOutput {
  navic_risk: Record<string, NavicRegionRisk>
  hf_blackout_risk: RiskLevel
  hf_blackout_duration_min: number
  radiation_risk: RiskLevel
  gaganyaan_advisory: string
  safing_queue: Array<{ name: string; orbit_type: string; priority: number }>
  goes_class: string
  timestamp: string
}

// Matches master_flare_catalog SQLite table
export interface FlareEvent {
  event_id: string
  start_time: string | null
  hope_time: string | null
  onset_time: string | null
  peak_time: string | null
  end_time: string | null
  goes_class: string
  goes_subclass: number
  peak_flux_solexs: number
  peak_flux_hel1os: number
  peak_T_MK: number
  peak_EM_cm3: number
  peak_DNI: number
  hope_lead_time_sec: number | null
  hope_fired: boolean
  instruments: string[]
  detection_method: string
  confidence: number
  flare_type: string
  pipeline_version: string
  provenance_hash: string
}

export interface PipelineStageHealth {
  status: 'OK' | 'DEGRADED' | 'DOWN'
  last_update_sec: number
  message?: string
}

export interface HealthResponse {
  status: 'HEALTHY' | 'DEGRADED' | 'CRITICAL'
  timestamp: string
  uptime_sec: number
  version: string
  db_connected: boolean
  stages: Record<string, PipelineStageHealth>
}
