import type { PhysicsFrame, ForecastFrame, ThreatOutput, FlareEvent, RiskLevel } from './types.js'

const MAX_PHYSICS = 600  // 10 min @ 1 Hz
const MAX_FORECAST = 100

// --- Synthetic quiet-sun initial state ---

function noise(scale: number): number {
  return (Math.random() - 0.5) * 2 * scale
}

export function makeQuietFrame(ts?: string): PhysicsFrame {
  return {
    frame_id: crypto.randomUUID(),
    timestamp: ts ?? new Date().toISOString(),
    T_MK: 8.0 + noise(0.15),
    T_uncertainty_MK: 0.5,
    EM_cm3: 1e47 * (1 + noise(0.015)),
    EM_uncertainty: 5e45,
    DNI: Math.max(0.05, 0.12 + noise(0.02)),
    tnpsv_magnitude: Math.abs(0.04 + noise(0.01)),
    tnpsv_angle_rad: 0.3 + noise(0.04),
    SHEI: -1.8 + noise(0.04),
    dSHEI_dt: noise(0.001),
    HFD: 1.4 + noise(0.02),
    QPP_period_sec: null,
    QPP_amplitude: null,
    FeKa_flux: 0.0,
    flare_phase: 'QUIET',
    hope_flag: false,
    hope_lead_time_sec: null,
    hope_reason: null,
    fit_chi_squared: 1.05 + Math.abs(noise(0.04)),
    solexs_3_6kev: 500 + noise(8),
    solexs_6_10kev: 300 + noise(6),
    hel1os_10_25kev: 100 + noise(3),
    hel1os_25_50kev: 50 + noise(2),
    hel1os_50_70kev: 20 + noise(1),
    pipeline_version: '2.0.0',
  }
}

function makeQuietForecast(): ForecastFrame {
  return {
    forecast_id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    horizon_minutes: 60,
    prob_quiet: 0.75,
    prob_B: 0.18,
    prob_C: 0.055,
    prob_M: 0.01,
    prob_X: 0.005,
    predicted_class: 'quiet',
    peak_class_prob: 0.75,
    confidence_interval_low: 0.66,
    confidence_interval_high: 0.84,
    prediction_set: ['quiet', 'B'],
    calibration_ece: 0.032,
    lead_time_estimate_sec: 3600,
    tcn_weight: 0.6,
    moirai_weight: 0.4,
    hope_flag_active: false,
    physics_state: 'QUIET',
  }
}

function makeQuietThreat(): ThreatOutput {
  const regions = ['India', 'Sri_Lanka', 'Nepal', 'Bangladesh', 'Pakistan', 'Afghanistan', 'Myanmar', 'Tibet']
  const navic_risk: Record<string, { level: RiskLevel; probability: number }> = {}
  for (const r of regions) navic_risk[r] = { level: 'LOW', probability: 0.04 }
  return {
    navic_risk,
    hf_blackout_risk: 'LOW',
    hf_blackout_duration_min: 0,
    radiation_risk: 'LOW',
    gaganyaan_advisory: 'SAFE — No significant solar activity expected in next 24h',
    safing_queue: [],
    goes_class: 'B',
    timestamp: new Date().toISOString(),
  }
}

// --- Store ---

export interface MissionEvent {
  id: string;
  timestamp: string;
  event_type: string;
  label: string;
  detail: string;
  severity: 'INFO' | 'WARNING' | 'CRITICAL';
}

export interface DecisionSnapshot {
  recommendation: string;
  confidence: number;
  actions: string[];
  affected_assets: string[];
  mission_brief: string;
  valid_until: string;
  reasoning: {
    physics_evidence: string[];
    trigger_chain: string[];
    confidence_contributors: Record<string, number>;
    analog_match: string;
    physics_consistency: string;
    mission_impact_summary: string;
  };
  updated_at: string;
}

interface AppStore {
  physics: PhysicsFrame
  physicsHistory: PhysicsFrame[]
  forecast: ForecastFrame
  forecastHistory: ForecastFrame[]
  threat: ThreatOutput
  bulletin: string
  bulletinAt: string
  startedAt: number
  lastCatalogRowid: number
  recentAlerts: FlareEvent[]
  replayActive: boolean
  replayEventId: string | null
  decisionFrame: DecisionSnapshot | null
  missionEvents: MissionEvent[]
}

const store: AppStore = {
  physics: makeQuietFrame(),
  physicsHistory: [],
  forecast: makeQuietForecast(),
  forecastHistory: [],
  threat: makeQuietThreat(),
  bulletin: 'Solar activity is at background levels. No significant events expected.',
  bulletinAt: new Date().toISOString(),
  startedAt: Date.now(),
  lastCatalogRowid: 0,
  recentAlerts: [],
  replayActive: false,
  replayEventId: null,
  decisionFrame: null,
  missionEvents: [],
}

export function getStore(): Readonly<AppStore> {
  return store
}

export function setDecisionFrame(d: DecisionSnapshot): void {
  store.decisionFrame = d
}

export function pushMissionEvent(event: Omit<MissionEvent, 'id'>): void {
  const full: MissionEvent = { id: crypto.randomUUID(), ...event };
  store.missionEvents = [full, ...store.missionEvents].slice(0, 500);
}

export function pushPhysicsFrame(frame: PhysicsFrame): void {
  store.physics = frame
  store.physicsHistory.push(frame)
  if (store.physicsHistory.length > MAX_PHYSICS) store.physicsHistory.shift()
}

export function pushForecastFrame(frame: ForecastFrame): void {
  store.forecast = frame
  store.forecastHistory.push(frame)
  if (store.forecastHistory.length > MAX_FORECAST) store.forecastHistory.shift()
}

export function setThreat(t: ThreatOutput): void {
  store.threat = t
}

export function setBulletin(text: string): void {
  store.bulletin = text
  store.bulletinAt = new Date().toISOString()
}

export function setLastCatalogRowid(r: number): void {
  store.lastCatalogRowid = r
}

export function addAlert(event: FlareEvent): void {
  store.recentAlerts.unshift(event)
  if (store.recentAlerts.length > 50) store.recentAlerts.pop()
}

export function setReplayState(active: boolean, eventId: string | null): void {
  store.replayActive = active
  store.replayEventId = eventId
}

export function getPhysicsHistory(minutes: number): PhysicsFrame[] {
  const cutoff = new Date(Date.now() - minutes * 60_000).toISOString()
  return store.physicsHistory.filter(f => f.timestamp >= cutoff)
}

export function getSunLevel(): 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED' {
  const { flare_phase, hope_flag } = store.physics
  const goesClass = store.threat.goes_class
  if (goesClass === 'X') return 'RED'
  if (goesClass === 'M') return 'ORANGE'
  if (flare_phase === 'IMPULSIVE' || hope_flag) return 'YELLOW'
  return 'GREEN'
}
