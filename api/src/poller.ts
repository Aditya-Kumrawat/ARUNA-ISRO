/**
 * Physics frame simulator + SQLite catalog poller.
 *
 * Every 1 second:  generate synthetic PhysicsFrame → store → emit `physics:frame`
 * Every 5 seconds: refresh SQLite → check new catalog events → emit `nowcast:event`
 * Every 10 seconds: emit `health:update`
 * Every 30 seconds: drift forecast probabilities → emit `forecast:frame`
 */

import type { Server as SocketServer } from 'socket.io'
import { refreshIfNeeded, getNewEventsSince, getMaxRowid } from './db.js'
import {
  getStore, pushPhysicsFrame, pushForecastFrame, setLastCatalogRowid,
  addAlert, makeQuietFrame, getSunLevel,
} from './store.js'
import type { ForecastFrame } from './types.js'

let _timer: ReturnType<typeof setInterval> | null = null
let _tick = 0

export function startPoller(io: SocketServer): void {
  if (_timer) return

  setLastCatalogRowid(getMaxRowid())

  _timer = setInterval(async () => {
    _tick++

    // 1. Physics frame every tick
    const frame = makeQuietFrame()
    pushPhysicsFrame(frame)
    io.emit('physics:frame', frame)

    // 2. Catalog poll every 5 ticks
    if (_tick % 5 === 0) {
      await refreshIfNeeded()
      const store = getStore()
      const { events, maxRowid } = getNewEventsSince(store.lastCatalogRowid)
      if (events.length > 0) {
        setLastCatalogRowid(maxRowid)
        for (const ev of events) {
          addAlert(ev)
          io.emit('nowcast:event', ev)
        }
      }
    }

    // 3. Health every 10 ticks
    if (_tick % 10 === 0) {
      const store = getStore()
      io.emit('health:update', {
        status: 'HEALTHY',
        level: getSunLevel(),
        uptime_sec: Math.round((Date.now() - store.startedAt) / 1000),
        replay_active: store.replayActive,
        timestamp: new Date().toISOString(),
      })
    }

    // 4. Forecast drift every 30 ticks
    if (_tick % 30 === 0) {
      const drifted = makeDriftedForecast()
      pushForecastFrame(drifted)
      io.emit('forecast:frame', drifted)
    }
  }, 1000)
}

export function stopPoller(): void {
  if (_timer) {
    clearInterval(_timer)
    _timer = null
  }
}

function makeDriftedForecast(): ForecastFrame {
  const store = getStore()
  const prev = store.forecast
  const drift = (v: number, s: number, lo: number, hi: number) =>
    Math.max(lo, Math.min(hi, v + (Math.random() - 0.5) * s))

  const q = drift(prev.prob_quiet, 0.02, 0.50, 0.90)
  const b = drift(prev.prob_B, 0.01, 0.05, 0.35)
  const c = drift(prev.prob_C, 0.005, 0.01, 0.15)
  const m = drift(prev.prob_M, 0.002, 0.001, 0.05)
  const x = drift(prev.prob_X, 0.001, 0.0001, 0.02)
  const total = q + b + c + m + x
  const n = (v: number) => Math.round((v / total) * 1000) / 1000
  const [pQ, pB, pC, pM, pX] = [n(q), n(b), n(c), n(m), n(x)]
  const probs = [pQ, pB, pC, pM, pX]
  const maxIdx = probs.indexOf(Math.max(...probs))
  const labels = ['quiet', 'B', 'C', 'M', 'X']

  return {
    ...prev,
    forecast_id: crypto.randomUUID(),
    timestamp: new Date().toISOString(),
    prob_quiet: pQ, prob_B: pB, prob_C: pC, prob_M: pM, prob_X: pX,
    peak_class_prob: probs[maxIdx],
    predicted_class: labels[maxIdx],
    confidence_interval_low: Math.max(0.50, prev.confidence_interval_low + (Math.random() - 0.5) * 0.015),
    confidence_interval_high: Math.min(0.97, prev.confidence_interval_high + (Math.random() - 0.5) * 0.015),
    hope_flag_active: store.physics.hope_flag,
    physics_state: store.physics.flare_phase,
  }
}
