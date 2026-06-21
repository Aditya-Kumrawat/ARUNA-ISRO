import type { FastifyPluginAsync } from 'fastify'
import { getStore } from '../store.js'

interface GaganyaanQuery { days?: string }

const gaganyaan: FastifyPluginAsync = async (fastify) => {
  fastify.get<{ Querystring: GaganyaanQuery }>('/gaganyaan/window', async (req) => {
    const days = Math.min(Math.max(1, Number(req.query.days ?? 7)), 30)
    const store = getStore()
    const { threat, forecast } = store

    // Build advisory window from forecast probabilities
    const mxProb = forecast.prob_M + forecast.prob_X
    const advisory = mxProb > 0.3
      ? 'CAUTION — Elevated M/X probability; EVA window under review'
      : mxProb > 0.1
      ? 'MONITOR — Low M/X probability; standard precautions apply'
      : threat.goes_class === 'X'
      ? 'HOLD — Active X-class flare; EVA not recommended'
      : 'SAFE — Favourable space weather conditions for EVA'

    const windows = buildEvaWindows(days, mxProb)

    return {
      advisory,
      days_requested: days,
      current_threat: threat.gaganyaan_advisory,
      mx_probability_60min: Math.round(mxProb * 1000) / 1000,
      eva_windows: windows,
      generated_at: new Date().toISOString(),
    }
  })
}

function buildEvaWindows(days: number, mxProb: number) {
  const windows = []
  const base = Date.now()
  for (let d = 0; d < days; d++) {
    const dayStart = new Date(base + d * 86_400_000)
    dayStart.setUTCHours(6, 0, 0, 0)
    const dayEnd = new Date(dayStart.getTime() + 43_200_000) // 12h window

    // First day inherits live probability; future days assume quiet (decaying uncertainty)
    const decay = d === 0 ? 1.0 : Math.exp(-0.3 * d)
    const dayMxProb = mxProb * decay
    const status = dayMxProb > 0.3 ? 'CAUTION' : dayMxProb > 0.1 ? 'MONITOR' : 'SAFE'

    windows.push({
      date: dayStart.toISOString().slice(0, 10),
      window_start: dayStart.toISOString(),
      window_end: dayEnd.toISOString(),
      mx_probability: Math.round(dayMxProb * 1000) / 1000,
      status,
    })
  }
  return windows
}

export default gaganyaan
