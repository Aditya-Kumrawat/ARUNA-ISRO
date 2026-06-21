import type { FastifyPluginAsync } from 'fastify'
import { getStore } from '../store.js'

const threat: FastifyPluginAsync = async (fastify) => {
  fastify.get('/threat/current', async () => {
    return getStore().threat
  })

  fastify.get('/threat/navic', async () => {
    const { threat } = getStore()
    return {
      timestamp: threat.timestamp,
      goes_class: threat.goes_class,
      navic_risk: threat.navic_risk,
      hf_blackout_risk: threat.hf_blackout_risk,
      hf_blackout_duration_min: threat.hf_blackout_duration_min,
      summary: buildNavicSummary(threat.navic_risk),
    }
  })
}

function buildNavicSummary(navic: Record<string, { level: string; probability: number }>) {
  const levels = Object.values(navic).map(v => v.level)
  const counts = { LOW: 0, MODERATE: 0, HIGH: 0, EXTREME: 0 }
  for (const l of levels) {
    const key = l as keyof typeof counts
    if (key in counts) counts[key]++
  }
  const maxLevel = levels.includes('EXTREME') ? 'EXTREME'
    : levels.includes('HIGH') ? 'HIGH'
    : levels.includes('MODERATE') ? 'MODERATE'
    : 'LOW'
  return { max_level: maxLevel, region_counts: counts }
}

export default threat
