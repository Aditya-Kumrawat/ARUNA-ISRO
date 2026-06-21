import type { FastifyPluginAsync } from 'fastify'
import { getValidationStats } from '../db.js'

const validation: FastifyPluginAsync = async (fastify) => {
  fastify.get('/validation/metrics', async () => {
    const stats = getValidationStats()

    // Derived metrics
    const hope_pct = stats.total_events > 0
      ? (stats.hope_fired_count / stats.total_events) * 100
      : 0

    const dual_pct = stats.total_events > 0
      ? (stats.dual_trigger_count / stats.total_events) * 100
      : 0

    return {
      timestamp: new Date().toISOString(),
      catalog_stats: stats,
      derived: {
        hope_precursor_rate_pct: Math.round(hope_pct * 10) / 10,
        dual_trigger_rate_pct: Math.round(dual_pct * 10) / 10,
        // Target from PRD §28: HOPE must fire ≥90s before X6.3 peak
        hope_lead_time_target_sec: 90,
        hope_lead_time_achieved: stats.avg_lead_time_sec != null && stats.avg_lead_time_sec >= 90,
      },
    }
  })
}

export default validation
