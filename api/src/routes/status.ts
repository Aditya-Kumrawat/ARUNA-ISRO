import type { FastifyPluginAsync } from 'fastify'
import { getStore, getSunLevel } from '../store.js'

const status: FastifyPluginAsync = async (fastify) => {
  fastify.get('/status/sun', async () => {
    const store = getStore()
    const level = getSunLevel()
    const frameAgeMs = Date.now() - new Date(store.physics.timestamp).getTime()
    const lastAlert = store.recentAlerts[0] ?? null

    const descriptions: Record<string, string> = {
      GREEN: 'Quiet Sun — background X-ray levels, no significant activity',
      YELLOW: 'Elevated activity — HOPE precursor or impulsive phase detected',
      ORANGE: 'M-class flare in progress — moderate disruption possible',
      RED: 'X-class flare — severe space weather event in progress',
    }

    return {
      level,
      description: descriptions[level],
      physics_frame_age_sec: Math.round(frameAgeMs / 1000),
      current_phase: store.physics.flare_phase,
      hope_flag: store.physics.hope_flag,
      goes_class: store.threat.goes_class,
      T_MK: store.physics.T_MK,
      EM_cm3: store.physics.EM_cm3,
      DNI: store.physics.DNI,
      last_event: lastAlert
        ? {
            event_id: lastAlert.event_id,
            goes_class: lastAlert.goes_class,
            peak_time: lastAlert.peak_time,
            confidence: lastAlert.confidence,
          }
        : null,
      timestamp: new Date().toISOString(),
    }
  })
}

export default status
