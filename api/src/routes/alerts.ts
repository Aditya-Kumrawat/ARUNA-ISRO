import type { FastifyPluginAsync } from 'fastify'
import { getRecentEvents, getEvent } from '../db.js'
import { getStore } from '../store.js'

interface AlertsQuery { limit?: string }
interface AlertParams { id: string }

const alerts: FastifyPluginAsync = async (fastify) => {
  fastify.get<{ Querystring: AlertsQuery }>('/alerts/latest', async (req) => {
    const limit = Math.min(Number(req.query.limit ?? 10), 100)
    const store = getStore()

    // Prefer in-memory recent alerts (already hydrated from DB by poller)
    const fromStore = store.recentAlerts.slice(0, limit)
    if (fromStore.length >= limit) {
      return { count: fromStore.length, events: fromStore }
    }

    // Fall back to direct DB read if store hasn't warmed up yet
    const events = getRecentEvents(limit)
    return { count: events.length, events }
  })

  fastify.get<{ Params: AlertParams }>('/alerts/:id', async (req, reply) => {
    const event = getEvent(req.params.id)
    if (!event) {
      return reply.code(404).send({ error: 'Event not found', event_id: req.params.id })
    }
    return event
  })
}

export default alerts
