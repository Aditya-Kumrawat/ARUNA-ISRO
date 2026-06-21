import type { FastifyPluginAsync } from 'fastify'
import { listEvents, getEvent, countEvents } from '../db.js'

interface CatalogQuery {
  page?: string
  limit?: string
  goes_class?: string
  date_from?: string
  date_to?: string
}

interface EventParams { id: string }

const catalog: FastifyPluginAsync = async (fastify) => {
  fastify.get<{ Querystring: CatalogQuery }>('/catalog/events', async (req) => {
    const page = Math.max(1, Number(req.query.page ?? 1))
    const limit = Math.min(Math.max(1, Number(req.query.limit ?? 20)), 100)
    const offset = (page - 1) * limit
    const opts = {
      limit,
      offset,
      goesClass: req.query.goes_class,
      dateFrom: req.query.date_from,
      dateTo: req.query.date_to,
    }
    const [events, total] = [listEvents(opts), countEvents(opts)]
    return {
      page,
      limit,
      total,
      pages: Math.ceil(total / limit),
      events,
    }
  })

  fastify.get<{ Params: EventParams }>('/catalog/events/:id', async (req, reply) => {
    const event = getEvent(req.params.id)
    if (!event) {
      return reply.code(404).send({ error: 'Event not found', event_id: req.params.id })
    }
    return event
  })
}

export default catalog
