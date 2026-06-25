import { FastifyInstance } from 'fastify';
import { getStore } from '../store.js';

export default async function missionEventsRoutes(app: FastifyInstance) {
  app.get<{ Querystring: { limit?: string } }>('/events/timeline', async (req, reply) => {
    const limit = Math.min(500, parseInt(req.query.limit ?? '100', 10));
    const store = getStore();
    return reply.send({ ok: true, events: store.missionEvents.slice(0, limit) });
  });
}
