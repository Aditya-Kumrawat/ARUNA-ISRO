import type { FastifyPluginAsync } from 'fastify'
import { getStore, getPhysicsHistory } from '../store.js'

interface PhysicsHistoryQuery { minutes?: string }

const physics: FastifyPluginAsync = async (fastify) => {
  fastify.get('/physics/current', async () => {
    return getStore().physics
  })

  fastify.get<{ Querystring: PhysicsHistoryQuery }>('/physics/history', async (req) => {
    const minutes = Math.min(Math.max(1, Number(req.query.minutes ?? 10)), 60)
    const frames = getPhysicsHistory(minutes)
    return { minutes, count: frames.length, frames }
  })
}

export default physics
