import type { FastifyPluginAsync } from 'fastify'
import { getStore } from '../store.js'

interface ForecastHistoryQuery { limit?: string }

const forecast: FastifyPluginAsync = async (fastify) => {
  fastify.get('/forecast/current', async () => {
    return getStore().forecast
  })

  fastify.get<{ Querystring: ForecastHistoryQuery }>('/forecast/history', async (req) => {
    const limit = Math.min(Math.max(1, Number(req.query.limit ?? 50)), 200)
    const store = getStore()
    const history = store.forecastHistory.slice(-limit)
    return { count: history.length, frames: history }
  })
}

export default forecast
