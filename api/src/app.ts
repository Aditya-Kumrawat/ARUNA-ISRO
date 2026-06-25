import Fastify from 'fastify'
import cors from '@fastify/cors'
import helmet from '@fastify/helmet'
import rateLimit from '@fastify/rate-limit'

import health from './routes/health.js'
import status from './routes/status.js'
import alerts from './routes/alerts.js'
import catalog from './routes/catalog.js'
import physics from './routes/physics.js'
import forecast from './routes/forecast.js'
import threat from './routes/threat.js'
import validation from './routes/validation.js'
import replay from './routes/replay.js'
import bulletin from './routes/bulletin.js'
import gaganyaan from './routes/gaganyaan.js'
import configRoutes from './routes/config.js'
import decisionRoutes from './routes/decision.js'
import missionEventsRoutes from './routes/missionEvents.js'

const API_PREFIX = '/api/v1'

export async function buildApp() {
  const app = Fastify({
    logger: { level: process.env.LOG_LEVEL ?? 'info' },
  })

  await app.register(cors, {
    origin: process.env.CORS_ORIGIN?.split(',') ?? [
      'http://localhost:5173',
      'http://localhost:3000',
    ],
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    credentials: true,
  })

  await app.register(helmet, {
    contentSecurityPolicy: false,  // disabled — frontend needs inline scripts
  })

  await app.register(rateLimit, {
    max: 200,
    timeWindow: '1 minute',
  })

  // All REST routes under /api/v1
  await app.register(health, { prefix: API_PREFIX })
  await app.register(status, { prefix: API_PREFIX })
  await app.register(alerts, { prefix: API_PREFIX })
  await app.register(catalog, { prefix: API_PREFIX })
  await app.register(physics, { prefix: API_PREFIX })
  await app.register(forecast, { prefix: API_PREFIX })
  await app.register(threat, { prefix: API_PREFIX })
  await app.register(validation, { prefix: API_PREFIX })
  await app.register(replay, { prefix: API_PREFIX })
  await app.register(bulletin, { prefix: API_PREFIX })
  await app.register(gaganyaan, { prefix: API_PREFIX })
  await app.register(configRoutes, { prefix: API_PREFIX })
  await app.register(decisionRoutes, { prefix: API_PREFIX })
  await app.register(missionEventsRoutes, { prefix: API_PREFIX })

  // Root redirect to health
  app.get('/', async (_, reply) => reply.redirect('/api/v1/health'))

  return app
}
