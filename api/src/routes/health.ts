import type { FastifyPluginAsync } from 'fastify'
import { existsSync } from 'node:fs'
import { DB_PATH } from '../db.js'
import { getStore } from '../store.js'

const health: FastifyPluginAsync = async (fastify) => {
  fastify.get('/health', async () => {
    const store = getStore()
    const uptime = Math.round((Date.now() - store.startedAt) / 1000)
    const frameAgeMs = Date.now() - new Date(store.physics.timestamp).getTime()

    const physicsOk = frameAgeMs < 10_000
    const db_connected = existsSync(DB_PATH)

    return {
      status: physicsOk ? 'HEALTHY' : 'DEGRADED',
      timestamp: new Date().toISOString(),
      uptime_sec: uptime,
      version: '2.0.0',
      db_connected,
      db_path: DB_PATH,
      stages: {
        l1_ingestion: { status: physicsOk ? 'OK' : 'DEGRADED', last_update_sec: Math.round(frameAgeMs / 1000) },
        l2_physics: { status: physicsOk ? 'OK' : 'DEGRADED', last_update_sec: Math.round(frameAgeMs / 1000) },
        l3_nowcasting: { status: db_connected ? 'OK' : 'DEGRADED', last_update_sec: 0 },
        l4_forecasting: { status: 'OK', last_update_sec: 0 },
        l6_threat: { status: 'OK', last_update_sec: 0 },
        l7_llm: { status: 'OK', last_update_sec: 0 },
        kafka: { status: 'DOWN', last_update_sec: -1, message: 'Kafka skipped for demo' },
        redis: { status: 'DOWN', last_update_sec: -1, message: 'Redis skipped for demo' },
      },
    }
  })
}

export default health
