import { buildApp } from './app.js'
import { setupSocketIO } from './ws/socket.js'
import { initDatabase, DB_PATH } from './db.js'

const PORT = Number(process.env.PORT ?? 3001)
const HOST = process.env.HOST ?? '0.0.0.0'

async function main() {
  // Initialise SQLite (WASM load + optional DB open) before accepting requests
  await initDatabase()

  const app = await buildApp()
  await app.listen({ port: PORT, host: HOST })

  // Socket.IO attaches to the raw HTTP server AFTER fastify.listen()
  setupSocketIO(app.server)

  console.log(`
╔═══════════════════════════════════════════════════════════╗
║          ARUNA API Gateway v2.0.0                         ║
╠═══════════════════════════════════════════════════════════╣
║  REST  →  http://localhost:${PORT}/api/v1/health             ║
║  WS    →  ws://localhost:${PORT}  (Socket.IO)                ║
║  DB    →  ${DB_PATH}
╚═══════════════════════════════════════════════════════════╝
  `)
}

main().catch((err) => {
  console.error('[ARUNA] Fatal startup error:', err)
  process.exit(1)
})
