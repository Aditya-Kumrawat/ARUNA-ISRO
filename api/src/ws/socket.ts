import { Server as SocketServer } from 'socket.io'
import type { Server as HttpServer } from 'node:http'
import { getStore, getSunLevel } from '../store.js'
import { startPoller } from '../poller.js'

export function setupSocketIO(httpServer: HttpServer): SocketServer {
  const io = new SocketServer(httpServer, {
    cors: {
      origin: process.env.CORS_ORIGIN?.split(',') ?? ['http://localhost:5173', 'http://localhost:3000'],
      methods: ['GET', 'POST'],
    },
    pingTimeout: 20_000,
    pingInterval: 10_000,
  })

  io.on('connection', (socket) => {
    const clientIp = socket.handshake.address
    console.log(`[socket.io] Client connected: ${socket.id} from ${clientIp}`)

    // Immediately push current state so client doesn't wait for the 1s tick
    const store = getStore()
    socket.emit('physics:frame', store.physics)
    socket.emit('forecast:frame', store.forecast)
    socket.emit('health:update', {
      status: 'HEALTHY',
      level: getSunLevel(),
      uptime_sec: Math.round((Date.now() - store.startedAt) / 1000),
      replay_active: store.replayActive,
      timestamp: new Date().toISOString(),
    })

    socket.on('disconnect', (reason) => {
      console.log(`[socket.io] Client disconnected: ${socket.id} (${reason})`)
    })

    // Client can request a physics history snapshot
    socket.on('physics:history', (payload: { minutes?: number }) => {
      const minutes = Math.min(Math.max(1, Number(payload?.minutes ?? 5)), 60)
      const cutoff = new Date(Date.now() - minutes * 60_000).toISOString()
      const frames = store.physicsHistory.filter(f => f.timestamp >= cutoff)
      socket.emit('physics:history', { minutes, count: frames.length, frames })
    })
  })

  // Start poller — feeds physics frames to all connected clients
  startPoller(io)

  return io
}
