import type { FastifyPluginAsync } from 'fastify'
import { spawn, type ChildProcess } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { setReplayState, getStore } from '../store.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const PROJECT_ROOT = path.resolve(__dirname, '..', '..', '..')

// Registry of known replayable events (matches test fixtures)
const KNOWN_EVENTS = new Set(['SOL2024-02-22-X6.3'])

interface ReplayBody {
  event_id: string
  speed?: number
  fits_dir?: string
}

let _replayProcess: ChildProcess | null = null

const replay: FastifyPluginAsync = async (fastify) => {
  fastify.post<{ Body: ReplayBody }>('/replay/event', async (req, reply) => {
    const { event_id, speed = 10, fits_dir = 'tests/fixtures/x63' } = req.body ?? {}

    if (!event_id || typeof event_id !== 'string') {
      return reply.code(400).send({ error: 'event_id is required' })
    }

    // Allow known events OR user-supplied arbitrary dates (e.g. "2024-05-09")
    const safeId = /^[A-Za-z0-9._-]+$/.test(event_id)
    if (!safeId) {
      return reply.code(400).send({ error: 'Invalid event_id format' })
    }

    const safeSpeed = Math.min(Math.max(1, Number(speed) || 10), 100)

    // Kill existing replay before starting new one
    if (_replayProcess && !_replayProcess.killed) {
      _replayProcess.kill('SIGTERM')
      setReplayState(false, null)
    }

    const pythonCmd = process.platform === 'win32' ? 'python' : 'python3'
    const args = [
      '-m', 'aruna.replay',
      '--event-id', event_id,
      '--fits-dir', fits_dir,
      '--speed', String(safeSpeed),
    ]

    fastify.log.info({ event_id, speed: safeSpeed }, 'Starting replay')

    _replayProcess = spawn(pythonCmd, args, {
      cwd: PROJECT_ROOT,
      stdio: ['ignore', 'pipe', 'pipe'],
    })

    setReplayState(true, event_id)

    _replayProcess.on('exit', (code) => {
      fastify.log.info({ event_id, code }, 'Replay process exited')
      setReplayState(false, null)
      _replayProcess = null
    })

    _replayProcess.stderr?.on('data', (d: Buffer) => {
      fastify.log.debug({ event_id }, d.toString().trim())
    })

    const replay_id = `replay-${event_id}-${Date.now()}`
    return {
      status: 'started',
      replay_id,
      event_id,
      speed: safeSpeed,
      message: `Replay of ${event_id} started at ${safeSpeed}× speed. Physics frames will stream via Socket.IO.`,
    }
  })

  fastify.delete('/replay/event', async () => {
    if (_replayProcess && !_replayProcess.killed) {
      _replayProcess.kill('SIGTERM')
      setReplayState(false, null)
      _replayProcess = null
      return { status: 'stopped' }
    }
    const store = getStore()
    return { status: store.replayActive ? 'stopping' : 'not_running' }
  })

  fastify.get('/replay/status', async () => {
    const store = getStore()
    return {
      active: store.replayActive,
      event_id: store.replayEventId,
      process_alive: _replayProcess != null && !_replayProcess.killed,
    }
  })
}

export default replay
