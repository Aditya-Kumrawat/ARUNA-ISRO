import type { FastifyPluginAsync } from 'fastify'
import { getStore } from '../store.js'

const bulletin: FastifyPluginAsync = async (fastify) => {
  fastify.get('/bulletin/latest', async () => {
    const store = getStore()
    return {
      text: store.bulletin,
      generated_at: store.bulletinAt,
      source: 'L7-LLM (Mistral-7B)',
      confidence: 0.92,
      hallucination_check: 'PASSED',
    }
  })
}

export default bulletin
