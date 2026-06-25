import { FastifyInstance } from 'fastify';
import { getStore } from '../store.js';

export default async function decisionRoutes(app: FastifyInstance) {
  app.get('/decision/current', async (_req, reply) => {
    const store = getStore();
    if (!store.decisionFrame) {
      return reply.send({
        recommendation: 'INCREASE_MONITORING',
        confidence: 0.12,
        actions: ['Continue nominal operations', 'Monitor SoLEXS/HEL1OS'],
        affected_assets: [],
        mission_brief: 'Nominal solar activity — no significant flare probability.',
        valid_until: new Date(Date.now() + 300_000).toISOString(),
        reasoning: {
          physics_evidence: ['Quiet sun conditions'],
          trigger_chain: ['QUIET'],
          confidence_contributors: {},
          analog_match: 'No recent analog.',
          physics_consistency: 'All instruments nominal.',
          mission_impact_summary: 'No operational impact.',
        },
        updated_at: new Date().toISOString(),
      });
    }
    return reply.send(store.decisionFrame);
  });
}
