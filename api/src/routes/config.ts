import { FastifyInstance } from 'fastify';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import yaml from 'js-yaml';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = path.resolve(__dirname, '../../../config/config.yaml');

export default async function configRoutes(app: FastifyInstance) {
  app.get('/config', async (_req, reply) => {
    try {
      const raw = fs.readFileSync(CONFIG_PATH, 'utf8');
      const parsed = yaml.load(raw);
      return reply.send({ ok: true, config: parsed });
    } catch (e) {
      return reply.status(500).send({ ok: false, error: 'Failed to read config.yaml' });
    }
  });

  app.post<{ Body: Record<string, unknown> }>('/config', async (req, reply) => {
    try {
      const updated = yaml.dump(req.body);
      fs.writeFileSync(CONFIG_PATH, updated, 'utf8');
      return reply.send({ ok: true });
    } catch (e) {
      return reply.status(500).send({ ok: false, error: 'Failed to write config.yaml' });
    }
  });
}
