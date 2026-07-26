import 'dotenv/config';

import { createApp } from './app.js';

const port = Number.parseInt(process.env.PORT ?? '3000', 10);
const requestTimeoutMs = Number.parseInt(
  process.env.REQUEST_TIMEOUT_MS ?? String(8 * 60 * 1000),
  10,
);

if (!process.env.GROQ_API_KEY?.trim()) {
  console.error(
    'Falta GROQ_API_KEY. Configúrala en el entorno antes de iniciar el servicio.',
  );
  process.exit(1);
}

const app = createApp({ logger: console });
const server = app.listen(port, '0.0.0.0', () => {
  console.log(`backend-gordon escuchando en el puerto ${port}`);
});
server.requestTimeout =
  Number.isFinite(requestTimeoutMs) && requestTimeoutMs > 0
    ? requestTimeoutMs
    : 8 * 60 * 1000;
server.headersTimeout = Math.min(server.requestTimeout, 65 * 1000);
