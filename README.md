# backend-gordon

Motor Node.js de evaluación oral para Gordon:

- Whisper Large-v3: transcripción y timestamps.
- DeepSeek V4 Flash mediante OpenCode Zen: rúbrica, evidencia lingüística,
  doble juez, pronunciación fonética y recomendación.
- Gordon: calidad del audio, abstención, puntuación y procedencia.
- Wav2Vec2 local: evidencia fonética experimental en desarrollo.

Pronunciación y fluidez quedan `unavailable` hasta integrar y calibrar el
alineador fonético local. El backend no inventa puntuaciones acústicas ni
redistribuye sus pesos.

## API v2

- `POST /api/v2/instructions/improve`
- `POST /api/v2/rubrics/draft`
- `POST /api/v2/rubrics/confirm`
- `POST /api/v2/assessments`
- `POST /api/v2/human-ratings`
- `DELETE /api/v2/pilot-records/:assessmentId`

`POST /api/v2/assessments` recibe `audio` y
`confirmedRubricToken` como multipart. MP3, Opus, WebM, OGG, M4A, MP4, MPEG,
FLAC y WAV se validan por firma y se normalizan a WAV PCM16 mono de 16 kHz.

La respuesta incluye calidad del audio, transcripción de Whisper, pausas,
dimensiones lingüísticas, recomendación y procedencia. El score global es
`null` mientras falte una dimensión obligatoria.

## Compatibilidad v1

`POST /api/transcriptions` conserva transcripción, marcas temporales, pausas y
análisis gramatical. Ya no devuelve evaluación remota de pronunciación.

## Configuración

Requiere Node.js 22 o posterior:

```bash
npm install
cp .env.example .env
npm run dev
```

Variables:

- `GROQ_API_KEY`
- `RUBRIC_SIGNING_SECRET`
- `CORS_ORIGIN`
- `OPENCODE_API_KEY`
- `OPENCODE_MODEL` (opcional; predeterminado `deepseek-v4-flash`)
- `OPENCODE_BASE_URL` (opcional; predeterminado `https://opencode.ai/zen/v1`)
- `OPENCODE_LINGUISTIC_TPM_LIMIT` (opcional)
- `MAX_CONCURRENT_ASSESSMENTS` (opcional)
- `RATE_LIMIT_PER_MINUTE` (opcional)

El almacenamiento consentido está deshabilitado hasta configurar un proveedor
nuevo. Los audios temporales se eliminan al terminar cada solicitud.

## Render

- Root Directory: `backend-gordon`
- Build Command: `npm ci`
- Start Command: `npm start`
- Health Check Path: `/health`
- Runtime: Node 22+ (Render lo selecciona desde `engines.node`)

## Pruebas

```bash
npm test
```
