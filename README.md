# backend-gordon

Motor Node.js de evaluación oral para Gordon:

- Whisper Large-v3: transcripción y timestamps.
- DeepSeek V4 Flash mediante OpenCode Zen: rúbrica, evidencia lingüística,
  doble juez, IPA esperada desde Whisper y recomendación.
- Gordon: calidad del audio, abstención, puntuación y procedencia.
- Wav2Vec2 remoto: evidencia fonética recibida desde el endpoint configurado.

El backend normaliza cada audio a WAV PCM16 mono de 16 kHz y lo envía al
endpoint Wav2Vec2 remoto. Si el endpoint no responde, la evidencia queda como
`providerError`; no se usa ningún modelo fonético local ni se inventa una
puntuación acústica.

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

La respuesta incluye calidad del audio, transcripción de Whisper, una
`speechEvidence.expectedPhonetic` con IPA esperada por palabra, pausas,
dimensiones lingüísticas, recomendación y procedencia. La IPA esperada se
genera únicamente desde Whisper y se conserva separada de
`speechEvidence.phonetic`, que contiene los sonidos observados por Wav2Vec2;
ninguna de las dos capas se reemplaza todavía. El score global es `null`
mientras falte una dimensión obligatoria.

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
- `GORDON_PHONEME_URL` (requerido para la evaluación fonética; acepta la URL
  base o `/api/v1/phonemes`)
- `GORDON_PHONEME_API_KEY` (opcional; se envía como Bearer)
- `GORDON_PHONEME_TIMEOUT_MS` (opcional; predeterminado `240000`)
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
