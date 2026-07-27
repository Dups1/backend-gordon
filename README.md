# backend-gordon

Motor Node.js de evaluación oral para Gordon. La API separa:

- Whisper Large-v3: transcripción y timestamps.
- Azure Pronunciation Assessment: evidencia acústica y fonética.
- GPT-OSS: extracción lingüística, dos jueces independientes, adjudicación y
  recomendación.
- Gordon: calidad del audio, estados de abstención, perfil de pesos, intervalos
  provisionales, procedencia y consentimiento.

Los scores actuales son **provisionales** hasta completar la calibración con
evaluadores humanos. No representan una clasificación CEFR global: miden el
dominio del nivel objetivo seleccionado por el docente.

## API v2

### Mejorar una consigna

```http
POST /api/v2/instructions/improve
Content-Type: application/json
```

Recibe la misma especificación básica de la rúbrica: `mode`,
`targetLocale`, `cefr`, `instruction`, `communicativePurpose` y, en lectura,
`referenceText`. GPT-OSS devuelve `originalInstruction` e
`improvedInstruction` por separado. La mejora conserva los requisitos
explícitos y nunca se activa sin que el docente pueda revisar, editar o volver
al texto original.

### Crear y confirmar una rúbrica

```http
POST /api/v2/rubrics/draft
Content-Type: application/json
```

```json
{
  "mode": "spontaneous",
  "targetLocale": "en-US",
  "cefr": "B1",
  "instruction": "Explain a learning experience and justify its importance.",
  "communicativePurpose": "Explain and justify"
}
```

Para lectura usa `"mode": "reading"` y agrega `referenceText`. GPT-OSS propone
descriptores estructurados; si el proveedor está temporalmente fuera de
servicio se devuelve una rúbrica CEFR conservadora con una advertencia. En
ambos casos el docente debe revisarla.

```http
POST /api/v2/rubrics/confirm
Content-Type: application/json
```

```json
{ "draft": { "...": "borrador revisado por el docente" } }
```

La respuesta contiene `confirmedRubricToken`, firmado con HMAC. Cualquier
alteración posterior invalida el token.

### Evaluar audio

```http
POST /api/v2/assessments
Content-Type: multipart/form-data
Idempotency-Key: intento-123
```

Campos:

- `audio`: obligatorio y con bytes reales.
- `confirmedRubricToken`: obligatorio.
- `consentToStore`: `true` o `false`.
- `consentVersion`: obligatorio cuando se autoriza almacenar.
- `participantPseudonym`: opcional.

Ejemplo:

```bash
curl -X POST http://localhost:3000/api/v2/assessments \
  -H "Idempotency-Key: intento-123" \
  -F "audio=@/ruta/student.webm" \
  -F "confirmedRubricToken=TOKEN_CONFIRMADO" \
  -F "consentToStore=false"
```

Todos los formatos admitidos —MP3, Opus, WebM, OGG, M4A, MP4, MPEG, FLAC y
WAV— se validan por firma y se decodifican a WAV PCM16 mono a 16 kHz. El
servidor mide duración, RMS, clipping, ruido, SNR estimado, proporción y
duración de voz antes de consumir proveedores.

En lectura Azure recibe el texto canónico confirmado, nunca la transcripción
de Whisper. Las lecturas largas se dividen en bloques de hasta 25 segundos y
se agregan por fonemas, palabras y duración elegible. Todas las respuestas
espontáneas usan Azure Speech SDK continuo sin `ReferenceText`; REST queda
reservado para lectura guiada.

Whisper usa bloques de 28 segundos con dos segundos de solapamiento para audio
largo. El backend corrige offsets y elimina duplicados del solapamiento. Solo
se envían pistas breves de vocabulario confirmadas; nunca se usa la respuesta
esperada completa como prompt del ASR.

La respuesta v2 incluye:

- calidad y hashes del audio;
- transcripciones independiente de Whisper y Azure y su desacuerdo;
- pausas y prolongaciones como eventos estructurados;
- cinco dimensiones con `status`, `score`, `rawScore`, intervalo 90 %,
  confiabilidad, método, versión, evidencia y limitaciones;
- score global únicamente cuando las cinco dimensiones fueron puntuadas;
- recomendación que no puede alterar puntuaciones;
- versiones de proveedores, prompts, perfil y calibración.

En respuestas espontáneas mayores de 30 segundos se intenta Azure continuo. Si
esa sesión se cancela, el backend divide el WAV normalizado en tomas naturales
de hasta 25 segundos y vuelve a evaluarlas con Azure single-shot antes de
agregar fonemas, palabras, fluidez y prosodia por duración elegible. Esto sigue
siendo evidencia de Azure, no un fallback heurístico. Si también falla la ruta
segmentada, Pronunciation y Fluency quedan `unavailable`. Si falta cualquier
dimensión no se redistribuyen pesos y el score global queda `null`.

### Revisión humana

```http
POST /api/v2/human-ratings
Authorization: Bearer TEACHER_REVIEW_TOKEN
Content-Type: application/json
```

```json
{
  "assessmentId": "uuid",
  "ratings": {
    "communication": { "band": 3, "rationale": "Evidencia docente" },
    "pronunciation": { "band": 2, "rationale": "Evidencia docente" }
  },
  "generalRationale": "Justificación de la revisión"
}
```

Las bandas humanas se guardan como etiquetas independientes; no sobrescriben
el reporte automático. Solo se aceptan cuando la evaluación y su consentimiento
ya existen en el contenedor privado.

Para retirar un registro:

```http
DELETE /api/v2/pilot-records/:assessmentId
Authorization: Bearer TEACHER_REVIEW_TOKEN
```

## Compatibilidad v1

`POST /api/transcriptions` se conserva durante la migración. Devuelve el
contrato anterior, pero ya no fuerza Grammar a 100 cuando un hallazgo no puede
verificarse. La aplicación Flutter usa v2; v1 debe considerarse
transcripción/adaptador y no una evaluación completa.

## Configuración local

Requiere Node.js 20 o superior.

```bash
npm install
cp .env.example .env
npm run dev
```

Variables obligatorias para producción:

- `GROQ_API_KEY`
- `RUBRIC_SIGNING_SECRET`: al menos 32 bytes aleatorios.
- `AZURE_SPEECH_KEY_PRIMARY`
- `AZURE_SPEECH_REGION`
- `CORS_ORIGIN`: dominio exacto del frontend.

Recomendadas:

- `AZURE_SPEECH_KEY_SECONDARY`: rotación automática en 401/403.
- `AZURE_SPEECH_ENDPOINT`
- `GROQ_GRAMMAR_MODEL` (`openai/gpt-oss-20b` por defecto).
- `TEACHER_REVIEW_TOKEN`: acceso al corpus y calificaciones humanas.
- `MAX_CONCURRENT_ASSESSMENTS` (`2` por defecto).
- `RATE_LIMIT_PER_MINUTE` (`30` por defecto).
- `REQUEST_TIMEOUT_MS` (`480000`, ocho minutos, por defecto).

Almacenamiento consentido:

- `AZURE_STORAGE_CONNECTION_STRING`
- `AZURE_STORAGE_CONTAINER` (`gordon-pilot-private` por defecto).
- `AUDIO_RETENTION_DAYS` (`90` por defecto).

Sin estas dos variables la evaluación sigue funcionando, pero el backend
indicará que el registro consentido no se pudo almacenar.

## Render

- Root Directory: `backend-gordon`
- Build Command: `npm ci`
- Start Command: `npm start`
- Health Check Path: `/health`
- Runtime: Node 20+

`/health` confirma que el proceso vive. `/ready` comprueba Groq, FFmpeg, secreto
de rúbricas, Azure, Blob y CORS sin exponer secretos.

Render usa disco efímero. Los archivos se crean por solicitud y se eliminan
antes de responder. Azure Blob es la única persistencia del piloto.

## Privacidad

- Sin consentimiento: audio original y normalizado se eliminan al terminar.
- Con consentimiento: Blob privado conserva original, normalizado, JSON de
  proveedores, reporte, consentimiento y ratings.
- Original y normalizado llevan vencimiento y el backend elimina audio vencido
  al iniciar y cada seis horas; la retención predeterminada es 90 días.
- Logs y errores no incluyen audio, transcripción ni claves.
- El token docente se recibe por encabezado y nunca se devuelve.

## Pruebas

```bash
npm test
npm run calibration:analyze -- /ruta/al/piloto.jsonl
```

La suite cubre compatibilidad v1, firma de rúbricas, no redistribución de
pesos, scores inválidos, formatos y firmas reales, quality gate, abstención
parcial y contrato síncrono v2. Los proveedores se simulan; para validar
precisión real se necesita el corpus humano descrito en
`calibration/README.md`.
