# backend-gordon

Servicio Node.js para recibir un archivo de audio y transcribirlo con
`whisper-large-v3` mediante Groq.

## API

### `GET /health`

Comprueba que el servicio está activo.

### `POST /api/transcriptions`

Recibe `multipart/form-data`:

- `audio` (obligatorio): FLAC, M4A, MP3, MP4, MPEG, MPGA, OGG, OPUS, WAV o WEBM,
  con un máximo de 25 MB.
- `language` (opcional): código ISO-639-1, por ejemplo `es` o `en`.
- `prompt` (opcional): contexto que puede orientar la transcripción.
- `evaluationPrompt` (opcional, máximo 1000 caracteres): instrucción del
  docente que orienta la evaluación lingüística. No se envía a Whisper.

Si `language` se omite, Groq detecta el idioma del audio. El backend conserva
esa transcripción y asigna a Azure el locale correspondiente; por ejemplo,
`Spanish` se evalúa como `es-MX` y `English` como `en-US`.

Ejemplo:

```bash
curl -X POST http://localhost:3000/api/transcriptions \
  -F "audio=@/ruta/grabacion.webm" \
  -F "language=es" \
  -F "evaluationPrompt=Evalúa el uso del pasado para un alumno de nivel A2"
```

Respuesta:

```json
{
  "transcription": "Texto reconocido en el audio.",
  "model": "whisper-large-v3",
  "language": "es",
  "duration": 3.4,
  "words": [
    { "word": "Texto", "start": 0.2, "end": 0.7 }
  ],
  "segments": [
    {
      "id": 0,
      "text": "Texto reconocido en el audio.",
      "start": 0.2,
      "end": 3.1,
      "avgLogprob": -0.24,
      "compressionRatio": 1.18,
      "noSpeechProb": 0.04
    }
  ],
  "speechEvidence": {
    "annotatedTranscript": "Hello, my name is [alargamiento 1.3 s] [pausa 0.7 s] David.",
    "pauses": [
      { "start": 2.55, "end": 3.25, "duration": 0.7 }
    ],
    "elongations": [
      {
        "word": "is",
        "start": 1.2,
        "end": 2.5,
        "duration": 1.3
      }
    ],
    "method": "ffmpeg-silencedetect+whisper-word-timestamps"
  },
  "grammar": {
    "provider": "groq",
    "model": "openai/gpt-oss-20b",
    "sufficientEvidence": true,
    "score": 72,
    "summary": "Hay un error de concordancia.",
    "correctedText": "She goes to school every day.",
    "errors": [
      {
        "original": "She go",
        "correction": "She goes",
        "category": "Subject-verb agreement",
        "severity": "moderate",
        "explanation": "La tercera persona singular requiere “goes”."
      }
    ]
  },
  "grammarError": null,
  "pronunciation": {
    "provider": "azure-speech",
    "pronunciationScore": 88.4,
    "accuracyScore": 86.2,
    "fluencyScore": 80.5,
    "completenessScore": 100,
    "prosodyScore": 79.1,
    "words": []
  },
  "pronunciationError": null
}
```

El backend solicita `verbose_json` y marcas por palabra y segmento. Como
Whisper normaliza repeticiones y palabras sostenidas, FFmpeg también mide los
silencios de la señal y el backend devuelve una transcripción anotada. Por
ejemplo, conserva evidencia como `[pausa 0.7 s]` o
`[alargamiento 1.3 s]` sin inventar letras repetidas que el reconocedor no
entregó. La transcripción limpia se mantiene en `transcription`; la evidencia
temporal queda separada en `speechEvidence`. La confianza de reconocimiento no
es una calificación de pronunciación.

La transcripción limpia y `evaluationPrompt` también se envían a
`openai/gpt-oss-20b` mediante Groq para detectar errores gramaticales conforme
al criterio del docente. La instrucción queda delimitada como contexto no
confiable: no puede cambiar el formato de salida ni solicitar que se alteren
otras dimensiones. La respuesta usa un esquema JSON estricto y cada error debe
citar literalmente un fragmento de la transcripción; el backend descarta
observaciones cuyo fragmento no exista. No se penalizan pausas, pronunciación,
puntuación ni estilo. Con menos de tres palabras léxicas, la muestra se marca
como insuficiente y no recibe puntuación gramatical.

Si Azure Speech está configurado, el backend convierte otra copia temporal a
WAV PCM16 mono de 16 kHz y solicita Pronunciation Assessment. La transcripción
de Groq se utiliza como texto de referencia para obtener puntuaciones por
audio, palabra y fonema. La API REST de pronunciación admite audios de hasta 30
segundos.

## Desarrollo local

Requiere Node.js 20 o superior.

```bash
npm install
cp .env.example .env
```

Edita `.env` y coloca tu clave:

```dotenv
GROQ_API_KEY=gsk_tu_clave_real
GROQ_GRAMMAR_MODEL=openai/gpt-oss-20b
AZURE_SPEECH_KEY_PRIMARY=key_1
AZURE_SPEECH_KEY_SECONDARY=key_2
AZURE_SPEECH_REGION=southcentralus
AZURE_SPEECH_ENDPOINT=https://tu-recurso.cognitiveservices.azure.com/
PORT=3000
CORS_ORIGIN=http://localhost:8080
```

Luego inicia el servicio:

```bash
npm run dev
```

La clave real queda excluida por `.gitignore`. Nunca la agregues al código ni al
repositorio.

## Configuración en Render

Al crear el Web Service usa:

- **Root Directory:** `backend-gordon`
- **Runtime:** `Node`
- **Build Command:** `npm ci`
- **Start Command:** `npm start`
- **Health Check Path:** `/health`

Agrega esta variable obligatoria en **Environment**:

- `GROQ_API_KEY`: tu clave privada de Groq.

Para obtener pronunciación y rotación automática agrega:

- `AZURE_SPEECH_KEY_PRIMARY`: `KEY 1` del recurso.
- `AZURE_SPEECH_KEY_SECONDARY`: `KEY 2` del recurso.
- `AZURE_SPEECH_REGION`: identificador como `southcentralus`.
- `AZURE_SPEECH_ENDPOINT`: extremo mostrado por Azure.

Por compatibilidad, `AZURE_SPEECH_KEY` funciona como clave primaria cuando
`AZURE_SPEECH_KEY_PRIMARY` no existe.

Las siguientes variables son opcionales:

- `GROQ_GRAMMAR_MODEL`: modelo para el análisis gramatical. El valor
  predeterminado es `openai/gpt-oss-20b`, que admite el esquema JSON estricto
  usado por el backend.
- `CORS_ORIGIN`: limita qué frontend puede llamar al backend desde un navegador.
  Si se omite, el servicio permite cualquier origen (`*`). Cuando el frontend ya
  esté publicado, es recomendable establecer aquí su URL pública. Se pueden
  indicar varias URLs separadas por comas.
- `NODE_VERSION`: `20`

Render proporciona `PORT` automáticamente; el servidor ya escucha esa variable
en `0.0.0.0`.

### Rotación de claves

El backend llama primero con la clave primaria. Solamente si Azure responde
`401` o `403`, repite una vez con la secundaria. Las claves nunca se incluyen
en respuestas ni registros.

Para rotarlas sin interrupción:

1. Mantén ambas variables configuradas en Render.
2. Regenera `KEY 1` en Azure.
3. Actualiza `AZURE_SPEECH_KEY_PRIMARY` en Render.
4. Comprueba una evaluación.
5. Regenera `KEY 2` y actualiza `AZURE_SPEECH_KEY_SECONDARY`.

No regeneres las dos claves al mismo tiempo.

## Pruebas

```bash
npm test
```

Las pruebas usan un cliente Groq simulado: no consumen créditos ni necesitan una
clave real.

## Conversión de OPUS

Los archivos `.opus` se convierten automáticamente a FLAC mono de 16 kHz antes
de enviarse a Groq. El proyecto incluye un binario de FFmpeg específico para la
plataforma mediante `ffmpeg-static`, por lo que Render lo instala junto con las
dependencias de Node. Tanto el archivo original como el convertido se eliminan
al terminar, incluso si Groq devuelve un error.
