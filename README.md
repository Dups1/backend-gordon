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

Ejemplo:

```bash
curl -X POST http://localhost:3000/api/transcriptions \
  -F "audio=@/ruta/grabacion.webm" \
  -F "language=es" \
  -F "prompt=Evaluación oral académica"
```

Respuesta:

```json
{
  "transcription": "Texto reconocido en el audio.",
  "model": "whisper-large-v3",
  "language": "es"
}
```

## Desarrollo local

Requiere Node.js 20 o superior.

```bash
npm install
cp .env.example .env
```

Edita `.env` y coloca tu clave:

```dotenv
GROQ_API_KEY=gsk_tu_clave_real
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

Las siguientes variables son opcionales:

- `CORS_ORIGIN`: limita qué frontend puede llamar al backend desde un navegador.
  Si se omite, el servicio permite cualquier origen (`*`). Cuando el frontend ya
  esté publicado, es recomendable establecer aquí su URL pública. Se pueden
  indicar varias URLs separadas por comas.
- `NODE_VERSION`: `20`

Render proporciona `PORT` automáticamente; el servidor ya escucha esa variable
en `0.0.0.0`.

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
