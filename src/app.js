import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createReadStream, mkdirSync } from 'node:fs';
import { stat, unlink } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import cors from 'cors';
import express from 'express';
import ffmpegPath from 'ffmpeg-static';
import Groq from 'groq-sdk';
import helmet from 'helmet';
import multer from 'multer';

export const GROQ_MODEL = 'whisper-large-v3';
export const MAX_AUDIO_BYTES = 25 * 1024 * 1024;

const extensionesAdmitidas = new Set([
  '.flac',
  '.m4a',
  '.mp3',
  '.mp4',
  '.mpeg',
  '.mpga',
  '.ogg',
  '.opus',
  '.wav',
  '.webm',
]);

const directorioTemporal = path.join(os.tmpdir(), 'backend-gordon-audio');
mkdirSync(directorioTemporal, { recursive: true });

class ErrorHttp extends Error {
  constructor(status, message, code) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

function crearAlmacenamientoTemporal() {
  return multer.diskStorage({
    destination: (_request, _archivo, callback) => {
      callback(null, directorioTemporal);
    },
    filename: (_request, archivo, callback) => {
      const extension = path.extname(archivo.originalname).toLowerCase();
      callback(null, `${randomUUID()}${extension}`);
    },
  });
}

function validarTipoDeAudio(_request, archivo, callback) {
  const extension = path.extname(archivo.originalname).toLowerCase();

  if (!extensionesAdmitidas.has(extension)) {
    callback(
      new ErrorHttp(
        415,
        'Formato no admitido. Usa FLAC, M4A, MP3, MP4, MPEG, MPGA, OGG, OPUS, WAV o WEBM.',
        'FORMATO_NO_ADMITIDO',
      ),
    );
    return;
  }

  callback(null, true);
}

function crearSubida(maxAudioBytes) {
  return multer({
    storage: crearAlmacenamientoTemporal(),
    limits: {
      fileSize: maxAudioBytes,
      files: 1,
      fields: 4,
    },
    fileFilter: validarTipoDeAudio,
  });
}

export async function convertirOpusAFlac(
  rutaEntrada,
  {
    binario = ffmpegPath,
    maxAudioBytes = MAX_AUDIO_BYTES,
    tiempoLimiteMs = 60000,
  } = {},
) {
  if (!binario) {
    throw new ErrorHttp(
      503,
      'El conversor de audio no está disponible.',
      'FFMPEG_NO_DISPONIBLE',
    );
  }

  const rutaSalida = path.join(
    directorioTemporal,
    `${randomUUID()}.flac`,
  );
  const argumentos = [
    '-nostdin',
    '-hide_banner',
    '-loglevel',
    'error',
    '-y',
    '-i',
    rutaEntrada,
    '-vn',
    '-ac',
    '1',
    '-ar',
    '16000',
    '-c:a',
    'flac',
    '-compression_level',
    '5',
    '-fs',
    String(maxAudioBytes),
    rutaSalida,
  ];

  try {
    await new Promise((resolve, reject) => {
      const proceso = spawn(binario, argumentos, {
        stdio: ['ignore', 'ignore', 'ignore'],
      });
      let excedioTiempo = false;
      const temporizador = setTimeout(() => {
        excedioTiempo = true;
        proceso.kill('SIGKILL');
      }, tiempoLimiteMs);

      proceso.once('error', () => {
        clearTimeout(temporizador);
        reject(
          new ErrorHttp(
            503,
            'No fue posible iniciar el conversor de audio.',
            'FFMPEG_NO_DISPONIBLE',
          ),
        );
      });
      proceso.once('close', (codigo) => {
        clearTimeout(temporizador);
        if (excedioTiempo) {
          reject(
            new ErrorHttp(
              504,
              'La conversión del audio tardó demasiado.',
              'CONVERSION_AGOTADA',
            ),
          );
          return;
        }
        if (codigo !== 0) {
          reject(
            new ErrorHttp(
              422,
              'El archivo OPUS no contiene audio válido.',
              'OPUS_INVALIDO',
            ),
          );
          return;
        }
        resolve();
      });
    });

    const informacion = await stat(rutaSalida);
    if (informacion.size === 0) {
      throw new ErrorHttp(
        422,
        'La conversión no produjo audio.',
        'OPUS_INVALIDO',
      );
    }
    if (informacion.size >= maxAudioBytes) {
      throw new ErrorHttp(
        413,
        'El audio convertido supera el límite de 25 MB.',
        'ARCHIVO_CONVERTIDO_DEMASIADO_GRANDE',
      );
    }

    return rutaSalida;
  } catch (error) {
    await unlink(rutaSalida).catch(() => {});
    throw error;
  }
}

function crearConfiguracionCors(origenesConfigurados) {
  if (!origenesConfigurados || origenesConfigurados.trim() === '*') {
    return { origin: '*' };
  }

  const origenes = new Set(
    origenesConfigurados
      .split(',')
      .map((origen) => origen.trim())
      .filter(Boolean),
  );

  return {
    origin(origen, callback) {
      if (!origen || origenes.has(origen)) {
        callback(null, true);
        return;
      }

      callback(
        new ErrorHttp(
          403,
          'El origen de la solicitud no está permitido.',
          'ORIGEN_NO_PERMITIDO',
        ),
      );
    },
  };
}

function obtenerClienteGroq(clienteInyectado) {
  if (clienteInyectado) {
    return clienteInyectado;
  }

  const apiKey = process.env.GROQ_API_KEY?.trim();
  if (!apiKey) {
    throw new ErrorHttp(
      503,
      'El servicio de transcripción aún no está configurado.',
      'GROQ_NO_CONFIGURADO',
    );
  }

  return new Groq({ apiKey });
}

function textoOpcional(valor) {
  return typeof valor === 'string' ? valor.trim() : '';
}

function validarIdioma(valor) {
  const idioma = textoOpcional(valor).toLowerCase();
  if (idioma && !/^[a-z]{2}$/.test(idioma)) {
    throw new ErrorHttp(
      400,
      'El idioma debe ser un código ISO-639-1 de dos letras, por ejemplo "es" o "en".',
      'IDIOMA_INVALIDO',
    );
  }
  return idioma;
}

function errorDeGroq(error) {
  if (error?.status === 429) {
    return new ErrorHttp(
      429,
      'Groq alcanzó temporalmente el límite de solicitudes. Intenta de nuevo en unos momentos.',
      'LIMITE_GROQ',
    );
  }

  return new ErrorHttp(
    502,
    'No fue posible transcribir el audio con Groq.',
    'ERROR_GROQ',
  );
}

export function createApp({
  groqClient,
  corsOrigin = process.env.CORS_ORIGIN ?? '*',
  maxAudioBytes = MAX_AUDIO_BYTES,
  convertirOpus = convertirOpusAFlac,
} = {}) {
  const app = express();
  const subirAudio = crearSubida(maxAudioBytes);

  app.disable('x-powered-by');
  app.use(helmet());
  app.use(cors(crearConfiguracionCors(corsOrigin)));

  app.get('/health', (_request, response) => {
    response.json({
      ok: true,
      service: 'backend-gordon',
      model: GROQ_MODEL,
    });
  });

  app.post(
    '/api/transcriptions',
    subirAudio.single('audio'),
    async (request, response, next) => {
      if (!request.file) {
        next(
          new ErrorHttp(
            400,
            'Adjunta un archivo en el campo multipart "audio".',
            'AUDIO_REQUERIDO',
          ),
        );
        return;
      }

      let rutaConvertida;
      try {
        const language = validarIdioma(request.body.language);
        const prompt = textoOpcional(request.body.prompt);
        const cliente = obtenerClienteGroq(groqClient);
        const esOpus =
          path.extname(request.file.originalname).toLowerCase() === '.opus';
        if (esOpus) {
          rutaConvertida = await convertirOpus(request.file.path, {
            maxAudioBytes,
          });
        }
        const rutaParaTranscribir = rutaConvertida ?? request.file.path;
        const opciones = {
          file: createReadStream(rutaParaTranscribir),
          model: GROQ_MODEL,
          response_format: 'json',
          temperature: 0,
        };

        if (language) {
          opciones.language = language;
        }
        if (prompt) {
          opciones.prompt = prompt;
        }

        let resultado;
        try {
          resultado = await cliente.audio.transcriptions.create(opciones);
        } catch (error) {
          throw errorDeGroq(error);
        }

        response.json({
          transcription: resultado.text,
          model: GROQ_MODEL,
          language: language || null,
        });
      } catch (error) {
        next(error);
      } finally {
        await unlink(request.file.path).catch(() => {});
        if (rutaConvertida) {
          await unlink(rutaConvertida).catch(() => {});
        }
      }
    },
  );

  app.use((_request, _response, next) => {
    next(new ErrorHttp(404, 'Ruta no encontrada.', 'RUTA_NO_ENCONTRADA'));
  });

  app.use((error, _request, response, _next) => {
    if (error instanceof multer.MulterError) {
      if (error.code === 'LIMIT_FILE_SIZE') {
        response.status(413).json({
          error: {
            code: 'ARCHIVO_DEMASIADO_GRANDE',
            message: 'El archivo supera el límite de 25 MB.',
          },
        });
        return;
      }

      response.status(400).json({
        error: {
          code: 'SUBIDA_INVALIDA',
          message: 'No fue posible procesar el archivo enviado.',
        },
      });
      return;
    }

    const status = Number.isInteger(error.status) ? error.status : 500;
    response.status(status).json({
      error: {
        code: error.code ?? 'ERROR_INTERNO',
        message:
          status >= 500 && !error.code
            ? 'Ocurrió un error interno.'
            : error.message,
      },
    });
  });

  return app;
}
