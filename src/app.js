import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createReadStream, mkdirSync } from 'node:fs';
import { readFile, stat, unlink } from 'node:fs/promises';
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
export const MAX_AZURE_PRONUNCIATION_SECONDS = 30;
export const MIN_PAUSE_SECONDS = 0.6;
export const MIN_ELONGATION_SECONDS = 0.9;

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

export async function convertirAudioAWav(
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

  const rutaSalida = path.join(directorioTemporal, `${randomUUID()}.wav`);
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
    'pcm_s16le',
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
              'El archivo no contiene audio válido para evaluar.',
              'AUDIO_INVALIDO',
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
        'AUDIO_INVALIDO',
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

export async function detectarSilenciosAudio(
  rutaEntrada,
  {
    binario = ffmpegPath,
    duracionSegundos = 0,
    tiempoLimiteMs = 60000,
  } = {},
) {
  if (!binario) {
    throw new ErrorHttp(
      503,
      'El analizador de audio no está disponible.',
      'FFMPEG_NO_DISPONIBLE',
    );
  }

  const argumentos = [
    '-nostdin',
    '-hide_banner',
    '-i',
    rutaEntrada,
    '-af',
    'silencedetect=noise=-35dB:d=0.35',
    '-f',
    'null',
    '-',
  ];

  return new Promise((resolve, reject) => {
    const proceso = spawn(binario, argumentos, {
      stdio: ['ignore', 'ignore', 'pipe'],
    });
    let diagnostico = '';
    let excedioTiempo = false;
    const temporizador = setTimeout(() => {
      excedioTiempo = true;
      proceso.kill('SIGKILL');
    }, tiempoLimiteMs);

    proceso.stderr.setEncoding('utf8');
    proceso.stderr.on('data', (fragmento) => {
      if (diagnostico.length < 1024 * 1024) {
        diagnostico += fragmento;
      }
    });
    proceso.once('error', () => {
      clearTimeout(temporizador);
      reject(
        new ErrorHttp(
          503,
          'No fue posible iniciar el analizador de audio.',
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
            'El análisis temporal del audio tardó demasiado.',
            'ANALISIS_AUDIO_AGOTADO',
          ),
        );
        return;
      }
      if (codigo !== 0) {
        reject(
          new ErrorHttp(
            422,
            'No fue posible medir los silencios del audio.',
            'AUDIO_INVALIDO',
          ),
        );
        return;
      }

      const silencios = [];
      const inicios = [
        ...diagnostico.matchAll(/silence_start:\s*([0-9.]+)/g),
      ].map((coincidencia) => Number(coincidencia[1]));
      const finales = [
        ...diagnostico.matchAll(
          /silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)/g,
        ),
      ];
      for (let indice = 0; indice < finales.length; indice++) {
        const fin = Number(finales[indice][1]);
        const duracion = Number(finales[indice][2]);
        const inicio = Number.isFinite(inicios[indice])
          ? inicios[indice]
          : fin - duracion;
        if (Number.isFinite(inicio) && Number.isFinite(fin) && fin > inicio) {
          silencios.push({ start: Math.max(0, inicio), end: fin });
        }
      }
      if (inicios.length > finales.length && duracionSegundos > 0) {
        const inicio = inicios.at(-1);
        if (Number.isFinite(inicio) && duracionSegundos > inicio) {
          silencios.push({ start: inicio, end: duracionSegundos });
        }
      }
      resolve(silencios);
    });
  });
}

function interseccionSegundos(inicio, fin, silencios) {
  return silencios.reduce((total, silencio) => {
    const interseccion =
      Math.min(fin, silencio.end) - Math.max(inicio, silencio.start);
    return total + Math.max(0, interseccion);
  }, 0);
}

function letrasDePalabra(texto) {
  return [...texto].filter((caracter) => /\p{L}/u.test(caracter)).length;
}

export function crearEvidenciaHabla({
  palabras,
  silencios,
  duracionSegundos,
}) {
  if (!Array.isArray(palabras) || palabras.length === 0) {
    return null;
  }
  const ordenadas = [...palabras].sort((a, b) => a.start - b.start);
  const inicioVoz = ordenadas[0].start;
  const finVoz = ordenadas.at(-1).end;
  const pausas = (Array.isArray(silencios) ? silencios : [])
    .map((silencio) => ({
      start: Math.max(inicioVoz, silencio.start),
      end: Math.min(finVoz, silencio.end),
    }))
    .filter(
      (silencio) =>
        silencio.end - silencio.start >= MIN_PAUSE_SECONDS &&
        silencio.end > silencio.start,
    )
    .map((silencio) => ({
      ...silencio,
      duration: silencio.end - silencio.start,
    }));

  const alargamientos = ordenadas
    .map((palabra, indice) => {
      const duracion = palabra.end - palabra.start;
      const silencio = interseccionSegundos(
        palabra.start,
        palabra.end,
        pausas,
      );
      const duracionConVoz = Math.max(0, duracion - silencio);
      const cantidadLetras = Math.max(1, letrasDePalabra(palabra.word));
      const duracionEsperada = Math.max(0.22, cantidadLetras * 0.09);
      const esAlargamiento =
        duracionConVoz >= MIN_ELONGATION_SECONDS &&
        duracionConVoz >= duracionEsperada * 2.2;
      if (!esAlargamiento) return null;
      return {
        word: palabra.word,
        wordIndex: indice,
        start: palabra.start,
        end: palabra.end,
        duration: duracionConVoz,
      };
    })
    .filter(Boolean);

  const anotaciones = [];
  let indicePausa = 0;
  for (let indice = 0; indice < ordenadas.length; indice++) {
    const palabra = ordenadas[indice];
    while (
      indicePausa < pausas.length &&
      pausas[indicePausa].end <= palabra.end
    ) {
      anotaciones.push(
        `[pausa ${pausas[indicePausa].duration.toFixed(1)} s]`,
      );
      indicePausa++;
    }
    anotaciones.push(palabra.word);
    const alargamiento = alargamientos.find(
      (elemento) => elemento.wordIndex === indice,
    );
    if (alargamiento) {
      anotaciones.push(
        `[alargamiento ${alargamiento.duration.toFixed(1)} s]`,
      );
    }
  }
  while (indicePausa < pausas.length) {
    anotaciones.push(`[pausa ${pausas[indicePausa].duration.toFixed(1)} s]`);
    indicePausa++;
  }

  return {
    annotatedTranscript: anotaciones.join(' '),
    pauses: pausas,
    elongations: alargamientos.map(({ wordIndex, ...elemento }) => elemento),
    method: 'ffmpeg-silencedetect+whisper-word-timestamps',
    duration: duracionSegundos,
  };
}

export async function analizarEvidenciaHablaAudio({
  rutaAudio,
  palabras,
  duracionSegundos,
}) {
  const silencios = await detectarSilenciosAudio(rutaAudio, {
    duracionSegundos,
  });
  return crearEvidenciaHabla({
    palabras,
    silencios,
    duracionSegundos,
  });
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

function obtenerConfiguracionAzure(configuracionInyectada) {
  if (configuracionInyectada !== undefined) {
    return configuracionInyectada;
  }

  const clavePrimaria =
    process.env.AZURE_SPEECH_KEY_PRIMARY?.trim() ||
    process.env.AZURE_SPEECH_KEY?.trim();
  const claveSecundaria = process.env.AZURE_SPEECH_KEY_SECONDARY?.trim();
  const region = process.env.AZURE_SPEECH_REGION?.trim().toLowerCase();
  const endpoint = process.env.AZURE_SPEECH_ENDPOINT?.trim();
  if (!clavePrimaria || !region) {
    return null;
  }
  return {
    clavePrimaria,
    claveSecundaria,
    region,
    endpoint,
  };
}

function endpointPronunciacionAzure(configuracion, idioma) {
  let endpoint;
  if (configuracion.endpoint) {
    endpoint = new URL(configuracion.endpoint);
    if (endpoint.protocol !== 'https:') {
      throw new ErrorHttp(
        503,
        'El endpoint de Azure Speech debe utilizar HTTPS.',
        'AZURE_ENDPOINT_INVALIDO',
      );
    }
    endpoint.pathname =
      '/stt/speech/recognition/conversation/cognitiveservices/v1';
  } else {
    endpoint = new URL(
      `https://${configuracion.region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1`,
    );
  }
  endpoint.searchParams.set('language', idioma);
  endpoint.searchParams.set('format', 'detailed');
  return endpoint;
}

function numeroAzure(valor) {
  return typeof valor === 'number' && Number.isFinite(valor) ? valor : null;
}

function normalizarResultadoPronunciacion(resultado) {
  const mejor = Array.isArray(resultado?.NBest) ? resultado.NBest[0] : null;
  if (!mejor) {
    throw new ErrorHttp(
      502,
      'Azure Speech no devolvió una evaluación de pronunciación.',
      'AZURE_SIN_EVALUACION',
    );
  }

  return {
    provider: 'azure-speech',
    pronunciationScore: numeroAzure(mejor.PronScore),
    accuracyScore: numeroAzure(mejor.AccuracyScore),
    fluencyScore: numeroAzure(mejor.FluencyScore),
    completenessScore: numeroAzure(mejor.CompletenessScore),
    prosodyScore: numeroAzure(mejor.ProsodyScore),
    words: Array.isArray(mejor.Words)
      ? mejor.Words.map((palabra) => ({
          word: typeof palabra.Word === 'string' ? palabra.Word : '',
          accuracyScore: numeroAzure(palabra.AccuracyScore),
          errorType:
            typeof palabra.ErrorType === 'string' ? palabra.ErrorType : null,
          phonemes: Array.isArray(palabra.Phonemes)
            ? palabra.Phonemes.map((fonema) => ({
                phoneme:
                  typeof fonema.Phoneme === 'string' ? fonema.Phoneme : '',
                accuracyScore: numeroAzure(fonema.AccuracyScore),
              }))
            : [],
        }))
      : [],
  };
}

export async function evaluarPronunciacionAzure({
  rutaWav,
  textoReferencia,
  idioma = 'en-US',
  configuracion,
  fetchImpl = globalThis.fetch,
}) {
  const claves = [
    configuracion.clavePrimaria,
    configuracion.claveSecundaria,
  ].filter((clave, indice, lista) => clave && lista.indexOf(clave) === indice);
  const parametros = Buffer.from(
    JSON.stringify({
      ReferenceText: textoReferencia,
      GradingSystem: 'HundredMark',
      Granularity: 'Phoneme',
      Dimension: 'Comprehensive',
      EnableMiscue: 'True',
      EnableProsodyAssessment: 'True',
    }),
    'utf8',
  ).toString('base64');
  const audio = await readFile(rutaWav);
  const endpoint = endpointPronunciacionAzure(configuracion, idioma);

  for (let indice = 0; indice < claves.length; indice++) {
    const respuesta = await fetchImpl(endpoint, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
        'Ocp-Apim-Subscription-Key': claves[indice],
        'Pronunciation-Assessment': parametros,
      },
      body: audio,
    });

    if ((respuesta.status === 401 || respuesta.status === 403) &&
        indice < claves.length - 1) {
      continue;
    }
    if (respuesta.status === 401 || respuesta.status === 403) {
      throw new ErrorHttp(
        502,
        'Azure Speech rechazó las claves configuradas.',
        'AZURE_AUTENTICACION',
      );
    }
    if (!respuesta.ok) {
      throw new ErrorHttp(
        502,
        'Azure Speech no pudo evaluar la pronunciación.',
        'ERROR_AZURE',
      );
    }
    return normalizarResultadoPronunciacion(await respuesta.json());
  }

  throw new ErrorHttp(
    503,
    'Azure Speech no tiene claves configuradas.',
    'AZURE_NO_CONFIGURADO',
  );
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

function localeAzure(idiomaSolicitado, idiomaDetectado) {
  const idioma = (idiomaSolicitado || idiomaDetectado || '')
    .trim()
    .toLowerCase();
  const locales = {
    en: 'en-US',
    english: 'en-US',
    es: 'es-MX',
    spanish: 'es-MX',
    español: 'es-MX',
    fr: 'fr-FR',
    french: 'fr-FR',
    de: 'de-DE',
    german: 'de-DE',
    it: 'it-IT',
    italian: 'it-IT',
    pt: 'pt-BR',
    portuguese: 'pt-BR',
  };
  return locales[idioma] ?? 'en-US';
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
  azureConfig,
  azureFetch = globalThis.fetch,
  corsOrigin = process.env.CORS_ORIGIN ?? '*',
  maxAudioBytes = MAX_AUDIO_BYTES,
  convertirOpus = convertirOpusAFlac,
  convertirAzure = convertirAudioAWav,
  analizarHabla = analizarEvidenciaHablaAudio,
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
      let rutaAzure;
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
          response_format: 'verbose_json',
          timestamp_granularities: ['word', 'segment'],
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

        const configuracionAzure = obtenerConfiguracionAzure(azureConfig);
        let pronunciacion = null;
        let errorPronunciacion = null;
        if (
          configuracionAzure &&
          resultado.text?.trim() &&
          (!resultado.duration ||
            resultado.duration <= MAX_AZURE_PRONUNCIATION_SECONDS)
        ) {
          try {
            rutaAzure = await convertirAzure(request.file.path, {
              maxAudioBytes,
            });
            pronunciacion = await evaluarPronunciacionAzure({
              rutaWav: rutaAzure,
              textoReferencia: resultado.text.trim(),
              idioma: localeAzure(language, resultado.language),
              configuracion: configuracionAzure,
              fetchImpl: azureFetch,
            });
          } catch (error) {
            errorPronunciacion = {
              code: error?.code ?? 'ERROR_AZURE',
              message:
                error?.message ??
                'No fue posible evaluar la pronunciación con Azure Speech.',
            };
          }
        } else if (
          configuracionAzure &&
          resultado.duration > MAX_AZURE_PRONUNCIATION_SECONDS
        ) {
          errorPronunciacion = {
            code: 'AUDIO_AZURE_DEMASIADO_LARGO',
            message:
              'La evaluación de pronunciación admite audios de hasta 30 segundos.',
          };
        }

        const numeroFinito = (valor) =>
          typeof valor === 'number' && Number.isFinite(valor) ? valor : null;
        const palabras = Array.isArray(resultado.words)
          ? resultado.words
              .map((palabra) => ({
                word:
                  typeof palabra.word === 'string' ? palabra.word.trim() : '',
                start: numeroFinito(palabra.start),
                end: numeroFinito(palabra.end),
              }))
              .filter(
                (palabra) =>
                  palabra.word &&
                  palabra.start !== null &&
                  palabra.end !== null,
              )
          : [];
        const segmentos = Array.isArray(resultado.segments)
          ? resultado.segments
              .map((segmento) => ({
                id: Number.isInteger(segmento.id) ? segmento.id : null,
                text:
                  typeof segmento.text === 'string'
                    ? segmento.text.trim()
                    : '',
                start: numeroFinito(segmento.start),
                end: numeroFinito(segmento.end),
                avgLogprob: numeroFinito(segmento.avg_logprob),
                compressionRatio: numeroFinito(segmento.compression_ratio),
                noSpeechProb: numeroFinito(segmento.no_speech_prob),
              }))
              .filter(
                (segmento) =>
                  segmento.start !== null && segmento.end !== null,
              )
          : [];
        const ultimaMarca = [...palabras, ...segmentos].reduce(
          (maximo, elemento) => Math.max(maximo, elemento.end ?? 0),
          0,
        );
        const duracion = numeroFinito(resultado.duration) ?? ultimaMarca;
        let evidenciaHabla = null;
        if (palabras.length > 0 && duracion > 0) {
          try {
            evidenciaHabla = await analizarHabla({
              rutaAudio: request.file.path,
              palabras,
              duracionSegundos: duracion,
            });
          } catch {
            // La transcripción sigue siendo útil si el análisis acústico
            // complementario no está disponible para un archivo concreto.
          }
        }

        const cuerpoRespuesta = {
          transcription: resultado.text,
          model: GROQ_MODEL,
          language:
            typeof resultado.language === 'string'
              ? resultado.language
              : language || null,
          duration: duracion,
          words: palabras,
          segments: segmentos,
          speechEvidence: evidenciaHabla,
          pronunciation: pronunciacion,
          pronunciationError: errorPronunciacion,
        };

        // Termina la limpieza antes de responder para no dejar archivos de
        // audio accesibles durante unos milisegundos después de la solicitud.
        await unlink(request.file.path).catch(() => {});
        if (rutaConvertida) {
          await unlink(rutaConvertida).catch(() => {});
          rutaConvertida = null;
        }
        if (rutaAzure) {
          await unlink(rutaAzure).catch(() => {});
          rutaAzure = null;
        }
        response.json(cuerpoRespuesta);
      } catch (error) {
        next(error);
      } finally {
        await unlink(request.file.path).catch(() => {});
        if (rutaConvertida) {
          await unlink(rutaConvertida).catch(() => {});
        }
        if (rutaAzure) {
          await unlink(rutaAzure).catch(() => {});
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
