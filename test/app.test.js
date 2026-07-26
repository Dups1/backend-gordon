import assert from 'node:assert/strict';
import { access, copyFile, readFile } from 'node:fs/promises';
import test from 'node:test';

import request from 'supertest';

import { createApp, GROQ_MODEL } from '../src/app.js';

function clienteGroqFalso(crearTranscripcion) {
  return {
    audio: {
      transcriptions: {
        create: crearTranscripcion,
      },
    },
  };
}

test('expone el estado del servicio y el modelo configurado', async () => {
  const response = await request(createApp()).get('/health').expect(200);

  assert.deepEqual(response.body, {
    ok: true,
    service: 'backend-gordon',
    model: GROQ_MODEL,
  });
});

test('requiere el archivo multipart en el campo audio', async () => {
  const response = await request(createApp())
    .post('/api/transcriptions')
    .expect(400);

  assert.equal(response.body.error.code, 'AUDIO_REQUERIDO');
});

test('rechaza extensiones de archivo no admitidas', async () => {
  const response = await request(createApp())
    .post('/api/transcriptions')
    .attach('audio', Buffer.from('contenido'), {
      filename: 'grabacion.txt',
      contentType: 'text/plain',
    })
    .expect(415);

  assert.equal(response.body.error.code, 'FORMATO_NO_ADMITIDO');
});

test('envía el audio a Groq y devuelve la transcripción', async () => {
  let rutaTemporal;
  let opcionesRecibidas;
  const groqClient = clienteGroqFalso(async (opciones) => {
    opcionesRecibidas = opciones;
    rutaTemporal = opciones.file.path;
    assert.equal((await readFile(rutaTemporal)).toString(), 'audio simulado');
    opciones.file.destroy();
    return {
      text: 'Transcripción completada.',
      language: 'es',
      duration: 3.4,
      words: [
        { word: 'Transcripción', start: 0.2, end: 1.1 },
        { word: 'completada.', start: 2.1, end: 3.1 },
      ],
      segments: [
        {
          id: 0,
          text: 'Transcripción completada.',
          start: 0.2,
          end: 3.1,
          avg_logprob: -0.24,
          compression_ratio: 1.18,
          no_speech_prob: 0.04,
        },
      ],
    };
  });

  const response = await request(createApp({ groqClient }))
    .post('/api/transcriptions')
    .field('language', 'ES')
    .field('prompt', 'Conversación académica')
    .attach('audio', Buffer.from('audio simulado'), {
      filename: 'grabacion.webm',
      contentType: 'audio/webm',
    })
    .expect(200);

  assert.equal(opcionesRecibidas.model, 'whisper-large-v3');
  assert.equal(opcionesRecibidas.language, 'es');
  assert.equal(opcionesRecibidas.prompt, 'Conversación académica');
  assert.equal(opcionesRecibidas.response_format, 'verbose_json');
  assert.deepEqual(opcionesRecibidas.timestamp_granularities, [
    'word',
    'segment',
  ]);
  assert.equal(opcionesRecibidas.temperature, 0);
  assert.deepEqual(response.body, {
    transcription: 'Transcripción completada.',
    model: 'whisper-large-v3',
    language: 'es',
    duration: 3.4,
    words: [
      { word: 'Transcripción', start: 0.2, end: 1.1 },
      { word: 'completada.', start: 2.1, end: 3.1 },
    ],
    segments: [
      {
        id: 0,
        text: 'Transcripción completada.',
        start: 0.2,
        end: 3.1,
        avgLogprob: -0.24,
        compressionRatio: 1.18,
        noSpeechProb: 0.04,
      },
    ],
    pronunciation: null,
    pronunciationError: null,
  });
  await assert.rejects(access(rutaTemporal));
});

test('elimina el archivo temporal aunque Groq responda con error', async () => {
  let rutaTemporal;
  const groqClient = clienteGroqFalso(async (opciones) => {
    rutaTemporal = opciones.file.path;
    opciones.file.destroy();
    throw new Error('fallo simulado');
  });

  const response = await request(createApp({ groqClient }))
    .post('/api/transcriptions')
    .attach('audio', Buffer.from('audio simulado'), {
      filename: 'grabacion.wav',
      contentType: 'audio/wav',
    })
    .expect(502);

  assert.equal(response.body.error.code, 'ERROR_GROQ');
  await assert.rejects(access(rutaTemporal));
});

test('convierte OPUS antes de enviarlo a Groq y limpia ambos archivos', async () => {
  let rutaOriginal;
  let rutaConvertida;
  let rutaEnviadaAGroq;
  const convertirOpus = async (ruta) => {
    rutaOriginal = ruta;
    rutaConvertida = `${ruta}.flac`;
    await copyFile(ruta, rutaConvertida);
    return rutaConvertida;
  };
  const groqClient = clienteGroqFalso(async (opciones) => {
    rutaEnviadaAGroq = opciones.file.path;
    opciones.file.destroy();
    return { text: 'Audio OPUS transcrito.' };
  });

  const response = await request(createApp({ groqClient, convertirOpus }))
    .post('/api/transcriptions')
    .attach('audio', Buffer.from('opus simulado'), {
      filename: 'nota-de-voz.opus',
      contentType: 'audio/opus',
    })
    .expect(200);

  assert.equal(rutaEnviadaAGroq, rutaConvertida);
  assert.match(rutaEnviadaAGroq, /\.flac$/);
  assert.equal(response.body.transcription, 'Audio OPUS transcrito.');
  await assert.rejects(access(rutaOriginal));
  await assert.rejects(access(rutaConvertida));
});

test('rota a la clave secundaria de Azure ante un error de autenticación', async () => {
  let rutaWav;
  const clavesRecibidas = [];
  const convertirAzure = async (ruta) => {
    rutaWav = `${ruta}.wav`;
    await copyFile(ruta, rutaWav);
    return rutaWav;
  };
  const groqClient = clienteGroqFalso(async (opciones) => {
    opciones.file.destroy();
    return {
      text: 'Good morning.',
      language: 'English',
      duration: 2,
      words: [
        { word: 'Good', start: 0, end: 0.5 },
        { word: 'morning.', start: 0.5, end: 1.2 },
      ],
      segments: [],
    };
  });
  const azureFetch = async (_url, opciones) => {
    clavesRecibidas.push(opciones.headers['Ocp-Apim-Subscription-Key']);
    if (clavesRecibidas.length === 1) {
      return new Response('{}', { status: 401 });
    }
    return Response.json({
      RecognitionStatus: 'Success',
      NBest: [
        {
          PronScore: 88.4,
          AccuracyScore: 86.2,
          FluencyScore: 80.5,
          CompletenessScore: 100,
          ProsodyScore: 79.1,
          Words: [
            {
              Word: 'morning',
              AccuracyScore: 72.3,
              ErrorType: 'Mispronunciation',
              Phonemes: [
                { Phoneme: 'ɔː', AccuracyScore: 61.2 },
              ],
            },
          ],
        },
      ],
    });
  };

  const response = await request(
    createApp({
      groqClient,
      convertirAzure,
      azureFetch,
      azureConfig: {
        clavePrimaria: 'clave-primaria-prueba',
        claveSecundaria: 'clave-secundaria-prueba',
        region: 'southcentralus',
        endpoint:
          'https://gordon-speech-pronunciation.cognitiveservices.azure.com/',
      },
    }),
  )
    .post('/api/transcriptions')
    .field('language', 'en')
    .attach('audio', Buffer.from('audio simulado'), {
      filename: 'grabacion.wav',
      contentType: 'audio/wav',
    })
    .expect(200);

  assert.deepEqual(clavesRecibidas, [
    'clave-primaria-prueba',
    'clave-secundaria-prueba',
  ]);
  assert.deepEqual(response.body.pronunciation, {
    provider: 'azure-speech',
    pronunciationScore: 88.4,
    accuracyScore: 86.2,
    fluencyScore: 80.5,
    completenessScore: 100,
    prosodyScore: 79.1,
    words: [
      {
        word: 'morning',
        accuracyScore: 72.3,
        errorType: 'Mispronunciation',
        phonemes: [{ phoneme: 'ɔː', accuracyScore: 61.2 }],
      },
    ],
  });
  assert.equal(response.body.pronunciationError, null);
  assert.doesNotMatch(
    JSON.stringify(response.body),
    /clave-(primaria|secundaria)-prueba/,
  );
  await assert.rejects(access(rutaWav));
});

test('no rota claves de Azure ante errores ajenos a autenticación', async () => {
  let llamadasAzure = 0;
  const convertirAzure = async (ruta) => {
    const rutaWav = `${ruta}.wav`;
    await copyFile(ruta, rutaWav);
    return rutaWav;
  };
  const groqClient = clienteGroqFalso(async (opciones) => {
    opciones.file.destroy();
    return {
      text: 'Good morning.',
      duration: 2,
      words: [],
      segments: [],
    };
  });

  const response = await request(
    createApp({
      groqClient,
      convertirAzure,
      azureFetch: async () => {
        llamadasAzure++;
        return new Response('{}', { status: 500 });
      },
      azureConfig: {
        clavePrimaria: 'primaria',
        claveSecundaria: 'secundaria',
        region: 'southcentralus',
      },
    }),
  )
    .post('/api/transcriptions')
    .field('language', 'en')
    .attach('audio', Buffer.from('audio simulado'), {
      filename: 'grabacion.wav',
      contentType: 'audio/wav',
    })
    .expect(200);

  assert.equal(llamadasAzure, 1);
  assert.equal(response.body.pronunciation, null);
  assert.equal(response.body.pronunciationError.code, 'ERROR_AZURE');
});
