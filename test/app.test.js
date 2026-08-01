import assert from 'node:assert/strict';
import { access, copyFile, readFile } from 'node:fs/promises';
import test from 'node:test';

import request from 'supertest';

import {
  createApp,
  crearEvidenciaHabla,
  GROQ_MODEL,
  LINGUISTIC_MODEL,
} from '../src/app.js';
import {
  WHISPER_LITERAL_POLICY_VERSION,
  whisperLiteralPrompt,
} from '../src/whisper.js';

function clienteGroqFalso(
  crearTranscripcion,
  crearAnalisis = async () => ({
    choices: [
      {
        message: {
          content: JSON.stringify({
            sufficientEvidence: true,
            score: 100,
            summary: 'No se detectaron errores gramaticales.',
            correctedText: 'Transcripción completada.',
            pedagogicalRecommendation: {
              focus: 'Amplía la respuesta.',
              action: 'Añade dos ejemplos relacionados con la consigna.',
              rationale:
                'La muestra es correcta, pero demasiado breve para desarrollar la idea.',
            },
            errors: [],
          }),
        },
      },
    ],
  }),
) {
  return {
    audio: {
      transcriptions: {
        create: crearTranscripcion,
      },
    },
    chat: {
      completions: {
        create: crearAnalisis,
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
    linguisticModel: LINGUISTIC_MODEL,
    phonemeEndpointConfigured: false,
    promptVersion: 'gordon-evidence-v1.14',
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

  const response = await request(
    createApp({ groqClient, linguisticClient: groqClient }),
  )
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
  assert.equal(opcionesRecibidas.prompt, whisperLiteralPrompt('es'));
  assert.doesNotMatch(opcionesRecibidas.prompt, /Conversación académica/);
  assert.equal(opcionesRecibidas.response_format, 'verbose_json');
  assert.deepEqual(opcionesRecibidas.timestamp_granularities, [
    'word',
    'segment',
  ]);
  assert.equal(opcionesRecibidas.temperature, 0);
  assert.deepEqual(response.body, {
    transcription: 'Transcripción completada.',
    model: 'whisper-large-v3',
    transcriptionPolicy: WHISPER_LITERAL_POLICY_VERSION,
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
    speechEvidence: null,
    grammar: {
      provider: 'opencode-zen',
      model: LINGUISTIC_MODEL,
      sufficientEvidence: true,
      score: 100,
      summary: 'No se detectaron errores gramaticales.',
      correctedText: 'Transcripción completada.',
      pedagogicalRecommendation: {
        focus: 'Amplía la respuesta.',
        action: 'Añade dos ejemplos relacionados con la consigna.',
        rationale:
          'La muestra es correcta, pero demasiado breve para desarrollar la idea.',
      },
      errors: [],
    },
    grammarError: null,
  });
  await assert.rejects(access(rutaTemporal));
});

test('detecta errores gramaticales con evidencia literal y descarta inventados', async () => {
  let opcionesGramatica;
  let opcionesWhisper;
  const groqClient = clienteGroqFalso(
    async (opciones) => {
      opcionesWhisper = opciones;
      opciones.file.destroy();
      return {
        text: 'She go to school every day.',
        language: 'English',
        duration: 2,
        words: [
          { word: 'She', start: 0, end: 0.2 },
          { word: 'go', start: 0.3, end: 0.5 },
          { word: 'to', start: 0.6, end: 0.7 },
          { word: 'school', start: 0.8, end: 1.1 },
          { word: 'every', start: 1.2, end: 1.5 },
          { word: 'day.', start: 1.6, end: 1.9 },
        ],
        segments: [],
      };
    },
    async (opciones) => {
      opcionesGramatica = opciones;
      return {
        choices: [
          {
            message: {
              content: JSON.stringify({
                sufficientEvidence: true,
                score: 72,
                summary: 'Hay un error de concordancia.',
                correctedText: 'She goes to school every day.',
                pedagogicalRecommendation: {
                  focus: 'Practica la concordancia verbal.',
                  action:
                    'Repite la respuesta cambiando el sujeto y ajustando el verbo.',
                  rationale:
                    'La forma “She go” muestra dificultad con la tercera persona.',
                },
                errors: [
                  {
                    original: 'She go',
                    correction: 'She goes',
                    category: 'Subject-verb agreement',
                    severity: 'moderate',
                    explanation:
                      'La tercera persona singular requiere “goes”.',
                  },
                  {
                    original: 'I has',
                    correction: 'I have',
                    category: 'Agreement',
                    severity: 'major',
                    explanation: 'Este fragmento no existe.',
                  },
                ],
              }),
            },
          },
        ],
      };
    },
  );

  const response = await request(
    createApp({ groqClient, linguisticClient: groqClient }),
  )
    .post('/api/transcriptions')
    .field('language', 'en')
    .field('evaluationPrompt', 'Evalúa concordancia para nivel A2.')
    .attach('audio', Buffer.from('audio simulado'), {
      filename: 'gramatica.wav',
      contentType: 'audio/wav',
    })
    .expect(200);

  assert.equal(opcionesGramatica.model, LINGUISTIC_MODEL);
  assert.equal(opcionesGramatica.response_format.type, 'json_object');
  assert.equal(opcionesGramatica.max_tokens, 8000);
  assert.deepEqual(opcionesGramatica.thinking, { type: 'disabled' });
  assert.match(
    opcionesGramatica.messages[1].content,
    /<instruccion_docente>\nEvalúa concordancia para nivel A2\.\n<\/instruccion_docente>/,
  );
  assert.match(
    opcionesGramatica.messages[1].content,
    /<evidencia_tecnica>\n.*"wordCount":6.*\n<\/evidencia_tecnica>/,
  );
  assert.equal(opcionesWhisper.prompt, whisperLiteralPrompt('en'));
  assert.equal(response.body.grammar.score, 72);
  assert.equal(response.body.grammar.errors.length, 1);
  assert.equal(response.body.grammar.errors[0].original, 'She go');
  assert.equal(
    response.body.grammar.pedagogicalRecommendation.focus,
    'Practica la concordancia verbal.',
  );
  assert.equal(response.body.grammarError, null);
});

test('rechaza instrucciones de evaluación mayores a 1000 caracteres', async () => {
  const response = await request(createApp())
    .post('/api/transcriptions')
    .field('evaluationPrompt', 'a'.repeat(1001))
    .attach('audio', Buffer.from('audio simulado'), {
      filename: 'evaluacion.wav',
      contentType: 'audio/wav',
    })
    .expect(400);

  assert.equal(response.body.error.code, 'INSTRUCCION_DEMASIADO_LARGA');
});

test('anota pausas sin inventar alargamientos fonéticos', () => {
  const evidencia = crearEvidenciaHabla({
    duracionSegundos: 4,
    palabras: [
      { word: 'Hello', start: 0, end: 0.45 },
      { word: 'my', start: 0.5, end: 0.75 },
      { word: 'name', start: 0.8, end: 1.15 },
      { word: 'is', start: 1.2, end: 2.7 },
      { word: 'David', start: 3.3, end: 3.8 },
    ],
    silencios: [{ start: 2.55, end: 3.25 }],
  });

  assert.equal(
    evidencia.annotatedTranscript,
    'Hello my name is ...... David',
  );
  assert.equal(evidencia.pauses.length, 1);
  assert.equal(evidencia.elongations.length, 0);
});

test('no confunde una pausa pegada al timestamp con un alargamiento', () => {
  const evidencia = crearEvidenciaHabla({
    duracionSegundos: 3.2,
    palabras: [
      { word: 'I', start: 0, end: 0.2 },
      { word: 'will', start: 0.25, end: 0.55 },
      { word: 'continue', start: 0.6, end: 2.8 },
    ],
    silencios: [{ start: 0.6, end: 2.05 }],
  });

  assert.equal(evidencia.pauses.length, 1);
  assert.equal(evidencia.elongations.length, 0);
  assert.match(evidencia.annotatedTranscript, /\.\.\.\.\.\./);
});

test('elimina el archivo temporal aunque Groq responda con error', async () => {
  let rutaTemporal;
  const groqClient = clienteGroqFalso(async (opciones) => {
    rutaTemporal = opciones.file.path;
    opciones.file.destroy();
    throw new Error('fallo simulado');
  });

  const response = await request(
    createApp({ groqClient, linguisticClient: groqClient }),
  )
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

  const response = await request(
    createApp({
      groqClient,
      linguisticClient: groqClient,
      convertirOpus,
    }),
  )
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
