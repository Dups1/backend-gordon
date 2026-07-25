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
    return { text: 'Transcripción completada.' };
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
  assert.equal(opcionesRecibidas.response_format, 'json');
  assert.equal(opcionesRecibidas.temperature, 0);
  assert.deepEqual(response.body, {
    transcription: 'Transcripción completada.',
    model: 'whisper-large-v3',
    language: 'es',
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
