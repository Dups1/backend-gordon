import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  normalizeRemotePhonemeResponse,
  phonemeAnalysisEndpoint,
  requestRemotePhonemeEvidence,
} from '../src/evaluation/phonetics.js';

test('acepta una URL base y añade la ruta del endpoint fonético', () => {
  assert.equal(
    phonemeAnalysisEndpoint('https://phonemes.example.test/'),
    'https://phonemes.example.test/api/v1/phonemes',
  );
  assert.equal(
    phonemeAnalysisEndpoint('https://phonemes.example.test/api/v1/phonemes'),
    'https://phonemes.example.test/api/v1/phonemes',
  );
});

test('normaliza la respuesta remota conservando eventos y confianza', () => {
  const evidence = normalizeRemotePhonemeResponse(
    {
      requestId: 'remote-1',
      phoneticTranscript: 'ðæpən ɹoʊt',
      model: 'wav2vec2-remote',
      meanConfidence: 0.7016,
      events: [
        {
          type: 'phoneme',
          phoneme: 'ð',
          startSec: 0.1,
          endSec: 0.2,
          confidence: 0.8,
        },
        { type: 'boundary', phoneme: '|', startSec: 0.8, endSec: 0.9 },
        { type: 'invalid', phoneme: 'x' },
      ],
    },
    'https://phonemes.example.test/api/v1/phonemes',
  );

  assert.equal(evidence.provider, 'remote-wav2vec2');
  assert.equal(evidence.transcript, 'ðæpən ɹoʊt');
  assert.equal(evidence.confidence, 0.7016);
  assert.equal(evidence.events.length, 2);
  assert.equal(evidence.events[0].phoneme, 'ð');
  assert.equal(evidence.events[1].confidence, null);
});

test('envía el WAV normalizado al endpoint remoto y devuelve su evidencia', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'gordon-phonetics-'));
  const audioPath = path.join(directory, 'normalized.wav');
  const audioBytes = Buffer.from('RIFF-normalized-wav');
  await writeFile(audioPath, audioBytes);
  let receivedUrl;
  let receivedOptions;

  try {
    const evidence = await requestRemotePhonemeEvidence({
      audioPath,
      endpoint: 'https://phonemes.example.test',
      apiKey: 'secret-token',
      requestId: 'gordon-request-1',
      fetchImpl: async (url, options) => {
        receivedUrl = url;
        receivedOptions = options;
        const file = options.body.get('audio');
        assert.equal(file.name, 'normalized.wav');
        assert.equal(file.type, 'audio/wav');
        assert.deepEqual(
          Buffer.from(await file.arrayBuffer()),
          audioBytes,
        );
        return new Response(
          JSON.stringify({
            requestId: 'remote-request-1',
            phoneticTranscript: 'ðæpən ɹoʊt',
            model: 'wav2vec2-remote',
            durationSeconds: 12.4,
            meanConfidence: 0.82,
            events: [],
          }),
          { status: 200, headers: { 'content-type': 'application/json' } },
        );
      },
    });

    assert.equal(
      receivedUrl,
      'https://phonemes.example.test/api/v1/phonemes',
    );
    assert.equal(receivedOptions.headers.Authorization, 'Bearer secret-token');
    assert.equal(receivedOptions.headers['X-Request-Id'], 'gordon-request-1');
    assert.equal(evidence.transcript, 'ðæpən ɹoʊt');
    assert.equal(evidence.durationSeconds, 12.4);
    assert.equal(evidence.confidence, 0.82);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
