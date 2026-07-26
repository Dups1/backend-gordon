import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { promisify } from 'node:util';

import ffmpegPath from 'ffmpeg-static';
import request from 'supertest';

import { createApp } from '../src/app.js';
import {
  analyzeAudioQuality,
  inspectAudioSignature,
  prepareAudio,
} from '../src/evaluation/audio.js';
import {
  SCORE_PROFILES,
  calculateOverall,
  confirmRubric,
  createRubricDraft,
  dimensionResult,
  verifyRubricToken,
} from '../src/evaluation/domain.js';
import { runAssessment } from '../src/evaluation/engine.js';
import { normalizeAzureResponse } from '../src/evaluation/providers.js';

const execFileAsync = promisify(execFile);

function wavPcm16({ durationSeconds = 1, amplitude = 0 }) {
  const sampleRate = 16000;
  const samples = Math.round(durationSeconds * sampleRate);
  const dataBytes = samples * 2;
  const buffer = Buffer.alloc(44 + dataBytes);
  buffer.write('RIFF', 0);
  buffer.writeUInt32LE(36 + dataBytes, 4);
  buffer.write('WAVE', 8);
  buffer.write('fmt ', 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(sampleRate, 24);
  buffer.writeUInt32LE(sampleRate * 2, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write('data', 36);
  buffer.writeUInt32LE(dataBytes, 40);
  if (amplitude !== 0) {
    for (let index = 0; index < samples; index++) {
      const sample = Math.round(
        Math.sin((2 * Math.PI * 220 * index) / sampleRate) *
          amplitude *
          32767,
      );
      buffer.writeInt16LE(sample, 44 + index * 2);
    }
  }
  return buffer;
}

function spontaneousDraft() {
  return createRubricDraft({
    mode: 'spontaneous',
    targetLocale: 'en-US',
    cefr: 'B1',
    instruction: 'Explain a learning experience and justify its importance.',
  });
}

test('firma una rúbrica inmutable y rechaza una alteración', () => {
  const secret = 'secret-de-prueba-con-suficiente-entropia';
  const draft = spontaneousDraft();
  const confirmed = confirmRubric(draft, { secret });

  assert.equal(
    verifyRubricToken(confirmed.confirmedRubricToken, { secret }).status,
    'confirmed',
  );

  const [payload, signature] = confirmed.confirmedRubricToken.split('.');
  const decoded = JSON.parse(Buffer.from(payload, 'base64url').toString());
  decoded.rubric.spec.cefr = 'C2';
  const alteredPayload = Buffer.from(JSON.stringify(decoded)).toString(
    'base64url',
  );
  assert.throws(
    () => verifyRubricToken(`${alteredPayload}.${signature}`, { secret }),
    /verificarse/,
  );

  const invalidScope = structuredClone(draft);
  invalidScope.dimensions[0].constructScope = 'reading_realization';
  assert.throws(
    () => confirmRubric(invalidScope, { secret }),
    /alcance/,
  );
  const duplicateDimension = structuredClone(draft);
  duplicateDimension.dimensions[1].id = duplicateDimension.dimensions[0].id;
  assert.throws(
    () => confirmRubric(duplicateDimension, { secret }),
    /repetidas/,
  );
});

test('lectura exige texto canónico y conserva el alcance de realización', () => {
  assert.throws(
    () =>
      createRubricDraft({
        mode: 'reading',
        targetLocale: 'en-US',
        cefr: 'A2',
        instruction: 'Read the assigned paragraph.',
      }),
    /referenceText/,
  );
  const draft = createRubricDraft({
    mode: 'reading',
    targetLocale: 'en-US',
    cefr: 'A2',
    instruction: 'Read the assigned paragraph.',
    referenceText:
      'The student reads a sufficiently long reference paragraph for evaluation.',
  });
  assert.equal(
    draft.dimensions.find((item) => item.id === 'grammar').constructScope,
    'reading_realization',
  );
});

test('no limita valores inválidos ni redistribuye pesos faltantes', () => {
  const invalid = dimensionResult({
    id: 'pronunciation',
    status: 'scored',
    score: 130,
    methodId: 'provider',
  });
  assert.equal(invalid.status, 'providerError');
  assert.equal(invalid.score, null);
  const invalidRaw = dimensionResult({
    id: 'fluency',
    status: 'scored',
    score: 80,
    rawScore: -3,
    methodId: 'provider',
  });
  assert.equal(invalidRaw.status, 'providerError');
  const invalidInterval = dimensionResult({
    id: 'grammar',
    status: 'scored',
    score: 80,
    interval90: { low: -10, high: 90 },
    methodId: 'provider',
  });
  assert.equal(invalidInterval.status, 'providerError');

  const dimensions = Object.fromEntries(
    ['communication', 'pronunciation', 'grammar', 'vocabulary', 'fluency'].map(
      (id) => [
        id,
        dimensionResult({
          id,
          status: id === 'fluency' ? 'unavailable' : 'scored',
          score: id === 'fluency' ? null : 80.5,
          methodId: 'test',
        }),
      ],
    ),
  );
  const overall = calculateOverall(
    dimensions,
    SCORE_PROFILES.spontaneous,
  );
  assert.equal(overall.score, null);
  assert.deepEqual(overall.missingDimensions, ['fluency']);
});

test('rechaza valores inválidos de Azure en vez de limitarlos', () => {
  assert.throws(
    () =>
      normalizeAzureResponse(
        {
          RecognitionStatus: 'Success',
          NBest: [
            {
              Display: 'Hello.',
              PronunciationAssessment: {
                PronScore: 130,
                AccuracyScore: 80,
                FluencyScore: 75,
                ProsodyScore: 70,
              },
            },
          ],
        },
        {
          mode: 'spontaneous',
          locale: 'en-US',
          requestId: 'invalid-provider-score',
        },
      ),
    (error) =>
      error.code === 'INVALID_PROVIDER_SCORE' &&
      error.details?.field === 'pronunciationScore',
  );
});

test('el quality gate distingue silencio de voz y conserva decimales', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'gordon-quality-'));
  const silent = path.join(directory, 'silent.wav');
  const voice = path.join(directory, 'voice.wav');
  await Promise.all([
    writeFile(silent, wavPcm16({ durationSeconds: 2 })),
    writeFile(
      voice,
      wavPcm16({ durationSeconds: 12, amplitude: 0.25 }),
    ),
  ]);
  try {
    const silentQuality = await analyzeAudioQuality(silent);
    const voiceQuality = await analyzeAudioQuality(voice);
    assert.equal(silentQuality.scorable, false);
    assert.ok(silentQuality.reasons.includes('NO_SPEECH'));
    assert.equal(voiceQuality.scorable, true);
    assert.equal(voiceQuality.metrics.durationSeconds, 12);
    assert.ok(voiceQuality.metrics.rmsDbfs < 0);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('quality gate advierte clipping y respeta el límite de cinco minutos', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'gordon-limits-'));
  const clipped = path.join(directory, 'clipped.wav');
  const exactLimit = path.join(directory, 'exact-limit.wav');
  const overLimit = path.join(directory, 'over-limit.wav');
  await Promise.all([
    writeFile(
      clipped,
      wavPcm16({ durationSeconds: 12, amplitude: 1 }),
    ),
    writeFile(exactLimit, wavPcm16({ durationSeconds: 300 })),
    writeFile(overLimit, wavPcm16({ durationSeconds: 301 })),
  ]);
  try {
    const [clippedQuality, exactQuality, overQuality] = await Promise.all([
      analyzeAudioQuality(clipped),
      analyzeAudioQuality(exactLimit),
      analyzeAudioQuality(overLimit),
    ]);
    assert.ok(clippedQuality.warnings.includes('CLIPPING'));
    assert.ok(!exactQuality.reasons.includes('AUDIO_TOO_LONG'));
    assert.ok(overQuality.reasons.includes('AUDIO_TOO_LONG'));
    assert.equal(overQuality.scorable, false);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('normaliza MP3, Opus, WebM, M4A y FLAC por su firma real', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'gordon-formats-'));
  const source = path.join(directory, 'voice.wav');
  await writeFile(
    source,
    wavPcm16({ durationSeconds: 2, amplitude: 0.25 }),
  );
  const variants = [
    { name: 'voice.mp3', container: 'mp3', codec: ['-c:a', 'libmp3lame'] },
    { name: 'voice.opus', container: 'ogg', codec: ['-c:a', 'libopus'] },
    { name: 'voice.webm', container: 'webm', codec: ['-c:a', 'libopus'] },
    { name: 'voice.m4a', container: 'mp4', codec: ['-c:a', 'aac'] },
    { name: 'voice.flac', container: 'flac', codec: ['-c:a', 'flac'] },
  ];
  try {
    for (const variant of variants) {
      const filePath = path.join(directory, variant.name);
      await execFileAsync(ffmpegPath, [
        '-nostdin',
        '-hide_banner',
        '-loglevel',
        'error',
        '-y',
        '-i',
        source,
        ...variant.codec,
        filePath,
      ]);
      const prepared = await prepareAudio({
        filePath,
        originalName: variant.name,
        mode: 'spontaneous',
        referenceWordCount: 0,
      });
      assert.equal(prepared.signature.container, variant.container);
      assert.equal(prepared.quality.metrics.sampleRate, 16000);
      assert.equal(prepared.quality.metrics.channels, 1);
      await rm(prepared.normalizedPath, { force: true });
    }

    const spoofed = path.join(directory, 'spoofed.wav');
    const mp3Bytes = await readFile(path.join(directory, 'voice.mp3'));
    await writeFile(spoofed, mp3Bytes);
    await assert.rejects(
      inspectAudioSignature(spoofed, 'spoofed.wav'),
      (error) => error.code === 'AUDIO_CONTAINER_MISMATCH',
    );
    const corrupt = path.join(directory, 'corrupt.wav');
    await writeFile(corrupt, Buffer.from('not audio'));
    await assert.rejects(
      inspectAudioSignature(corrupt, 'corrupt.wav'),
      (error) => error.code === 'AUDIO_SIGNATURE_INVALID',
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('expone el flujo v2 síncrono con confirmación previa y audio real', async () => {
  const secret = 'rubric-secret-test';
  const storage = {
    configured: false,
    async saveAssessment() {
      throw new Error('No debe almacenar sin consentimiento.');
    },
    async saveHumanRating() {
      return { stored: true };
    },
    async deleteAssessment() {
      return { deleted: true, deletedObjects: 1 };
    },
  };
  const app = createApp({
    groqClient: {},
    rubricSigningSecret: secret,
    almacenamientoPiloto: storage,
    compilarRubrica: async ({ draft }) => draft,
    prepararAudio: async ({ filePath }) => ({
      normalizedPath: filePath,
      signature: {
        container: 'wav',
        originalBytes: 44,
        originalSha256: 'audio-hash',
      },
      quality: {
        status: 'accepted',
        scorable: true,
        reasons: [],
        warnings: [],
        metrics: { durationSeconds: 12, voicedSeconds: 11 },
      },
    }),
    ejecutarEvaluacion: async ({ rubric, requestId }) => ({
      schemaVersion: '2.0.0',
      requestId,
      assessmentId: '11111111-1111-4111-8111-111111111111',
      status: 'needsReview',
      taskSnapshot: {
        mode: rubric.spec.mode,
        targetLocale: rubric.spec.targetLocale,
        cefr: rubric.spec.cefr,
      },
      consentStorage: { requested: false, stored: false },
      providerEvidence: {},
    }),
  });

  const draftResponse = await request(app)
    .post('/api/v2/rubrics/draft')
    .send({
      mode: 'spontaneous',
      targetLocale: 'en-US',
      cefr: 'B1',
      instruction: 'Explain a learning experience.',
    })
    .expect(201);
  assert.equal(draftResponse.body.draft.status, 'draft');

  const confirmation = await request(app)
    .post('/api/v2/rubrics/confirm')
    .send({ draft: draftResponse.body.draft })
    .expect(200);

  const assessment = await request(app)
    .post('/api/v2/assessments')
    .field('confirmedRubricToken', confirmation.body.confirmedRubricToken)
    .attach('audio', wavPcm16({ durationSeconds: 1 }), {
      filename: 'student.wav',
      contentType: 'audio/wav',
    })
    .expect(200);

  assert.equal(assessment.body.schemaVersion, '2.0.0');
  assert.equal(assessment.body.status, 'needsReview');
  assert.equal(assessment.body.consentStorage.stored, false);
});

test('el endpoint v2 nunca evalúa sin bytes de audio', async () => {
  await request(createApp())
    .post('/api/v2/assessments')
    .expect(400)
    .expect(({ body }) => {
      assert.equal(body.error.code, 'AUDIO_REQUIRED');
    });
});

test('si Whisper falla conserva Azure y se abstiene en lingüística', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'gordon-whisper-down-'));
  const audioPath = path.join(directory, 'voice.wav');
  await writeFile(
    audioPath,
    wavPcm16({ durationSeconds: 12, amplitude: 0.25 }),
  );
  try {
    const quality = await analyzeAudioQuality(audioPath);
    const report = await runAssessment({
      rubric: spontaneousDraft(),
      originalAudio: audioPath,
      normalizedAudio: audioPath,
      signature: {
        container: 'wav',
        originalBytes: (await stat(audioPath)).size,
        originalSha256: 'original-hash',
      },
      quality,
      groqClient: {
        audio: {
          transcriptions: {
            create: async () => {
              const error = new Error('Whisper no disponible');
              error.status = 400;
              throw error;
            },
          },
        },
      },
      linguisticModel: 'test-linguistic',
      whisperModel: 'test-whisper',
      azureConfig: {
        clavePrimaria: 'test-key',
        region: 'test-region',
      },
      evaluateAzureRest: async () => {
        throw new Error('REST no debe usarse en modo espontáneo.');
      },
      evaluateAzureContinuous: async () => [
        {
          RecognitionStatus: 'Success',
          Offset: 0,
          Duration: 120000000,
          DisplayText: 'I learned from practice.',
          NBest: [
            {
              Lexical: 'i learned from practice',
              Display: 'I learned from practice.',
              PronunciationAssessment: {
                PronScore: 81.25,
                AccuracyScore: 82.5,
                FluencyScore: 77.75,
                ProsodyScore: 79.5,
              },
              Words: [
                {
                  Word: 'learned',
                  Offset: 1000000,
                  Duration: 5000000,
                  PronunciationAssessment: {
                    AccuracyScore: 82.5,
                    ErrorType: 'None',
                  },
                },
              ],
            },
          ],
        },
      ],
      requestId: 'request-whisper-down',
    });

    assert.equal(report.status, 'partial');
    assert.equal(report.transcript.primary.status, 'unavailable');
    assert.equal(report.dimensions.pronunciation.status, 'scored');
    assert.equal(report.dimensions.fluency.status, 'scored');
    assert.equal(report.dimensions.communication.status, 'providerError');
    assert.equal(report.dimensions.grammar.status, 'providerError');
    assert.equal(report.dimensions.vocabulary.status, 'providerError');
    assert.equal(report.overall.score, null);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
