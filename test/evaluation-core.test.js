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
import {
  pronunciationFromPhoneticJudge,
  provisionalFluencyFromTimings,
  runAssessment,
} from '../src/evaluation/engine.js';

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

test('conserva la lengua materna como contexto sin cambiar el idioma objetivo', () => {
  const draft = createRubricDraft({
    mode: 'spontaneous',
    targetLocale: 'en-US',
    cefr: 'B1',
    nativeLanguage: 'es',
    instruction: 'Describe a memorable trip.',
  });

  assert.equal(draft.spec.nativeLanguage, 'es');
  assert.equal(draft.spec.targetLocale, 'en-US');
  assert.throws(
    () =>
      createRubricDraft({
        mode: 'spontaneous',
        targetLocale: 'en-US',
        cefr: 'B1',
        nativeLanguage: 'invalid-language',
        instruction: 'Describe a memorable trip.',
      }),
    /lengua materna/i,
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

test('puntúa fluidez con cualquier secuencia fonética temporizada', () => {
  const result = provisionalFluencyFromTimings({
    phoneticEvidence: {
      confidence: 0.31,
      events: [
        { phoneme: 'h', startSec: 0, endSec: 0.12 },
        { phoneme: 'ə', startSec: 0.14, endSec: 0.28 },
        { phoneme: 'l', startSec: 0.72, endSec: 0.84 },
        { phoneme: 'oʊ', startSec: 0.86, endSec: 1.04 },
      ],
    },
    whisperEvidence: { text: 'hello' },
    quality: {
      metrics: { durationSeconds: 1.2, voicedSeconds: 0.9 },
    },
    constructScope: 'productive_speaking',
  });

  assert.equal(result.status, 'scored');
  assert.equal(result.reliability, 'low');
  assert.equal(result.methodId, 'wav2vec2-timing-provisional-v1');
  assert.ok(result.score >= 0 && result.score <= 100);
  assert.match(result.evidence[0].claim, /4\.4 fonemas\/s/);
});

test('no descarta pronunciación cuando el juez considera limitada la confianza', () => {
  const result = pronunciationFromPhoneticJudge({
    judged: {
      status: 'insufficientEvidence',
      band: 0,
      rationale: 'La confianza CTC es moderada.',
      observations: [],
    },
    error: null,
    phoneticEvidence: {
      transcript: 'ðæpən ɹoʊt',
      confidence: 0.42,
    },
    constructScope: 'productive_speaking',
  });

  assert.equal(result.status, 'scored');
  assert.equal(result.score, 25);
  assert.equal(result.reliability, 'low');
  assert.match(result.limitations.join(' '), /en lugar de descartarse/i);
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
  const logs = [];
  let evaluationInput;
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
    linguisticClient: {},
    logger: {
      info(message) {
        logs.push(JSON.parse(message));
      },
    },
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
    ejecutarEvaluacion: async (input) => {
      evaluationInput = input;
      const { rubric, requestId } = input;
      return {
      schemaVersion: '2.0.0',
      requestId,
      assessmentId: '11111111-1111-4111-8111-111111111111',
      status: 'needsReview',
      taskSnapshot: {
        mode: rubric.spec.mode,
        targetLocale: rubric.spec.targetLocale,
        cefr: rubric.spec.cefr,
      },
      quality: {
        status: 'accepted',
        reasons: [],
        warnings: [],
        metrics: { durationSeconds: 12, voicedSeconds: 11 },
      },
      dimensions: {
        communication: {
          status: 'insufficientEvidence',
          score: null,
          reasonCode: 'LINGUISTIC_EVIDENCE_MISSING',
        },
      },
      provenance: {
        promptVersion: 'gordon-evidence-v1.2',
        providers: { whisper: { provider: 'groq' } },
      },
      consentStorage: { requested: false, stored: false },
      providerEvidence: {},
      };
    },
  });

  const draftResponse = await request(app)
    .post('/api/v2/rubrics/draft')
    .send({
      mode: 'spontaneous',
      targetLocale: 'en-US',
      cefr: 'B1',
      nativeLanguage: 'es',
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
    .field('phoneticTranscript', 'ðæpən ɹoʊt')
    .field('phoneticConfidence', '0.702')
    .field('phoneticModel', 'wav2vec2-phoneme-en')
    .field(
      'phoneticEvents',
      JSON.stringify([
        {
          type: 'phoneme',
          phoneme: 'ð',
          startSec: 0.1,
          endSec: 0.2,
          confidence: 0.61,
        },
      ]),
    )
    .attach('audio', wavPcm16({ durationSeconds: 1 }), {
      filename: 'student.wav',
      contentType: 'audio/wav',
    })
    .expect(200);

  assert.equal(assessment.body.schemaVersion, '2.0.0');
  assert.equal(assessment.body.status, 'needsReview');
  assert.equal(assessment.body.consentStorage.stored, false);
  assert.equal(evaluationInput.rubric.spec.nativeLanguage, 'es');
  assert.deepEqual(evaluationInput.phoneticEvidence, {
    transcript: 'ðæpən ɹoʊt',
    confidence: 0.702,
    model: 'wav2vec2-phoneme-en',
    events: [
      {
        type: 'phoneme',
        phoneme: 'ð',
        startSec: 0.1,
        endSec: 0.2,
        confidence: 0.61,
      },
    ],
  });
  const completionLog = logs.find(
    (entry) => entry.event === 'assessment_completed',
  );
  assert.equal(completionLog.quality.durationSeconds, 12);
  assert.equal(completionLog.quality.voicedSeconds, 11);
  assert.equal(
    completionLog.dimensions.communication.reasonCode,
    'LINGUISTIC_EVIDENCE_MISSING',
  );
});

test('mejora una consigna sin ocultar ni reemplazar el texto original', async () => {
  let specRecibida;
  const app = createApp({
    groqClient: {},
    linguisticClient: {},
    mejorarConsigna: async ({ spec }) => {
      specRecibida = spec;
      return {
        originalInstruction: spec.instruction,
        improvedInstruction:
          'Narra una experiencia pasada y explica qué podías hacer antes y qué eres capaz de hacer ahora.',
        detectedAudience: 'evaluator',
        summary: 'La instrucción se convirtió en una tarea para el estudiante.',
        preservedRequirements: [
          'Usar pasado simple',
          'Usar can, could y be able to',
        ],
        warnings: [],
        generatedBy: {
          provider: 'opencode-zen',
          model: 'test-model',
        },
      };
    },
  });

  const response = await request(app)
    .post('/api/v2/instructions/improve')
    .send({
      mode: 'spontaneous',
      targetLocale: 'en-US',
      cefr: 'B1',
      instruction:
        'Fíjate que use pasado simple, can, could y be able to.',
      communicativePurpose:
        'Narrar una experiencia o una secuencia de hechos',
    })
    .expect(200);

  assert.equal(specRecibida.mode, 'spontaneous');
  assert.equal(
    response.body.originalInstruction,
    'Fíjate que use pasado simple, can, could y be able to.',
  );
  assert.match(response.body.improvedInstruction, /Narra una experiencia/);
  assert.equal(response.body.generatedBy.reviewRequired, true);
  assert.match(response.body.generatedBy.promptHash, /^[a-f0-9]{64}$/);
});

test('el endpoint v2 nunca evalúa sin bytes de audio', async () => {
  await request(createApp())
    .post('/api/v2/assessments')
    .expect(400)
    .expect(({ body }) => {
      assert.equal(body.error.code, 'AUDIO_REQUIRED');
    });
});
