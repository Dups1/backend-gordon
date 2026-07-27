import assert from 'node:assert/strict';
import test from 'node:test';

import {
  extractLinguisticEvidence,
  runDoubleLinguisticJudging,
} from '../src/evaluation/linguistic.js';
import {
  createContinuousPronunciationConfig,
  evaluateAzureV2,
} from '../src/evaluation/providers.js';

function structuredClient(responses) {
  const pending = [...responses];
  const calls = [];
  return {
    chat: {
      completions: {
        async create(options) {
          calls.push(options);
          const content = pending.shift();
          assert.ok(content, 'La prueba no configuró una respuesta del modelo.');
          return {
            choices: [{ message: { content: JSON.stringify(content) } }],
          };
        },
      },
    },
    get pendingResponses() {
      return pending.length;
    },
    get calls() {
      return calls;
    },
  };
}

function linguisticFindings() {
  return [
    {
      id: 'communication-1',
      dimension: 'communication',
      kind: 'coverage',
      claim: 'La respuesta desarrolla la secuencia solicitada.',
    },
    {
      id: 'grammar-1',
      dimension: 'grammar',
      kind: 'strength',
      claim: 'La respuesta utiliza pasado simple.',
    },
    {
      id: 'vocabulary-1',
      dimension: 'vocabulary',
      kind: 'strength',
      claim: 'La respuesta usa vocabulario pertinente.',
    },
  ];
}

function judgeResponse({ invalidCitations = false } = {}) {
  return {
    dimensions: [
      {
        id: 'communication',
        status: 'scored',
        band: 3,
        evidenceIds: invalidCitations ? ['missing-communication'] : [],
        rationale: 'Cumple la intención comunicativa.',
      },
      {
        id: 'grammar',
        status: 'scored',
        band: 2,
        evidenceIds: ['communication-1'],
        rationale: 'Muestra control parcial de la gramática objetivo.',
      },
      {
        id: 'vocabulary',
        status: 'scored',
        band: 3,
        evidenceIds: invalidCitations ? ['missing-vocabulary'] : [],
        rationale: 'El vocabulario es adecuado para la tarea.',
      },
    ],
  };
}

test('configura prosodia de Azure como propiedad booleana', async () => {
  const module = await import('microsoft-cognitiveservices-speech-sdk');
  const sdk = module.default ?? module;

  const enUs = createContinuousPronunciationConfig(sdk, 'en-US');
  const enUsJson = JSON.parse(enUs.toJSON());
  assert.equal(enUsJson.enableProsodyAssessment, true);
  assert.equal(enUsJson.phonemeAlphabet, 'IPA');
  assert.equal(enUsJson.nbestPhonemeCount, 5);

  const otherLocale = createContinuousPronunciationConfig(sdk, 'es-MX');
  assert.equal(
    JSON.parse(otherLocale.toJSON()).enableProsodyAssessment,
    false,
  );

  const reading = createContinuousPronunciationConfig(
    sdk,
    'en-US',
    'Read this canonical text.',
  );
  assert.equal(
    JSON.parse(reading.toJSON()).referenceText,
    'Read this canonical text.',
  );
});

test('usa una sola toma para audios de hasta 30 segundos', async () => {
  let restOptions;
  let continuousCalls = 0;
  const result = await evaluateAzureV2({
    normalizedPath: '/tmp/short.wav',
    durationSeconds: 30,
    rubric: {
      spec: {
        mode: 'spontaneous',
        targetLocale: 'en-US',
        referenceText: '',
      },
    },
    whisper: { words: [] },
    quality: {},
    azureConfig: {
      clavePrimaria: 'primary',
      region: 'test-region',
    },
    evaluateRest: async (options) => {
      restOptions = options;
      return {
        provider: 'azure-speech',
        pronunciationScore: 82.5,
        accuracyScore: 84.25,
        fluencyScore: 79.75,
        completenessScore: 99,
        prosodyScore: 78.5,
        words: [],
      };
    },
    evaluateContinuous: async () => {
      continuousCalls++;
      return [];
    },
  });

  assert.equal(continuousCalls, 0);
  assert.equal(restOptions.rutaWav, '/tmp/short.wav');
  assert.equal(restOptions.textoReferencia, '');
  assert.equal(restOptions.mode, 'spontaneous');
  assert.equal(result.recognitionMode, 'single-shot');
  assert.equal(result.recognitionThresholdSeconds, 30);
  assert.equal(result.completenessScore, null);
});

test('rota a modo continuo después de 30 segundos y reconstruye omisiones e inserciones de lectura', async () => {
  let restCalls = 0;
  let continuousOptions;
  const referenceText = 'alpha beta gamma delta epsilon';
  const spokenWords = ['alpha', 'gamma', 'zeta', 'delta', 'epsilon', 'extra'];
  const result = await evaluateAzureV2({
    normalizedPath: '/tmp/long.wav',
    durationSeconds: 30.1,
    rubric: {
      spec: {
        mode: 'reading',
        targetLocale: 'en-US',
        referenceText,
      },
    },
    whisper: { words: [] },
    quality: {},
    azureConfig: {
      clavePrimaria: 'primary',
      region: 'test-region',
    },
    evaluateRest: async () => {
      restCalls++;
      throw new Error('REST no debe usarse para audio largo.');
    },
    evaluateContinuous: async (options) => {
      continuousOptions = options;
      return [
        {
          RecognitionStatus: 'Success',
          Offset: 0,
          Duration: 301000000,
          DisplayText: spokenWords.join(' '),
          NBest: [
            {
              Lexical: spokenWords.join(' '),
              Display: spokenWords.join(' '),
              PronunciationAssessment: {
                PronScore: 81,
                AccuracyScore: 82,
                FluencyScore: 78,
                CompletenessScore: 100,
                ProsodyScore: 76,
              },
              Words: spokenWords.map((word, index) => ({
                Word: word,
                Offset: index * 40000000,
                Duration: 30000000,
                PronunciationAssessment: {
                  AccuracyScore: 82,
                  ErrorType: 'None',
                },
              })),
            },
          ],
        },
      ];
    },
  });

  assert.equal(restCalls, 0);
  assert.equal(continuousOptions.referenceText, referenceText);
  assert.equal(continuousOptions.locale, 'en-US');
  assert.equal(result.recognitionMode, 'continuous');
  assert.equal(result.aggregation.reconstructedMiscues, true);
  assert.equal(result.aggregation.omissionCount, 1);
  assert.equal(result.aggregation.insertionCount, 2);
  assert.equal(result.completenessScore, 80);
  assert.equal(
    result.words.find((word) => word.errorType === 'Omission')?.word,
    'beta',
  );
  assert.deepEqual(
    result.words
      .filter((word) => word.errorType === 'Insertion')
      .map((word) => word.word),
    ['zeta', 'extra'],
  );
});

test('si Azure continuo falla segmenta una respuesta espontánea en tomas menores de 30 segundos', async () => {
  const restPaths = [];
  let cleaned = false;
  let splitOptions;
  const result = await evaluateAzureV2({
    normalizedPath: '/tmp/long-spontaneous.wav',
    durationSeconds: 65,
    rubric: {
      spec: {
        mode: 'spontaneous',
        targetLocale: 'en-US',
        referenceText: '',
      },
    },
    whisper: { words: [] },
    quality: {
      activity: {
        silenceIntervals: [{ start: 24.5, end: 25 }],
      },
    },
    azureConfig: {
      clavePrimaria: 'primary',
      region: 'test-region',
    },
    evaluateContinuous: async () => {
      const error = new Error('Azure canceló la sesión.');
      error.code = 'AZURE_CONTINUOUS_ERROR';
      throw error;
    },
    splitAudio: async (_path, options) => {
      splitOptions = options;
      return [
        { path: '/tmp/chunk-1.wav', start: 0, end: 25, owned: true },
        { path: '/tmp/chunk-2.wav', start: 25, end: 50, owned: true },
        { path: '/tmp/chunk-3.wav', start: 50, end: 65, owned: true },
      ];
    },
    cleanChunks: async () => {
      cleaned = true;
    },
    evaluateRest: async ({ rutaWav }) => {
      restPaths.push(rutaWav);
      const index = restPaths.length;
      return {
        provider: 'azure-speech',
        recognitionStatus: 'Success',
        displayText: `chunk ${index}`,
        lexicalText: `chunk ${index}`,
        pronunciationScore: 80,
        accuracyScore: 80,
        fluencyScore: 75,
        completenessScore: null,
        prosodyScore: 70,
        words: [
          {
            word: `word-${index}`,
            start: 1,
            duration: 0.5,
            accuracyScore: 80,
            errorType: 'None',
            phonemes: [],
            syllables: [],
          },
        ],
        rawResponse: { index },
      };
    },
  });

  assert.deepEqual(restPaths, [
    '/tmp/chunk-1.wav',
    '/tmp/chunk-2.wav',
    '/tmp/chunk-3.wav',
  ]);
  assert.equal(splitOptions.chunkSeconds, 25);
  assert.equal(splitOptions.overlapSeconds, 0);
  assert.equal(cleaned, true);
  assert.equal(result.recognitionMode, 'segmented-single-shot-fallback');
  assert.equal(result.fallback.reasonCode, 'AZURE_CONTINUOUS_ERROR');
  assert.equal(result.fallback.chunkCount, 3);
  assert.deepEqual(
    result.words.map((word) => word.start),
    [1, 26, 51],
  );
  assert.equal(result.accuracyScore, 80);
  assert.equal(result.fluencyScore, 75);
});

test('una muestra de 24.8 s con voz y palabras suficientes llega al extractor', async () => {
  const client = structuredClient([
    {
      sufficientEvidence: true,
      taskCoverage: 0.9,
      summary: 'La muestra contiene evidencia suficiente.',
      findings: [
        {
          id: 'communication-sequence',
          dimension: 'communication',
          type: 'coverage',
          claim: 'La respuesta inicia una narración situada en el pasado.',
          tokenStart: 0,
          tokenEnd: 3,
          quote: 'When I was ten',
          correction: '',
          certainty: 0.98,
        },
        {
          id: 'grammar-used-to',
          dimension: 'grammar',
          type: 'strength',
          claim: 'La respuesta utiliza used to.',
          tokenStart: 4,
          tokenEnd: 7,
          quote: 'I used to ride',
          correction: '',
          certainty: 0.98,
        },
        {
          id: 'vocabulary-bike',
          dimension: 'vocabulary',
          type: 'strength',
          claim: 'La respuesta emplea vocabulario cotidiano pertinente.',
          tokenStart: 8,
          tokenEnd: 11,
          quote: 'my bike to school',
          correction: '',
          certainty: 0.98,
        },
      ],
    },
  ]);
  const transcript =
    'When I was ten I used to ride my bike to school every day. One afternoon I fell because the road was wet. I could not move my arm at first, but a neighbor helped me. After visiting the doctor I was able to go home safely.';

  const evidence = await extractLinguisticEvidence({
    client,
    model: 'test-model',
    rubric: {
      spec: { mode: 'spontaneous' },
    },
    transcript,
    secondaryTranscript: '',
    quality: {
      metrics: {
        durationSeconds: 24.8,
        voicedSeconds: 24.8 * 0.67,
      },
    },
    providerDisagreement: null,
  });

  assert.equal(evidence.sufficientEvidence, true);
  assert.ok(evidence.sufficiency.lexicalCount >= 30);
  assert.ok(evidence.sufficiency.voicedSeconds >= 15);
  assert.equal(client.pendingResponses, 0);
  assert.equal(client.calls[0].max_completion_tokens, 3200);
});

test('respeta retry-after de Groq y recupera el extractor tras un 429', async () => {
  let attempts = 0;
  const response = {
    sufficientEvidence: true,
    taskCoverage: 0.8,
    summary: 'La muestra contiene evidencia suficiente.',
    findings: [
      {
        id: 'communication-retry',
        dimension: 'communication',
        type: 'coverage',
        claim: 'La respuesta desarrolla la tarea.',
        tokenStart: 0,
        tokenEnd: 2,
        quote: 'I described it',
        correction: '',
        certainty: 0.9,
      },
      {
        id: 'grammar-retry',
        dimension: 'grammar',
        type: 'strength',
        claim: 'La respuesta usa pasado simple.',
        tokenStart: 3,
        tokenEnd: 5,
        quote: 'and explained why',
        correction: '',
        certainty: 0.9,
      },
      {
        id: 'vocabulary-retry',
        dimension: 'vocabulary',
        type: 'strength',
        claim: 'La respuesta usa vocabulario pertinente.',
        tokenStart: 6,
        tokenEnd: 8,
        quote: 'the trip mattered',
        correction: '',
        certainty: 0.9,
      },
    ],
  };
  const client = {
    chat: {
      completions: {
        async create() {
          attempts++;
          if (attempts === 1) {
            const error = new Error('rate limited');
            error.status = 429;
            error.headers = new Map([['retry-after', '0']]);
            throw error;
          }
          return {
            choices: [{ message: { content: JSON.stringify(response) } }],
          };
        },
      },
    },
  };

  const evidence = await extractLinguisticEvidence({
    client,
    model: 'test-model',
    rubric: { spec: { mode: 'spontaneous' } },
    transcript:
      'I described it and explained why the trip mattered to my family.',
    secondaryTranscript: '',
    quality: { metrics: { durationSeconds: 20, voicedSeconds: 16 } },
    providerDisagreement: null,
  });

  assert.equal(attempts, 2);
  assert.equal(evidence.sufficientEvidence, true);
  assert.equal(evidence.findings.length, 3);
});

test('una muestra espontánea pasa con 15 s de voz aunque tenga menos de 30 palabras', async () => {
  const client = structuredClient([
    {
      sufficientEvidence: true,
      taskCoverage: 0.9,
      summary: 'La muestra contiene voz suficiente.',
      findings: [
        {
          id: 'communication-family',
          dimension: 'communication',
          type: 'coverage',
          claim: 'La respuesta relata una visita familiar.',
          tokenStart: 0,
          tokenEnd: 3,
          quote: 'Last weekend I visited',
          correction: '',
          certainty: 0.98,
        },
        {
          id: 'grammar-past',
          dimension: 'grammar',
          type: 'strength',
          claim: 'La respuesta utiliza pasado simple.',
          tokenStart: 4,
          tokenEnd: 7,
          quote: 'my grandmother and helped',
          correction: '',
          certainty: 0.97,
        },
        {
          id: 'vocabulary-family',
          dimension: 'vocabulary',
          type: 'strength',
          claim: 'La respuesta contiene vocabulario familiar pertinente.',
          tokenStart: 8,
          tokenEnd: 11,
          quote: 'her prepare dinner We',
          correction: '',
          certainty: 0.96,
        },
      ],
    },
  ]);
  const transcript =
    'Last weekend I visited my grandmother and helped her prepare dinner. We talked about school, watched a movie, and planned another family visit.';

  const evidence = await extractLinguisticEvidence({
    client,
    model: 'test-model',
    rubric: {
      spec: { mode: 'spontaneous' },
    },
    transcript,
    secondaryTranscript: '',
    quality: {
      metrics: {
        durationSeconds: 19,
        voicedSeconds: 16,
      },
    },
    providerDisagreement: null,
  });

  assert.equal(evidence.sufficientEvidence, true);
  assert.ok(evidence.sufficiency.lexicalCount < 30);
  assert.equal(evidence.sufficiency.meetsVoice, true);
  assert.equal(evidence.sufficiency.requirementsOperator, 'or');
  assert.equal(client.pendingResponses, 0);
});

test('una respuesta fuera del tema no apaga gramática ni vocabulario', async () => {
  const client = structuredClient([
    {
      sufficientEvidence: false,
      taskCoverage: 0.2,
      summary: 'La respuesta habla de otro país.',
      findings: [],
    },
    {
      sufficientEvidence: true,
      taskCoverage: 0.2,
      summary:
        'La cobertura comunicativa es baja, pero existe evidencia lingüística.',
      findings: [
        {
          id: 'communication-topic',
          dimension: 'communication',
          type: 'coverage',
          claim: 'La respuesta desarrolla una ruta por un país distinto.',
          tokenStart: 0,
          tokenEnd: 1,
          quote: 'Japan route',
          correction: '',
          certainty: 0.98,
        },
        {
          id: 'grammar-present-perfect',
          dimension: 'grammar',
          type: 'strength',
          claim: 'La respuesta contiene una estructura verbal identificable.',
          tokenStart: 2,
          tokenEnd: 4,
          quote: 'uses present perfect',
          correction: '',
          certainty: 0.95,
        },
        {
          id: 'vocabulary-travel',
          dimension: 'vocabulary',
          type: 'strength',
          claim: 'La respuesta contiene vocabulario temático de viajes.',
          tokenStart: 6,
          tokenEnd: 7,
          quote: 'travel vocabulary',
          correction: '',
          certainty: 0.95,
        },
      ],
    },
  ]);
  const transcript =
    'Japan route uses present perfect and travel vocabulary but discusses another country. ' +
    'The student also compares trains, buses, ferries, hotels, cities, roads, schedules, prices, comfort, safety, traffic, journeys, stops, plans, choices, routes, maps, and tickets.';

  const evidence = await extractLinguisticEvidence({
    client,
    model: 'test-model',
    rubric: {
      spec: {
        mode: 'spontaneous',
        instruction: 'Describe an itinerary through Italy.',
      },
    },
    transcript,
    secondaryTranscript: '',
    quality: {
      metrics: {
        durationSeconds: 35,
        voicedSeconds: 30,
      },
    },
    providerDisagreement: null,
  });

  assert.equal(evidence.sufficientEvidence, true);
  assert.equal(evidence.extractorRepaired, true);
  assert.equal(evidence.taskCoverage, 0.2);
  assert.deepEqual(
    [...new Set(evidence.findings.map((finding) => finding.dimension))].sort(),
    ['communication', 'grammar', 'vocabulary'],
  );
  assert.equal(client.pendingResponses, 0);
});

test('repara dimensiones omitidas aunque el extractor declare evidencia suficiente', async () => {
  const client = structuredClient([
    {
      sufficientEvidence: true,
      taskCoverage: 0.8,
      summary: 'La respuesta contiene evidencia parcial.',
      findings: [
        {
          id: 'communication-trip',
          dimension: 'communication',
          type: 'coverage',
          claim: 'La respuesta desarrolla una experiencia de viaje.',
          tokenStart: 0,
          tokenEnd: 3,
          quote: 'Last year I traveled',
          correction: '',
          certainty: 0.96,
        },
      ],
    },
    {
      sufficientEvidence: true,
      taskCoverage: 0.8,
      summary: 'Hay evidencia verificable para las tres dimensiones.',
      findings: [
        {
          id: 'grammar-past',
          dimension: 'grammar',
          type: 'strength',
          claim: 'La respuesta emplea pasado simple.',
          tokenStart: 0,
          tokenEnd: 3,
          quote: 'Last year I traveled',
          correction: '',
          certainty: 0.97,
        },
        {
          id: 'vocabulary-travel',
          dimension: 'vocabulary',
          type: 'strength',
          claim: 'La respuesta emplea vocabulario pertinente de viajes.',
          tokenStart: 4,
          tokenEnd: 7,
          quote: 'by train through several',
          correction: '',
          certainty: 0.95,
        },
      ],
    },
  ]);
  const transcript =
    'Last year I traveled by train through several cities and visited museums, markets, stations, hotels, restaurants, parks, bridges, neighborhoods, landmarks, and cultural centers. I compared prices, planned routes, bought tickets, asked for directions, described the weather, and explained which places were comfortable, convenient, safe, crowded, memorable, and interesting.';

  const evidence = await extractLinguisticEvidence({
    client,
    model: 'test-model',
    rubric: {
      spec: {
        mode: 'spontaneous',
        instruction: 'Describe a trip.',
      },
    },
    transcript,
    secondaryTranscript: '',
    quality: {
      metrics: {
        durationSeconds: 35,
        voicedSeconds: 30,
      },
    },
    providerDisagreement: null,
  });

  assert.equal(evidence.sufficientEvidence, true);
  assert.equal(evidence.extractorRepairAttempted, true);
  assert.equal(evidence.extractorRepaired, true);
  assert.deepEqual(evidence.missingDimensions, []);
  assert.deepEqual(
    [...new Set(evidence.findings.map((finding) => finding.dimension))].sort(),
    ['communication', 'grammar', 'vocabulary'],
  );
  assert.equal(client.pendingResponses, 0);
});

test('repara citas omitidas o inválidas usando evidencia de la misma dimensión', async () => {
  const client = structuredClient([
    judgeResponse(),
    judgeResponse({ invalidCitations: true }),
  ]);
  const evidence = {
    sufficientEvidence: true,
    taskCoverage: 1,
    summary: 'La respuesta contiene evidencia verificable.',
    findings: linguisticFindings(),
  };

  const result = await runDoubleLinguisticJudging({
    client,
    model: 'test-model',
    rubric: { spec: { mode: 'spontaneous', cefr: 'B1' } },
    evidence,
  });

  for (const [id, expectedEvidenceId] of [
    ['communication', 'communication-1'],
    ['grammar', 'grammar-1'],
    ['vocabulary', 'vocabulary-1'],
  ]) {
    assert.equal(result.dimensions[id].status, 'scored');
    assert.deepEqual(result.dimensions[id].evidenceIds, [expectedEvidenceId]);
    assert.equal(result.dimensions[id].reviewRequired, true);
  }
  assert.equal(client.pendingResponses, 0);
});

test('no inventa una puntuación cuando la dimensión carece de evidencia', async () => {
  const response = judgeResponse();
  const client = structuredClient([response, response]);
  const evidence = {
    sufficientEvidence: true,
    taskCoverage: 0.8,
    summary: 'Solo hay evidencia para dos dimensiones.',
    findings: linguisticFindings().filter(
      (finding) => finding.dimension !== 'vocabulary',
    ),
  };

  const result = await runDoubleLinguisticJudging({
    client,
    model: 'test-model',
    rubric: { spec: { mode: 'spontaneous', cefr: 'B1' } },
    evidence,
  });

  assert.equal(result.dimensions.communication.status, 'scored');
  assert.equal(result.dimensions.grammar.status, 'scored');
  assert.equal(
    result.dimensions.vocabulary.status,
    'insufficientEvidence',
  );
  assert.equal(
    result.dimensions.vocabulary.reasonCode,
    'JUDGE_EVIDENCE_NOT_CITED',
  );
});
