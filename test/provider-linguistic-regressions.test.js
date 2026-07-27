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
  return {
    chat: {
      completions: {
        async create() {
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

test('una muestra de 24.8 s con voz y palabras suficientes llega al extractor', async () => {
  const client = structuredClient([
    {
      sufficientEvidence: true,
      taskCoverage: 0.9,
      summary: 'La muestra contiene evidencia suficiente.',
      findings: [],
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
