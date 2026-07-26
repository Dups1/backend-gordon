import assert from 'node:assert/strict';
import test from 'node:test';

import {
  extractLinguisticEvidence,
  runDoubleLinguisticJudging,
} from '../src/evaluation/linguistic.js';
import {
  createContinuousPronunciationConfig,
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
