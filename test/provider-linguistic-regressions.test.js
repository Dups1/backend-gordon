import assert from 'node:assert/strict';
import test from 'node:test';

import {
  extractLinguisticEvidence,
  generateExpectedPhoneticFromWhisper,
  improveStudentInstructionWithAI,
  judgePronunciationFromPhonetics,
  resetLinguisticGovernorForTests,
  runDoubleLinguisticJudging,
  transcribePhonemesLiterally,
} from '../src/evaluation/linguistic.js';

function structuredClient(responses) {
  const pending = [...responses];
  const calls = [];
  return {
    chat: {
      completions: {
        async create(options, requestOptions) {
          calls.push({ ...options, requestOptions });
          const content = pending.shift();
          assert.ok(content, 'La prueba no configuró una respuesta del modelo.');
          if (content instanceof Error) throw content;
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

test('juzga IPA con inteligibilidad y contexto de lengua materna', async () => {
  const client = structuredClient([
    {
      band: 3,
      rationale:
        'La realización conserva suficiente inteligibilidad aunque refleja transferencia del español.',
      observations: [
        {
          alignmentId: 'w0',
          expected: 'dʒəˈpæn',
          explanation:
            'Hay sustitución consonántica, pero la palabra permanece reconocible en contexto.',
          affectsIntelligibility: false,
        },
      ],
    },
  ]);

  const result = await judgePronunciationFromPhonetics({
    client,
    model: 'test-model',
    rubric: {
      spec: {
        mode: 'spontaneous',
        targetLocale: 'en-US',
        cefr: 'B1',
        nativeLanguage: 'es',
      },
    },
    transcript: 'Japan route.',
    words: [
      { word: 'Japan', start: 0, end: 0.55 },
      { word: 'route', start: 0.6, end: 1.1 },
    ],
    phoneticEvidence: {
      transcript: 'ðæpən ɹoʊt',
      model: 'wav2vec2-phoneme-en',
      confidence: 0.702,
      events: [
        { phoneme: 'ð', startSec: 0.02, endSec: 0.1, confidence: 0.7 },
        { phoneme: 'æ', startSec: 0.12, endSec: 0.2, confidence: 0.72 },
        { phoneme: 'p', startSec: 0.22, endSec: 0.3, confidence: 0.71 },
        { phoneme: 'ə', startSec: 0.32, endSec: 0.4, confidence: 0.69 },
        { phoneme: 'n', startSec: 0.42, endSec: 0.5, confidence: 0.7 },
        { phoneme: 'ɹ', startSec: 0.62, endSec: 0.7, confidence: 0.73 },
        { phoneme: 'oʊ', startSec: 0.72, endSec: 0.88, confidence: 0.72 },
        { phoneme: 't', startSec: 0.9, endSec: 1.02, confidence: 0.71 },
      ],
    },
  });

  assert.equal(result.status, 'scored');
  assert.equal(result.band, 3);
  assert.equal(result.observations[0].observed, 'ðæpən');
  const systemPrompt = client.calls[0].messages[0].content;
  assert.match(systemPrompt, /no exijas acento nativo/i);
  const payload = JSON.parse(client.calls[0].messages[1].content).data;
  assert.equal(payload.nativeLanguage, 'español');
  assert.equal(payload.observedIpa, undefined);
  assert.equal(payload.orthographicTranscript, 'Japan route.');
  assert.deepEqual(
    payload.wordAlignments.map((item) => [item.word, item.observedIpa]),
    [
      ['Japan', 'ðæpən'],
      ['route', 'ɹoʊt'],
    ],
  );
  assert.equal(client.pendingResponses, 0);
});

test('convierte fonemas en texto literal sin recibir Whisper ni una frase esperada', async () => {
  const client = structuredClient([
    {
      segments: [
        {
          id: 'p0',
          original_ipa: 'ðæpən',
          tokens: [{ ipa: 'ðæpən', written: 'dhapen' }],
        },
        {
          id: 'p1',
          original_ipa: 'ɹoʊ',
          tokens: [{ ipa: 'ɹoʊ', written: 'rou' }],
        },
      ],
    },
  ]);

  const result = await transcribePhonemesLiterally({
    client,
    model: 'deepseek-v4-flash',
    targetLocale: 'en-US',
    phoneticEvidence: {
      transcript: 'ðæpən  ·  ɹoʊ',
      model: 'wav2vec2-phoneme-en',
      confidence: 0.7,
      events: [],
    },
  });

  assert.equal(result.text, 'dhapen rou');
  assert.equal(result.usesWhisperReference, false);
  assert.equal(result.exactIpaCoverage, true);
  assert.equal(
    result.methodId,
    'deepseek-phoneme-literal-transcription-v3',
  );
  const request = JSON.parse(client.calls[0].messages[1].content).data;
  assert.deepEqual(request, {
    targetLocale: 'en-US',
    acousticModel: 'wav2vec2-phoneme-en',
    segments: [
      { id: 'p0', original_ipa: 'ðæpən' },
      { id: 'p1', original_ipa: 'ɹoʊ' },
    ],
  });
  assert.equal(JSON.stringify(request).includes('Whisper'), false);
  assert.equal(JSON.stringify(request).includes('instruction'), false);
  assert.match(
    client.calls[0].messages[0].content,
    /no conviertas una secuencia en una palabra correcta/i,
  );
});

test('genera IPA esperada únicamente desde la transcripción de Whisper', async () => {
  const client = structuredClient([
    {
      fullIpa: 'dʒəˈpæn ɹuːt',
      words: [
        { id: 'w0', text: 'Japan', ipa: 'dʒəˈpæn' },
        { id: 'w1', text: 'route', ipa: 'ɹuːt' },
      ],
    },
  ]);

  const result = await generateExpectedPhoneticFromWhisper({
    client,
    model: 'deepseek-v4-flash',
    targetLocale: 'en-US',
    transcript: 'Japan route.',
    words: [
      { word: 'Japan', start: 0, end: 0.55 },
      { word: 'route', start: 0.6, end: 1.1 },
    ],
  });

  assert.equal(result.transcript, 'dʒəˈpæn ɹuːt');
  assert.equal(result.source, 'whisper-primary');
  assert.deepEqual(
    result.words.map((word) => [word.id, word.text, word.ipa]),
    [
      ['w0', 'Japan', 'dʒəˈpæn'],
      ['w1', 'route', 'ɹuːt'],
    ],
  );
  const payload = JSON.parse(client.calls[0].messages[1].content).data;
  assert.equal(payload.transcript, 'Japan route.');
  assert.deepEqual(payload.words, [
    { id: 'w0', text: 'Japan' },
    { id: 'w1', text: 'route' },
  ]);
  assert.match(client.calls[0].messages[0].content, /pronunciación IPA esperada/i);
  assert.equal(JSON.stringify(payload).includes('observed'), false);
  assert.equal(client.pendingResponses, 0);
});

test('conserva una conversión legible si DeepSeek no copia el IPA con exactitud', async () => {
  const response = {
    segments: [
      {
        id: 'p0',
        original_ipa: 'fəloʊmaɪneɪmɪz',
        tokens: [
          { ipa: 'həloʊ', written: 'fello' },
          { ipa: 'maɪ', written: 'my' },
          { ipa: 'neɪm', written: 'name' },
          { ipa: 'ɪz', written: 'is' },
        ],
      },
    ],
  };
  const client = structuredClient([response]);

  const result = await transcribePhonemesLiterally({
    client,
    model: 'deepseek-v4-flash',
    targetLocale: 'en-US',
    phoneticEvidence: {
      transcript: 'fəloʊmaɪneɪmɪz',
      model: 'wav2vec2-phoneme-en',
      confidence: 0.83,
      events: [],
    },
  });

  assert.equal(result.text, 'fello my name is');
  assert.equal(result.exactIpaCoverage, false);
  assert.equal(client.calls.length, 1);
});

test('infiere espacios sin corregir un fonema pronunciado incorrectamente', async () => {
  const client = structuredClient([
    {
      segments: [
        {
          id: 'p0',
          original_ipa: 'fəloʊmaɪneɪmɪz',
          tokens: [
            { ipa: 'fəloʊ', written: 'fello' },
            { ipa: 'maɪ', written: 'my' },
            { ipa: 'neɪm', written: 'name' },
            { ipa: 'ɪz', written: 'is' },
          ],
        },
      ],
    },
  ]);

  const result = await transcribePhonemesLiterally({
    client,
    model: 'deepseek-v4-flash',
    targetLocale: 'en-US',
    phoneticEvidence: {
      transcript: 'fəloʊmaɪneɪmɪz',
      model: 'wav2vec2-phoneme-en',
      confidence: 0.83,
      events: [],
    },
  });

  assert.equal(result.text, 'fello my name is');
  assert.equal(result.text.includes('hello'), false);
  assert.deepEqual(
    result.segments[0].tokens.map((token) => token.ipa).join(''),
    'fəloʊmaɪneɪmɪz',
  );
});

test('reintenta cuando DeepSeek concatena una frase como una sola palabra', async () => {
  const ipa = 'həloʊmaɪneɪmɪzwɛn';
  const client = structuredClient([
    {
      segments: [
        {
          id: 'p0',
          original_ipa: ipa,
          tokens: [{ ipa, written: 'hellomynameis' }],
        },
      ],
    },
    {
      segments: [
        {
          id: 'p0',
          original_ipa: ipa,
          tokens: [
            { ipa: 'həloʊ', written: 'hello' },
            { ipa: 'maɪ', written: 'my' },
            { ipa: 'neɪm', written: 'name' },
            { ipa: 'ɪz', written: 'is' },
            { ipa: 'wɛn', written: 'when' },
          ],
        },
      ],
    },
  ]);

  const result = await transcribePhonemesLiterally({
    client,
    model: 'deepseek-v4-flash',
    targetLocale: 'en-US',
    phoneticEvidence: {
      transcript: ipa,
      model: 'wav2vec2-phoneme-en',
      confidence: 0.83,
      events: [],
    },
  });

  assert.equal(result.text, 'hello my name is when');
  assert.equal(client.calls.length, 2);
  assert.match(
    client.calls[1].messages[0].content,
    /concatenó una frase completa/i,
  );
});

test('usa JSON Object Mode y valida localmente el juez fonético', async () => {
  const client = structuredClient([
    {
      band: 3,
      rationale: 'La palabra sigue siendo inteligible.',
      observations: [
        {
          alignmentId: 'w0',
          expected: 'həˈloʊ',
          explanation: 'La realización conserva la palabra.',
          affectsIntelligibility: false,
        },
      ],
    },
  ]);

  const result = await judgePronunciationFromPhonetics({
    client,
    model: 'test-model',
    rubric: {
      spec: {
        mode: 'spontaneous',
        targetLocale: 'en-US',
        cefr: 'B1',
        nativeLanguage: 'es',
      },
    },
    transcript: 'Hello.',
    words: [{ word: 'Hello', start: 0, end: 0.5 }],
    phoneticEvidence: {
      transcript: 'həloʊ',
      model: 'wav2vec2-phoneme-en',
      confidence: 0.8,
      events: [
        { phoneme: 'h', startSec: 0, endSec: 0.1, confidence: 0.8 },
        { phoneme: 'ə', startSec: 0.12, endSec: 0.2, confidence: 0.8 },
        { phoneme: 'l', startSec: 0.22, endSec: 0.3, confidence: 0.8 },
        { phoneme: 'oʊ', startSec: 0.32, endSec: 0.48, confidence: 0.8 },
      ],
    },
  });

  assert.equal(result.status, 'scored');
  assert.equal(result.band, 3);
  assert.equal(result.observations[0].observed, 'həloʊ');
  assert.equal(client.calls.length, 1);
  assert.equal(client.calls[0].response_format.type, 'json_object');
  assert.deepEqual(client.calls[0].thinking, { type: 'disabled' });
  assert.equal(client.calls[0].max_tokens, 8000);
  const outputPayload = JSON.parse(client.calls[0].messages[1].content);
  assert.equal(outputPayload.data.observedIpa, undefined);
  assert.equal(outputPayload.outputSchema.required.includes('band'), true);
  assert.equal(client.pendingResponses, 0);
});

test('recupera una mejora de consigna cuando OpenCode entrega contenido vacío', async () => {
  const calls = [];
  const improvement = {
    improvedInstruction:
      'Describe una experiencia pasada y explica qué pudiste hacer.',
    detectedAudience: 'student',
    summary: 'La consigna quedó dirigida al estudiante.',
    preservedRequirements: ['Usar pasado simple', 'Usar could'],
    warnings: [],
  };
  const client = {
    chat: {
      completions: {
        async create(options) {
          calls.push(options);
          if (calls.length === 1) {
            return {
              choices: [
                {
                  finish_reason: 'length',
                  message: { content: null },
                },
              ],
              usage: { completion_tokens: options.max_tokens },
            };
          }
          return {
            choices: [
              {
                finish_reason: 'stop',
                message: {
                  content: [
                    {
                      type: 'text',
                      text: `\`\`\`json\n${JSON.stringify(improvement)}\n\`\`\``,
                    },
                  ],
                },
              },
            ],
          };
        },
      },
    },
  };

  const result = await improveStudentInstructionWithAI({
    client,
    model: 'deepseek-v4-flash',
    spec: {
      mode: 'spontaneous',
      targetLocale: 'en-US',
      cefr: 'B1',
      instruction: 'Revisa que use pasado y could.',
      communicativePurpose: 'Narrar una experiencia',
    },
  });

  assert.equal(result.improvedInstruction, improvement.improvedInstruction);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].response_format.type, 'json_object');
  assert.deepEqual(calls[0].thinking, { type: 'disabled' });
  assert.equal(calls[1].response_format, undefined);
  assert.deepEqual(calls[1].thinking, { type: 'disabled' });
  assert.ok(calls[1].max_tokens > calls[0].max_tokens);
});

test('recupera evidencia y pronunciación si DeepSeek agota el primer intento razonando', async () => {
  const emptyLengthResponse = (maxTokens) => ({
    choices: [
      {
        finish_reason: 'length',
        message: { content: '' },
      },
    ],
    usage: { completion_tokens: maxTokens },
  });
  const evidenceResponse = {
    sufficientEvidence: true,
    taskCoverage: 0.8,
    summary: 'La respuesta contiene evidencia evaluable.',
    findings: [
      {
        id: 'communication-recovery',
        dimension: 'communication',
        type: 'coverage',
        claim: 'La respuesta desarrolla la experiencia solicitada.',
        tokenStart: 0,
        tokenEnd: 3,
        quote: 'I described my experience',
        correction: '',
        certainty: 0.9,
      },
      {
        id: 'grammar-recovery',
        dimension: 'grammar',
        type: 'strength',
        claim: 'La respuesta utiliza pasado simple.',
        tokenStart: 4,
        tokenEnd: 7,
        quote: 'and explained what happened',
        correction: '',
        certainty: 0.9,
      },
      {
        id: 'vocabulary-recovery',
        dimension: 'vocabulary',
        type: 'strength',
        claim: 'La respuesta usa vocabulario pertinente.',
        tokenStart: 8,
        tokenEnd: 11,
        quote: 'during the school trip',
        correction: '',
        certainty: 0.9,
      },
    ],
  };
  const evidenceCalls = [];
  const evidenceClient = {
    chat: {
      completions: {
        async create(options) {
          evidenceCalls.push(options);
          if (evidenceCalls.length === 1) {
            return emptyLengthResponse(options.max_tokens);
          }
          return {
            choices: [
              {
                finish_reason: 'stop',
                message: { content: JSON.stringify(evidenceResponse) },
              },
            ],
          };
        },
      },
    },
  };

  const evidence = await extractLinguisticEvidence({
    client: evidenceClient,
    model: 'deepseek-v4-flash',
    rubric: { spec: { mode: 'spontaneous' } },
    transcript:
      'I described my experience and explained what happened during the school trip with my classmates and teacher.',
    secondaryTranscript: '',
    quality: { metrics: { durationSeconds: 20, voicedSeconds: 16 } },
    providerDisagreement: null,
  });

  assert.equal(evidence.sufficientEvidence, true);
  assert.equal(evidenceCalls.length, 2);
  assert.equal(evidenceCalls[0].max_tokens, 8000);
  assert.equal(evidenceCalls[1].max_tokens, 16000);
  assert.deepEqual(evidenceCalls[0].thinking, { type: 'disabled' });

  const pronunciationCalls = [];
  const pronunciationClient = {
    chat: {
      completions: {
        async create(options) {
          pronunciationCalls.push(options);
          if (pronunciationCalls.length === 1) {
            return emptyLengthResponse(options.max_tokens);
          }
          return {
            choices: [
              {
                finish_reason: 'stop',
                message: {
                  content: JSON.stringify({
                    band: 3,
                    rationale: 'La palabra permanece inteligible.',
                    observations: [
                      {
                        alignmentId: 'w0',
                        expected: 'həˈloʊ',
                        explanation:
                          'La realización conserva los contrastes principales.',
                        affectsIntelligibility: false,
                      },
                    ],
                  }),
                },
              },
            ],
          };
        },
      },
    },
  };

  const pronunciation = await judgePronunciationFromPhonetics({
    client: pronunciationClient,
    model: 'deepseek-v4-flash',
    rubric: {
      spec: {
        mode: 'spontaneous',
        targetLocale: 'en-US',
        cefr: 'B1',
        nativeLanguage: 'es',
      },
    },
    transcript: 'Hello.',
    words: [{ word: 'Hello', start: 0, end: 0.5 }],
    phoneticEvidence: {
      transcript: 'həloʊ',
      model: 'wav2vec2-phoneme-en',
      confidence: 0.8,
      events: [
        { phoneme: 'h', startSec: 0, endSec: 0.1, confidence: 0.8 },
        { phoneme: 'ə', startSec: 0.12, endSec: 0.2, confidence: 0.8 },
        { phoneme: 'l', startSec: 0.22, endSec: 0.3, confidence: 0.8 },
        { phoneme: 'oʊ', startSec: 0.32, endSec: 0.48, confidence: 0.8 },
      ],
    },
  });

  assert.equal(pronunciation.status, 'scored');
  assert.equal(pronunciationCalls.length, 2);
  assert.equal(pronunciationCalls[0].max_tokens, 8000);
  assert.equal(pronunciationCalls[1].max_tokens, 16000);
  assert.deepEqual(pronunciationCalls[0].thinking, {
    type: 'disabled',
  });
});

test('una muestra breve sigue siendo evidencia válida para las tres dimensiones', async () => {
  const client = structuredClient([
    {
      sufficientEvidence: true,
      taskCoverage: 0.2,
      summary: 'La muestra es breve, pero contiene lenguaje evaluable.',
      findings: ['communication', 'grammar', 'vocabulary'].map(
        (dimension) => ({
          id: `${dimension}-brief`,
          dimension,
          type: 'uncertainty',
          claim: 'La única palabra disponible aporta evidencia limitada.',
          tokenStart: 0,
          tokenEnd: 0,
          quote: 'Hello',
          correction: '',
          certainty: 0.35,
        }),
      ),
    },
  ]);

  const evidence = await extractLinguisticEvidence({
    client,
    model: 'test-model',
    rubric: { spec: { mode: 'spontaneous' } },
    transcript: 'Hello',
    secondaryTranscript: '',
    quality: {
      metrics: {
        durationSeconds: 1.2,
        voicedSeconds: 0.7,
      },
    },
    providerDisagreement: null,
  });

  assert.equal(evidence.sufficientEvidence, true);
  assert.equal(evidence.sufficiency.recommendedSample, false);
  assert.equal(evidence.findings.length, 3);
  assert.equal(client.pendingResponses, 0);
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
  assert.equal(client.calls[0].max_tokens, 8000);
  assert.equal(client.calls[0].requestOptions.maxRetries, 0);
});

test('respeta retry-after de OpenCode y recupera el extractor tras un 429', async () => {
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

test('compacta una muestra larga antes de enviarla al extractor', async () => {
  const client = structuredClient([
    {
      sufficientEvidence: true,
      taskCoverage: 0.85,
      summary: 'La muestra contiene evidencia suficiente.',
      findings: [
        {
          id: 'communication-long',
          dimension: 'communication',
          type: 'coverage',
          claim: 'La respuesta presenta una experiencia pasada.',
          tokenStart: 0,
          tokenEnd: 3,
          quote: 'Last year I planned',
          correction: '',
          certainty: 0.95,
        },
        {
          id: 'grammar-long',
          dimension: 'grammar',
          type: 'strength',
          claim: 'La respuesta usa pasado simple.',
          tokenStart: 4,
          tokenEnd: 7,
          quote: 'a long trip through',
          correction: '',
          certainty: 0.9,
        },
        {
          id: 'vocabulary-long',
          dimension: 'vocabulary',
          type: 'strength',
          claim: 'La respuesta usa vocabulario de viajes.',
          tokenStart: 8,
          tokenEnd: 11,
          quote: 'Japan with two friends',
          correction: '',
          certainty: 0.9,
        },
      ],
    },
  ]);
  const paragraph =
    'Last year I planned a long trip through Japan with two friends from university. ' +
    'First we traveled from Tokyo to Kyoto by train because it was faster and more comfortable than taking a bus. ' +
    'I reserved a hotel near the station but booked the wrong dates so a local teacher helped us find a guest house. ';
  const transcript = `${paragraph}${paragraph}${paragraph}`;
  const secondaryTranscript = transcript.replaceAll('train', 'plane');

  await extractLinguisticEvidence({
    client,
    model: 'test-model',
    rubric: {
      spec: {
        mode: 'spontaneous',
        targetLocale: 'en-US',
        cefr: 'B2',
        instruction:
          'Describe a past trip, compare transportation, and explain each decision.',
        communicativePurpose: 'Narrate and explain',
        targetConcepts: ['sequence', 'comparison', 'justification'],
        vocabularyHints: ['train', 'hotel', 'route'],
      },
      dimensions: [
        {
          id: 'communication',
          descriptor: 'Transmite una narración coherente.',
          constructScope: 'productive_speaking',
          bands: Array.from({ length: 5 }, (_value, band) => ({
            band,
            label: `Band ${band}`,
          })),
        },
        {
          id: 'grammar',
          descriptor: 'Usa estructuras del nivel.',
          constructScope: 'productive_speaking',
          bands: [],
        },
        {
          id: 'vocabulary',
          descriptor: 'Usa léxico adecuado.',
          constructScope: 'productive_speaking',
          bands: [],
        },
      ],
      scoreProfile: { weights: { communication: 1 } },
      limitations: ['No debe enviarse al modelo en esta etapa.'],
    },
    transcript,
    secondaryTranscript,
    quality: {
      status: 'accepted',
      warnings: [],
      metrics: {
        durationSeconds: 103,
        voicedSeconds: 60,
        speechRatio: 0.58,
        snrDb: 25,
        clippingRatio: 0,
      },
      signature: { originalSha256: 'not-needed-by-the-model' },
    },
    providerDisagreement: 0.08,
  });

  const request = JSON.parse(client.calls[0].messages[1].content);
  assert.ok(request.data.tokens.every((token) => typeof token === 'string'));
  assert.equal('secondaryTranscript' in request.data, false);
  assert.equal('scoreProfile' in request.data.rubric, false);
  assert.equal('signature' in request.data.audioQuality, false);
  assert.equal(request.data.rubric.dimensions[0].bands.length, 5);
  assert.equal(
    request.data.rubric.dimensions[0].bands[3].label,
    'Band 3',
  );
  assert.ok(
    Array.isArray(request.data.asrComparison.uncertainPrimaryTokenIndices),
  );
  assert.ok(client.calls[0].messages[1].content.length < 16_000);
  assert.equal(client.calls[0].max_tokens, 8000);
});

test('conserva la extracción válida si falla una reparación complementaria', async () => {
  let attempts = 0;
  const client = {
    chat: {
      completions: {
        async create() {
          attempts++;
          if (attempts === 1) {
            return {
              choices: [
                {
                  message: {
                    content: JSON.stringify({
                      sufficientEvidence: true,
                      taskCoverage: 0.7,
                      summary: 'Hay evidencia comunicativa válida.',
                      findings: [
                        {
                          id: 'communication-valid',
                          dimension: 'communication',
                          type: 'coverage',
                          claim: 'La respuesta explica una decisión.',
                          tokenStart: 0,
                          tokenEnd: 3,
                          quote: 'I explained my decision',
                          correction: '',
                          certainty: 0.9,
                        },
                      ],
                    }),
                  },
                },
              ],
            };
          }
          const error = new Error('Request too large.');
          error.status = 413;
          throw error;
        },
      },
    },
  };

  const evidence = await extractLinguisticEvidence({
    client,
    model: 'test-model',
    rubric: { spec: { mode: 'spontaneous' } },
    transcript:
      'I explained my decision and described the complete journey with several details about routes tickets stations hotels schedules prices safety comfort and transportation.',
    secondaryTranscript: '',
    quality: { metrics: { durationSeconds: 20, voicedSeconds: 16 } },
    providerDisagreement: null,
  });

  assert.equal(attempts, 2);
  assert.equal(evidence.findings.length, 3);
  assert.deepEqual(
    [...new Set(evidence.findings.map((finding) => finding.dimension))].sort(),
    ['communication', 'grammar', 'vocabulary'],
  );
  assert.equal(evidence.extractorRepairAttempted, true);
  assert.equal(evidence.extractorRepaired, false);
  assert.equal(
    evidence.extractorRepairError?.code,
    'LINGUISTIC_REQUEST_TOO_LARGE',
  );
});

test('serializa las llamadas lingüísticas concurrentes por instancia', async () => {
  const previousLimit = process.env.OPENCODE_LINGUISTIC_TPM_LIMIT;
  process.env.OPENCODE_LINGUISTIC_TPM_LIMIT = '50000';
  resetLinguisticGovernorForTests();
  let inFlight = 0;
  let maximumInFlight = 0;
  const response = {
    sufficientEvidence: true,
    taskCoverage: 0.8,
    summary: 'La muestra contiene evidencia suficiente.',
    findings: [
      {
        id: 'communication-queue',
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
        id: 'grammar-queue',
        dimension: 'grammar',
        type: 'strength',
        claim: 'La respuesta usa una estructura verbal.',
        tokenStart: 3,
        tokenEnd: 5,
        quote: 'and explained why',
        correction: '',
        certainty: 0.9,
      },
      {
        id: 'vocabulary-queue',
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
          inFlight++;
          maximumInFlight = Math.max(maximumInFlight, inFlight);
          await new Promise((resolve) => setTimeout(resolve, 10));
          inFlight--;
          return {
            choices: [
              { message: { content: JSON.stringify(response) } },
            ],
          };
        },
      },
    },
  };
  const input = {
    client,
    model: 'deepseek-v4-flash',
    rubric: { spec: { mode: 'spontaneous' } },
    transcript:
      'I described it and explained why the trip mattered to my family.',
    secondaryTranscript: '',
    quality: { metrics: { durationSeconds: 20, voicedSeconds: 16 } },
    providerDisagreement: null,
  };

  try {
    await Promise.all([
      extractLinguisticEvidence(input),
      extractLinguisticEvidence(input),
    ]);
    assert.equal(maximumInFlight, 1);
  } finally {
    resetLinguisticGovernorForTests();
    if (previousLimit === undefined) {
      delete process.env.OPENCODE_LINGUISTIC_TPM_LIMIT;
    } else {
      process.env.OPENCODE_LINGUISTIC_TPM_LIMIT = previousLimit;
    }
  }
});

test('rechaza localmente una etapa que no cabe en el presupuesto TPM', async () => {
  const previousLimit = process.env.OPENCODE_LINGUISTIC_TPM_LIMIT;
  process.env.OPENCODE_LINGUISTIC_TPM_LIMIT = '500';
  resetLinguisticGovernorForTests();
  let attempts = 0;
  const client = {
    chat: {
      completions: {
        async create() {
          attempts++;
          throw new Error('No debe contactar al proveedor.');
        },
      },
    },
  };

  try {
    await assert.rejects(
      extractLinguisticEvidence({
        client,
        model: 'deepseek-v4-flash',
        rubric: { spec: { mode: 'spontaneous' } },
        transcript:
          'I described a complete journey with routes tickets stations hotels schedules prices safety comfort transportation plans and several detailed reasons for each decision.',
        secondaryTranscript: '',
        quality: { metrics: { durationSeconds: 20, voicedSeconds: 16 } },
        providerDisagreement: null,
      }),
      (error) => {
        assert.equal(
          error.code,
          'LINGUISTIC_LOCAL_TOKEN_BUDGET_EXCEEDED',
        );
        assert.ok(error.details.estimatedRequestTokens > 500);
        return true;
      },
    );
    assert.equal(attempts, 0);
  } finally {
    resetLinguisticGovernorForTests();
    if (previousLimit === undefined) {
      delete process.env.OPENCODE_LINGUISTIC_TPM_LIMIT;
    } else {
      process.env.OPENCODE_LINGUISTIC_TPM_LIMIT = previousLimit;
    }
  }
});

test('conserva dimensiones acordadas si falla la adjudicación', async () => {
  let attempts = 0;
  const analytic = judgeResponse();
  const holistic = judgeResponse();
  holistic.dimensions[0].band = 2;
  const client = {
    chat: {
      completions: {
        async create() {
          attempts++;
          if (attempts <= 2) {
            const content = attempts === 1 ? analytic : holistic;
            return {
              choices: [
                { message: { content: JSON.stringify(content) } },
              ],
            };
          }
          const error = new Error('Request too large.');
          error.status = 413;
          throw error;
        },
      },
    },
  };

  const result = await runDoubleLinguisticJudging({
    client,
    model: 'test-model',
    rubric: {
      spec: { mode: 'spontaneous' },
      dimensions: [],
    },
    evidence: {
      sufficientEvidence: true,
      summary: 'Hay evidencia.',
      findings: linguisticFindings(),
    },
  });

  assert.equal(result.dimensions.communication.status, 'scored');
  assert.equal(result.dimensions.communication.reviewRequired, true);
  assert.equal(result.dimensions.grammar.status, 'scored');
  assert.equal(result.dimensions.vocabulary.status, 'scored');
  assert.equal(result.adjudicated, false);
  assert.equal(result.adjudicationAttempted, true);
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
