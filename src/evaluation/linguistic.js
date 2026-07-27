import {
  DIMENSION_IDS,
  EvaluationError,
  PROMPT_VERSION,
  expectedScore,
  normalizeTokens,
  probabilitiesForBand,
} from './domain.js';
import {
  INTERNAL_PROMPTS,
  PROMPT_MANIFEST_HASH,
} from './prompts.js';
import { withCircuitBreaker } from './resilience.js';

const LINGUISTIC_DIMENSIONS = ['communication', 'grammar', 'vocabulary'];

const evidenceSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    sufficientEvidence: { type: 'boolean' },
    taskCoverage: { type: 'number', minimum: 0, maximum: 1 },
    summary: { type: 'string' },
    findings: {
      type: 'array',
      maxItems: 30,
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          id: { type: 'string' },
          dimension: {
            type: 'string',
            enum: LINGUISTIC_DIMENSIONS,
          },
          type: {
            type: 'string',
            enum: ['strength', 'error', 'coverage', 'uncertainty'],
          },
          claim: { type: 'string' },
          tokenStart: { type: 'integer', minimum: 0 },
          tokenEnd: { type: 'integer', minimum: 0 },
          quote: { type: 'string' },
          correction: { type: 'string' },
          certainty: { type: 'number', minimum: 0, maximum: 1 },
        },
        required: [
          'id',
          'dimension',
          'type',
          'claim',
          'tokenStart',
          'tokenEnd',
          'quote',
          'correction',
          'certainty',
        ],
      },
    },
  },
  required: ['sufficientEvidence', 'taskCoverage', 'summary', 'findings'],
};

const judgeSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    dimensions: {
      type: 'array',
      minItems: 3,
      maxItems: 3,
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          id: { type: 'string', enum: LINGUISTIC_DIMENSIONS },
          status: {
            type: 'string',
            enum: ['scored', 'insufficientEvidence'],
          },
          band: { type: 'integer', minimum: 0, maximum: 4 },
          evidenceIds: {
            type: 'array',
            maxItems: 12,
            items: { type: 'string' },
          },
          rationale: { type: 'string' },
        },
        required: ['id', 'status', 'band', 'evidenceIds', 'rationale'],
      },
    },
  },
  required: ['dimensions'],
};

const feedbackSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    priority: { type: 'string' },
    activity: { type: 'string' },
    rationale: { type: 'string' },
    byDimension: {
      type: 'array',
      maxItems: 5,
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          id: { type: 'string', enum: DIMENSION_IDS },
          recommendation: { type: 'string' },
          evidenceIds: {
            type: 'array',
            maxItems: 8,
            items: { type: 'string' },
          },
        },
        required: ['id', 'recommendation', 'evidenceIds'],
      },
    },
  },
  required: ['priority', 'activity', 'rationale', 'byDimension'],
};

const rubricDraftSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    communicativePurpose: { type: 'string' },
    targetConcepts: {
      type: 'array',
      maxItems: 12,
      items: { type: 'string' },
    },
    vocabularyHints: {
      type: 'array',
      maxItems: 24,
      items: { type: 'string' },
    },
    descriptors: {
      type: 'array',
      minItems: 5,
      maxItems: 5,
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          id: { type: 'string', enum: DIMENSION_IDS },
          descriptor: { type: 'string' },
        },
        required: ['id', 'descriptor'],
      },
    },
    warnings: {
      type: 'array',
      maxItems: 8,
      items: { type: 'string' },
    },
  },
  required: [
    'communicativePurpose',
    'targetConcepts',
    'vocabularyHints',
    'descriptors',
    'warnings',
  ],
};

const instructionImprovementSchema = {
  type: 'object',
  additionalProperties: false,
  properties: {
    improvedInstruction: { type: 'string' },
    detectedAudience: {
      type: 'string',
      enum: ['student', 'evaluator', 'mixed', 'unclear'],
    },
    summary: { type: 'string' },
    preservedRequirements: {
      type: 'array',
      maxItems: 16,
      items: { type: 'string' },
    },
    warnings: {
      type: 'array',
      maxItems: 8,
      items: { type: 'string' },
    },
  },
  required: [
    'improvedInstruction',
    'detectedAudience',
    'summary',
    'preservedRequirements',
    'warnings',
  ],
};

function parseStructured(response, name) {
  const content = response?.choices?.[0]?.message?.content;
  if (typeof content !== 'string' || !content.trim()) {
    throw new EvaluationError(
      502,
      `El modelo no devolvió ${name}.`,
      'LINGUISTIC_MODEL_EMPTY',
    );
  }
  try {
    return JSON.parse(content);
  } catch {
    throw new EvaluationError(
      502,
      `El modelo devolvió ${name} inválido.`,
      'LINGUISTIC_MODEL_INVALID_JSON',
    );
  }
}

async function callStructured({
  client,
  model,
  schema,
  schemaName,
  system,
  payload,
  maxTokens = 2000,
}) {
  let response;
  let lastError;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      response = await withCircuitBreaker('groq-linguistic', () =>
        client.chat.completions.create(
          {
          model,
          temperature: 0,
          reasoning_effort: 'low',
          max_completion_tokens: maxTokens,
          messages: [
            { role: 'system', content: system },
            {
              role: 'user',
              content: JSON.stringify({
          security:
            INTERNAL_PROMPTS.securityEnvelope,
                data: payload,
              }),
            },
          ],
          response_format: {
            type: 'json_schema',
            json_schema: { name: schemaName, strict: true, schema },
          },
          },
          { timeout: 90_000 },
        ),
      );
      break;
    } catch (error) {
      lastError = error;
      const retryable =
        error?.status === 429 ||
        (Number.isInteger(error?.status) && error.status >= 500);
      if (!retryable || attempt === 2) break;
      await new Promise((resolve) =>
        setTimeout(
          resolve,
          250 * 3 ** attempt + Math.floor(Math.random() * 150),
        ),
      );
    }
  }
  if (!response) {
    throw new EvaluationError(
      lastError?.status === 429 ? 429 : 502,
      lastError?.status === 429
        ? 'El modelo lingüístico alcanzó su límite temporal.'
        : 'El modelo lingüístico no estuvo disponible.',
      lastError?.status === 429
        ? 'LINGUISTIC_RATE_LIMIT'
        : 'LINGUISTIC_PROVIDER_ERROR',
    );
  }
  return parseStructured(response, schemaName);
}

export async function enhanceRubricDraftWithAI({
  client,
  model,
  draft,
}) {
  const suggestion = await callStructured({
    client,
    model,
    schema: rubricDraftSchema,
    schemaName: 'gordon_rubric_draft',
    maxTokens: 1600,
    system: INTERNAL_PROMPTS.rubricCompiler,
    payload: { draft },
  });
  const descriptorMap = new Map(
    (Array.isArray(suggestion.descriptors) ? suggestion.descriptors : []).map(
      (item) => [item.id, item.descriptor],
    ),
  );
  return {
    ...draft,
    spec: {
      ...draft.spec,
      communicativePurpose:
        typeof suggestion.communicativePurpose === 'string'
          ? suggestion.communicativePurpose.trim().slice(0, 2000)
          : draft.spec.communicativePurpose,
      targetConcepts: Array.isArray(suggestion.targetConcepts)
        ? suggestion.targetConcepts
            .filter((item) => typeof item === 'string')
            .map((item) => item.trim().slice(0, 120))
            .filter(Boolean)
            .slice(0, 12)
        : draft.spec.targetConcepts,
      vocabularyHints: Array.isArray(suggestion.vocabularyHints)
        ? suggestion.vocabularyHints
            .filter((item) => typeof item === 'string')
            .map((item) => item.trim().slice(0, 80))
            .filter(Boolean)
            .slice(0, 24)
        : draft.spec.vocabularyHints,
    },
    dimensions: draft.dimensions.map((dimension) => ({
      ...dimension,
      descriptor:
        typeof descriptorMap.get(dimension.id) === 'string'
          ? descriptorMap.get(dimension.id).trim().slice(0, 500)
          : dimension.descriptor,
    })),
    warnings: Array.isArray(suggestion.warnings)
      ? suggestion.warnings
          .filter((item) => typeof item === 'string')
          .map((item) => item.trim().slice(0, 500))
          .filter(Boolean)
          .slice(0, 8)
      : [],
    generatedBy: {
      provider: 'groq',
      model,
      promptVersion: PROMPT_VERSION,
    },
  };
}

export async function improveStudentInstructionWithAI({
  client,
  model,
  spec,
}) {
  const suggestion = await callStructured({
    client,
    model,
    schema: instructionImprovementSchema,
    schemaName: 'gordon_instruction_improvement',
    maxTokens: 1000,
    system: INTERNAL_PROMPTS.instructionImprover,
    payload: { spec },
  });
  const improvedInstruction =
    typeof suggestion.improvedInstruction === 'string'
      ? suggestion.improvedInstruction.trim()
      : '';
  if (!improvedInstruction || improvedInstruction.length > 2000) {
    throw new EvaluationError(
      502,
      'El modelo no devolvió una consigna mejorada válida.',
      'INVALID_IMPROVED_INSTRUCTION',
    );
  }
  const cleanList = (value, maxItems, maxLength) =>
    Array.isArray(value)
      ? value
          .filter((item) => typeof item === 'string')
          .map((item) => item.trim().slice(0, maxLength))
          .filter(Boolean)
          .slice(0, maxItems)
      : [];
  return {
    originalInstruction: spec.instruction,
    improvedInstruction,
    detectedAudience: [
      'student',
      'evaluator',
      'mixed',
      'unclear',
    ].includes(suggestion.detectedAudience)
      ? suggestion.detectedAudience
      : 'unclear',
    summary:
      typeof suggestion.summary === 'string'
        ? suggestion.summary.trim().slice(0, 500)
        : '',
    preservedRequirements: cleanList(
      suggestion.preservedRequirements,
      16,
      160,
    ),
    warnings: cleanList(suggestion.warnings, 8, 500),
    generatedBy: {
      provider: 'groq',
      model,
      promptVersion: PROMPT_VERSION,
    },
  };
}

function transcriptTokens(transcript) {
  const matches = [
    ...(typeof transcript === 'string' ? transcript : '').matchAll(
      /[\p{L}\p{N}']+/gu,
    ),
  ];
  return matches.map((match, index) => ({
    id: `t${index}`,
    index,
    text: match[0],
    charStart: match.index,
    charEnd: match.index + match[0].length,
  }));
}

function validateFindings(findings, tokens) {
  const usedIds = new Set();
  return (Array.isArray(findings) ? findings : [])
    .filter((finding) => {
      if (
        !finding ||
        !LINGUISTIC_DIMENSIONS.includes(finding.dimension) ||
        !Number.isInteger(finding.tokenStart) ||
        !Number.isInteger(finding.tokenEnd) ||
        finding.tokenStart < 0 ||
        finding.tokenEnd < finding.tokenStart ||
        finding.tokenEnd >= tokens.length
      ) {
        return false;
      }
      const expectedQuote = tokens
        .slice(finding.tokenStart, finding.tokenEnd + 1)
        .map((token) => token.text)
        .join(' ');
      if (
        normalizeTokens(expectedQuote).join(' ') !==
        normalizeTokens(finding.quote).join(' ')
      ) {
        return false;
      }
      const baseId =
        typeof finding.id === 'string' && finding.id.trim()
          ? finding.id.trim().slice(0, 80)
          : `e${usedIds.size}`;
      let id = baseId;
      let suffix = 1;
      while (usedIds.has(id)) id = `${baseId}-${suffix++}`;
      finding.id = id;
      usedIds.add(id);
      return true;
    })
    .map((finding) => ({
      id: finding.id,
      dimension: finding.dimension,
      kind: finding.type,
      claim: String(finding.claim ?? '').trim(),
      quote: String(finding.quote ?? '').trim(),
      correction: String(finding.correction ?? '').trim(),
      transcriptStart: finding.tokenStart,
      transcriptEnd: finding.tokenEnd,
      certainty:
        typeof finding.certainty === 'number' ? finding.certainty : null,
      source: 'gpt-oss-evidence-extractor',
    }));
}

function deterministicSufficiency({ rubric, transcript, quality }) {
  const lexicalCount = normalizeTokens(transcript).filter(
    (token) => token.length > 1,
  ).length;
  const voicedSeconds = quality?.metrics?.voicedSeconds ?? 0;
  const minimumWords = rubric.spec.mode === 'reading' ? 10 : 30;
  const minimumVoice = rubric.spec.mode === 'reading' ? 5 : 15;
  return {
    sufficient: lexicalCount >= minimumWords && voicedSeconds >= minimumVoice,
    lexicalCount,
    voicedSeconds,
    minimumWords,
    minimumVoice,
  };
}

export async function extractLinguisticEvidence({
  client,
  model,
  rubric,
  transcript,
  secondaryTranscript,
  quality,
  providerDisagreement,
}) {
  const tokens = transcriptTokens(transcript);
  const sufficiency = deterministicSufficiency({
    rubric,
    transcript,
    quality,
  });
  if (!sufficiency.sufficient) {
    return {
      sufficientEvidence: false,
      taskCoverage: null,
      summary: 'La muestra no alcanza el mínimo determinista de evidencia.',
      findings: [],
      tokens,
      sufficiency,
      promptVersion: PROMPT_VERSION,
    };
  }
  let raw = await callStructured({
    client,
    model,
    schema: evidenceSchema,
    schemaName: 'gordon_linguistic_evidence',
    system: INTERNAL_PROMPTS.evidenceExtractor,
    payload: {
      rubric,
      transcript,
      secondaryTranscript,
      tokens,
      quality,
      providerDisagreement,
    },
  });
  let findings = validateFindings(raw.findings, tokens);
  let extractorRepaired = false;
  if (raw.sufficientEvidence !== true) {
    const repaired = await callStructured({
      client,
      model,
      schema: evidenceSchema,
      schemaName: 'gordon_linguistic_evidence_repair',
      system: INTERNAL_PROMPTS.evidenceExtractor,
      payload: {
        rubric,
        transcript,
        secondaryTranscript,
        tokens,
        quality,
        providerDisagreement,
        repair: {
          reason:
            'La muestra superó los mínimos deterministas. Revisa cada dimensión por separado y no confundas baja cobertura de la tarea con ausencia de gramática o vocabulario.',
          previousResult: {
            sufficientEvidence: raw.sufficientEvidence,
            taskCoverage: raw.taskCoverage,
            summary: raw.summary,
            findings: raw.findings,
          },
        },
      },
    });
    const repairedFindings = validateFindings(repaired.findings, tokens);
    if (
      repaired.sufficientEvidence === true ||
      repairedFindings.length > findings.length
    ) {
      raw = repaired;
      findings = repairedFindings;
      extractorRepaired = true;
    }
  }
  return {
    sufficientEvidence: sufficiency.sufficient,
    taskCoverage:
      typeof raw.taskCoverage === 'number' ? raw.taskCoverage : null,
    summary: typeof raw.summary === 'string' ? raw.summary.trim() : '',
    findings,
    tokens,
    sufficiency,
    extractorDeclaredSufficient: raw.sufficientEvidence === true,
    extractorRepaired,
    promptVersion: PROMPT_VERSION,
  };
}

function normalizeJudge(raw, findings) {
  const evidenceIdsByDimension = Object.fromEntries(
    LINGUISTIC_DIMENSIONS.map((id) => [
      id,
      (Array.isArray(findings) ? findings : [])
        .filter(
          (finding) =>
            finding?.dimension === id &&
            typeof finding?.id === 'string' &&
            finding.id.trim(),
        )
        .map((finding) => finding.id.trim())
        .filter((id, index, ids) => ids.indexOf(id) === index)
        .slice(0, 12),
    ]),
  );
  const byId = {};
  for (const item of Array.isArray(raw?.dimensions) ? raw.dimensions : []) {
    if (
      !LINGUISTIC_DIMENSIONS.includes(item?.id) ||
      byId[item.id] ||
      !['scored', 'insufficientEvidence'].includes(item.status)
    ) {
      continue;
    }
    const band =
      Number.isInteger(item.band) && item.band >= 0 && item.band <= 4
        ? item.band
        : 0;
    const availableEvidenceIds = evidenceIdsByDimension[item.id];
    const availableEvidenceSet = new Set(availableEvidenceIds);
    const citedEvidenceIds = Array.isArray(item.evidenceIds)
      ? item.evidenceIds
          .filter((id) => typeof id === 'string')
          .map((id) => id.trim())
          .filter((id) => availableEvidenceSet.has(id))
          .filter((id, index, ids) => ids.indexOf(id) === index)
          .slice(0, 12)
      : [];
    const citationRepaired =
      item.status === 'scored' &&
      citedEvidenceIds.length === 0 &&
      availableEvidenceIds.length > 0;
    const evidenceIds = citationRepaired
      ? availableEvidenceIds
      : citedEvidenceIds;
    const status =
      item.status === 'scored' && evidenceIds.length
        ? 'scored'
        : 'insufficientEvidence';
    byId[item.id] = {
      id: item.id,
      status,
      band,
      evidenceIds,
      citationRepaired,
      reasonCode:
        status === 'scored'
          ? null
          : item.status === 'scored'
            ? 'JUDGE_EVIDENCE_NOT_CITED'
            : 'INSUFFICIENT_LINGUISTIC_EVIDENCE',
      rationale:
        typeof item.rationale === 'string' ? item.rationale.trim() : '',
    };
  }
  return Object.fromEntries(
    LINGUISTIC_DIMENSIONS.map((id) => [
      id,
      byId[id] ?? {
        id,
        status: 'insufficientEvidence',
        band: 0,
        evidenceIds: [],
        citationRepaired: false,
        reasonCode: 'JUDGE_DIMENSION_MISSING',
        rationale: 'El juez no produjo evidencia verificable.',
      },
    ]),
  );
}

async function runJudge({
  client,
  model,
  rubric,
  evidence,
  perspective,
}) {
  const isAnalytic = perspective === 'analytic';
  const raw = await callStructured({
    client,
    model,
    schema: judgeSchema,
    schemaName: `gordon_${perspective}_judge`,
    system: isAnalytic
      ? INTERNAL_PROMPTS.analyticJudge
      : INTERNAL_PROMPTS.holisticJudge,
    payload: {
      rubric,
      evidence: {
        sufficientEvidence: evidence.sufficientEvidence,
        taskCoverage: evidence.taskCoverage,
        summary: evidence.summary,
        findings: evidence.findings,
      },
    },
  });
  return normalizeJudge(raw, evidence.findings);
}

async function adjudicate({
  client,
  model,
  rubric,
  evidence,
  analytic,
  holistic,
  disputedIds,
}) {
  const raw = await callStructured({
    client,
    model,
    schema: judgeSchema,
    schemaName: 'gordon_adjudication',
    system: INTERNAL_PROMPTS.adjudicator,
    payload: { rubric, evidence, analytic, holistic, disputedIds },
  });
  return normalizeJudge(raw, evidence.findings);
}

export async function runDoubleLinguisticJudging({
  client,
  model,
  rubric,
  evidence,
}) {
  if (!evidence.sufficientEvidence) {
    return {
      dimensions: Object.fromEntries(
        LINGUISTIC_DIMENSIONS.map((id) => [
          id,
          {
            status: 'insufficientEvidence',
            score: null,
            probabilities: null,
            interval90: null,
            evidenceIds: [],
            rationale: evidence.summary,
            reasonCode:
              evidence.sufficiency?.sufficient === false
                ? 'INSUFFICIENT_LINGUISTIC_SAMPLE'
                : 'INSUFFICIENT_LINGUISTIC_EVIDENCE',
            judgeAgreement: null,
            reviewRequired: true,
          },
        ]),
      ),
      adjudicated: false,
      promptVersion: PROMPT_VERSION,
    };
  }
  const [analytic, holistic] = await Promise.all([
    runJudge({
      client,
      model,
      rubric,
      evidence,
      perspective: 'analytic',
    }),
    runJudge({
      client,
      model,
      rubric,
      evidence,
      perspective: 'holistic',
    }),
  ]);
  const disputedIds = LINGUISTIC_DIMENSIONS.filter((id) => {
    const a = analytic[id];
    const b = holistic[id];
    return (
      a.status !== b.status ||
      (a.status === 'scored' && a.band !== b.band)
    );
  });
  const adjudicated =
    disputedIds.length > 0
      ? await adjudicate({
          client,
          model,
          rubric,
          evidence,
          analytic,
          holistic,
          disputedIds,
        })
      : null;
  const dimensions = {};
  for (const id of LINGUISTIC_DIMENSIONS) {
    const a = analytic[id];
    const b = holistic[id];
    const final = disputedIds.includes(id) ? adjudicated[id] : null;
    if (
      (final && final.status !== 'scored') ||
      (!final && (a.status !== 'scored' || b.status !== 'scored'))
    ) {
      dimensions[id] = {
        status: 'insufficientEvidence',
        score: null,
        probabilities: null,
        interval90: null,
        evidenceIds: [...new Set([...a.evidenceIds, ...b.evidenceIds])],
        rationale: final?.rationale ?? `${a.rationale} ${b.rationale}`.trim(),
        reasonCode:
          final?.reasonCode ??
          a.reasonCode ??
          b.reasonCode ??
          'INSUFFICIENT_LINGUISTIC_EVIDENCE',
        judgeAgreement: false,
        reviewRequired: true,
      };
      continue;
    }
    const band = final
      ? final.band
      : Math.round((a.band + b.band) / 2);
    const uncertainty = final ? 0.3 : a.band === b.band ? 0.15 : 0.25;
    const probabilities = probabilitiesForBand(band, uncertainty);
    const score = expectedScore(probabilities);
    const halfWidth = final ? 15 : a.band === b.band ? 10 : 12.5;
    dimensions[id] = {
      status: 'scored',
      score,
      probabilities,
      interval90: {
        low: Math.max(0, score - halfWidth),
        high: Math.min(100, score + halfWidth),
      },
      evidenceIds: [
        ...new Set([
          ...a.evidenceIds,
          ...b.evidenceIds,
          ...(final?.evidenceIds ?? []),
        ]),
      ],
      rationale: final?.rationale ?? `${a.rationale} ${b.rationale}`.trim(),
      judgeAgreement: a.band === b.band,
      judgeBands: { analytic: a.band, holistic: b.band, adjudicated: final?.band },
      reviewRequired:
        disputedIds.includes(id) ||
        a.citationRepaired ||
        b.citationRepaired ||
        final?.citationRepaired === true,
    };
  }
  return {
    dimensions,
    adjudicated: disputedIds.length > 0,
    disputedIds,
    promptVersion: PROMPT_VERSION,
    promptHash: PROMPT_MANIFEST_HASH,
  };
}

export async function generatePedagogicalFeedback({
  client,
  model,
  rubric,
  dimensions,
  evidence,
}) {
  return callStructured({
    client,
    model,
    schema: feedbackSchema,
    schemaName: 'gordon_pedagogical_feedback',
    maxTokens: 1200,
    system: INTERNAL_PROMPTS.feedbackGenerator,
    payload: {
      rubric,
      dimensions,
      evidence: evidence.findings,
    },
  });
}
