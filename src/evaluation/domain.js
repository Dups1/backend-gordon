import {
  createHash,
  createHmac,
  randomBytes,
  randomUUID,
  timingSafeEqual,
} from 'node:crypto';

export const SCHEMA_VERSION = '2.0.0';
export const CALIBRATION_VERSION = 'provisional-en-US-v1';
export const PROMPT_VERSION = 'gordon-evidence-v1.1';
export const SUPPORTED_LOCALE = 'en-US';
export const DIMENSION_IDS = Object.freeze([
  'communication',
  'pronunciation',
  'grammar',
  'vocabulary',
  'fluency',
]);

export const SCORE_PROFILES = Object.freeze({
  reading: Object.freeze({
    id: 'reading-enUS-cefr-v1',
    version: 1,
    weights: Object.freeze({
      communication: 0.2,
      pronunciation: 0.3,
      grammar: 0.15,
      vocabulary: 0.15,
      fluency: 0.2,
    }),
  }),
  spontaneous: Object.freeze({
    id: 'spontaneous-enUS-cefr-v1',
    version: 1,
    weights: Object.freeze({
      communication: 0.25,
      pronunciation: 0.2,
      grammar: 0.2,
      vocabulary: 0.2,
      fluency: 0.15,
    }),
  }),
});

const cefrLevels = new Set(['A1', 'A2', 'B1', 'B2', 'C1', 'C2']);
const modes = new Set(['reading', 'spontaneous']);
const processSecret = randomBytes(32).toString('hex');

export class EvaluationError extends Error {
  constructor(status, message, code, details) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

function requiredText(value, field, { max = 5000 } = {}) {
  const text = typeof value === 'string' ? value.trim() : '';
  if (!text) {
    throw new EvaluationError(
      400,
      `El campo ${field} es obligatorio.`,
      'INVALID_EVALUATION_SPEC',
      { field },
    );
  }
  if (text.length > max) {
    throw new EvaluationError(
      400,
      `El campo ${field} supera el límite permitido.`,
      'INVALID_EVALUATION_SPEC',
      { field, max },
    );
  }
  return text;
}

function optionalText(value, { max = 2000 } = {}) {
  const text = typeof value === 'string' ? value.trim() : '';
  return text.slice(0, max);
}

function uniqueTexts(value, { maxItems = 40, maxLength = 120 } = {}) {
  if (!Array.isArray(value)) return [];
  return [
    ...new Set(
      value
        .filter((item) => typeof item === 'string')
        .map((item) => item.trim())
        .filter(Boolean)
        .map((item) => item.slice(0, maxLength)),
    ),
  ].slice(0, maxItems);
}

export function normalizeEvaluationSpec(input = {}) {
  const mode = typeof input.mode === 'string' ? input.mode.trim() : '';
  if (!modes.has(mode)) {
    throw new EvaluationError(
      400,
      'El modo debe ser reading o spontaneous.',
      'INVALID_EVALUATION_MODE',
    );
  }
  const targetLocale =
    typeof input.targetLocale === 'string' ? input.targetLocale.trim() : '';
  if (targetLocale !== SUPPORTED_LOCALE) {
    throw new EvaluationError(
      422,
      'La evaluación completa está calibrada únicamente para en-US.',
      'LOCALE_NOT_CALIBRATED',
      { supportedLocale: SUPPORTED_LOCALE },
    );
  }
  const cefr =
    typeof input.cefr === 'string' ? input.cefr.trim().toUpperCase() : '';
  if (!cefrLevels.has(cefr)) {
    throw new EvaluationError(
      400,
      'El nivel CEFR debe ser A1, A2, B1, B2, C1 o C2.',
      'INVALID_CEFR_LEVEL',
    );
  }
  const instruction = requiredText(input.instruction, 'instruction', {
    max: 2000,
  });
  const referenceText =
    mode === 'reading'
      ? requiredText(input.referenceText, 'referenceText', { max: 15000 })
      : '';
  const profile = SCORE_PROFILES[mode];
  return {
    mode,
    targetLocale,
    asrLanguage: 'en',
    cefr,
    instruction,
    referenceText,
    communicativePurpose: optionalText(input.communicativePurpose),
    targetConcepts: uniqueTexts(input.targetConcepts),
    vocabularyHints: uniqueTexts(input.vocabularyHints, {
      maxItems: 24,
      maxLength: 80,
    }),
    teacherNotes: optionalText(input.teacherNotes),
    profileId: profile.id,
    profileVersion: profile.version,
  };
}

const descriptors = Object.freeze({
  communication:
    'Cumple la tarea y transmite un significado comprensible para el nivel objetivo.',
  pronunciation:
    'Produce palabras y fonemas de manera inteligible, sin exigir semejanza con un acento nativo.',
  grammar:
    'Utiliza o realiza las estructuras gramaticales exigidas por la tarea.',
  vocabulary:
    'Utiliza o realiza vocabulario adecuado y comprensible para la tarea y el nivel.',
  fluency:
    'Mantiene continuidad, ritmo, pausas y prosodia funcionales para el nivel.',
});

export function createRubricDraft(input) {
  const spec = normalizeEvaluationSpec(input);
  const readingLimitations =
    spec.mode === 'reading'
      ? [
          'Grammar mide la realización oral de formas presentes en el texto, no producción gramatical libre.',
          'Vocabulary mide cobertura e inteligibilidad del léxico del texto, no selección léxica espontánea.',
          'Communication mide preservación e inteligibilidad del significado del texto.',
        ]
      : [];
  return {
    id: randomUUID(),
    version: 1,
    status: 'draft',
    createdAt: new Date().toISOString(),
    spec,
    dimensions: DIMENSION_IDS.map((id) => ({
      id,
      descriptor: descriptors[id],
      constructScope:
        spec.mode === 'reading' &&
        ['communication', 'grammar', 'vocabulary'].includes(id)
          ? 'reading_realization'
          : 'productive_speaking',
      bands: [
        { band: 0, label: 'Sin evidencia suficiente' },
        { band: 1, label: 'Muy por debajo del objetivo' },
        { band: 2, label: 'Dominio parcial' },
        { band: 3, label: 'Cumple el objetivo' },
        { band: 4, label: 'Cumple de manera consistente' },
      ],
    })),
    scoreProfile: SCORE_PROFILES[spec.mode],
    limitations: [
      'El score representa dominio del nivel CEFR objetivo; no estima el nivel CEFR global.',
      'Los resultados automáticos son provisionales hasta completar la calibración humana.',
      ...readingLimitations,
    ],
  };
}

function normalizeConfirmedDimensions(value, canonicalDimensions) {
  if (!Array.isArray(value) || value.length !== DIMENSION_IDS.length) {
    throw new EvaluationError(
      400,
      'La rúbrica debe contener exactamente las cinco dimensiones.',
      'INVALID_RUBRIC_DIMENSIONS',
    );
  }
  const byId = new Map();
  for (const dimension of value) {
    const id =
      dimension && typeof dimension.id === 'string'
        ? dimension.id.trim()
        : '';
    if (!DIMENSION_IDS.includes(id) || byId.has(id)) {
      throw new EvaluationError(
        400,
        'Las dimensiones de la rúbrica no son válidas o están repetidas.',
        'INVALID_RUBRIC_DIMENSIONS',
      );
    }
    byId.set(id, dimension);
  }
  return canonicalDimensions.map((canonical) => {
    const proposed = byId.get(canonical.id);
    if (proposed.constructScope !== canonical.constructScope) {
      throw new EvaluationError(
        400,
        `No se puede alterar el alcance de ${canonical.id}.`,
        'INVALID_CONSTRUCT_SCOPE',
      );
    }
    return {
      ...canonical,
      descriptor: requiredText(
        proposed.descriptor,
        `dimensions.${canonical.id}.descriptor`,
        { max: 800 },
      ),
    };
  });
}

function canonicalJson(value) {
  if (Array.isArray(value)) {
    return `[${value.map(canonicalJson).join(',')}]`;
  }
  if (value && typeof value === 'object') {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(',')}}`;
  }
  return JSON.stringify(value);
}

export function hashObject(value) {
  return createHash('sha256').update(canonicalJson(value)).digest('hex');
}

function signingSecret(override) {
  return (
    (typeof override === 'string' && override.trim()) ||
    process.env.RUBRIC_SIGNING_SECRET?.trim() ||
    processSecret
  );
}

export function confirmRubric(draft, { secret } = {}) {
  if (!draft || typeof draft !== 'object' || draft.status !== 'draft') {
    throw new EvaluationError(
      400,
      'La rúbrica por confirmar no es válida.',
      'INVALID_RUBRIC_DRAFT',
    );
  }
  const normalized = createRubricDraft(draft.spec);
  const confirmedDimensions = normalizeConfirmedDimensions(
    draft.dimensions,
    normalized.dimensions,
  );
  const confirmedLimitations = uniqueTexts(draft.limitations, {
    maxItems: 12,
    maxLength: 500,
  });
  const confirmed = {
    ...normalized,
    id:
      typeof draft.id === 'string' && draft.id.trim()
        ? draft.id.trim().slice(0, 120)
        : normalized.id,
    version:
      Number.isInteger(draft.version) &&
      draft.version > 0 &&
      draft.version <= 1000
        ? draft.version
        : 1,
    status: 'confirmed',
    confirmedAt: new Date().toISOString(),
    dimensions: confirmedDimensions,
    limitations: confirmedLimitations.length
      ? confirmedLimitations
      : normalized.limitations,
  };
  delete confirmed.createdAt;
  const payload = Buffer.from(
    JSON.stringify({
      schemaVersion: SCHEMA_VERSION,
      rubric: confirmed,
      rubricHash: hashObject(confirmed),
      issuedAt: Date.now(),
    }),
  ).toString('base64url');
  const signature = createHmac('sha256', signingSecret(secret))
    .update(payload)
    .digest('base64url');
  return {
    confirmedRubric: confirmed,
    confirmedRubricToken: `${payload}.${signature}`,
    rubricHash: hashObject(confirmed),
  };
}

export function verifyRubricToken(token, { secret } = {}) {
  if (typeof token !== 'string' || !token.includes('.')) {
    throw new EvaluationError(
      400,
      'Falta una rúbrica confirmada.',
      'CONFIRMED_RUBRIC_REQUIRED',
    );
  }
  const [payload, suppliedSignature, ...extra] = token.split('.');
  if (!payload || !suppliedSignature || extra.length) {
    throw new EvaluationError(
      400,
      'El token de rúbrica no es válido.',
      'INVALID_RUBRIC_TOKEN',
    );
  }
  const expected = createHmac('sha256', signingSecret(secret))
    .update(payload)
    .digest();
  let supplied;
  try {
    supplied = Buffer.from(suppliedSignature, 'base64url');
  } catch {
    supplied = Buffer.alloc(0);
  }
  if (
    expected.length !== supplied.length ||
    !timingSafeEqual(expected, supplied)
  ) {
    throw new EvaluationError(
      401,
      'La rúbrica confirmada no pudo verificarse.',
      'INVALID_RUBRIC_SIGNATURE',
    );
  }
  let decoded;
  try {
    decoded = JSON.parse(Buffer.from(payload, 'base64url').toString('utf8'));
  } catch {
    throw new EvaluationError(
      400,
      'El token de rúbrica está dañado.',
      'INVALID_RUBRIC_TOKEN',
    );
  }
  if (
    decoded.schemaVersion !== SCHEMA_VERSION ||
    decoded.rubric?.status !== 'confirmed' ||
    decoded.rubricHash !== hashObject(decoded.rubric)
  ) {
    throw new EvaluationError(
      400,
      'La versión o contenido de la rúbrica no es válido.',
      'INVALID_RUBRIC_TOKEN',
    );
  }
  return decoded.rubric;
}

export function finiteScore(value) {
  return typeof value === 'number' &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 100
    ? value
    : null;
}

export function scoreFromBand(band) {
  return Number.isInteger(band) && band >= 0 && band <= 4 ? band * 25 : null;
}

export function probabilitiesForBand(band, uncertainty = 0.2) {
  if (!Number.isInteger(band) || band < 0 || band > 4) return null;
  const spread = Math.max(0.05, Math.min(0.45, uncertainty));
  const probabilities = Array(5).fill(0);
  probabilities[band] = 1 - spread;
  if (band > 0 && band < 4) {
    probabilities[band - 1] = spread / 2;
    probabilities[band + 1] = spread / 2;
  } else {
    probabilities[band === 0 ? 1 : 3] = spread;
  }
  return probabilities;
}

export function probabilitiesForScore(score) {
  const valid = finiteScore(score);
  if (valid === null) return null;
  const position = valid / 25;
  const lower = Math.min(4, Math.floor(position));
  const upper = Math.min(4, Math.ceil(position));
  const probabilities = Array(5).fill(0);
  if (lower === upper) {
    probabilities[lower] = 1;
    return probabilities;
  }
  const upperWeight = position - lower;
  probabilities[lower] = 1 - upperWeight;
  probabilities[upper] = upperWeight;
  return probabilities;
}

export function expectedScore(probabilities) {
  if (
    !Array.isArray(probabilities) ||
    probabilities.length !== 5 ||
    probabilities.some(
      (value) => typeof value !== 'number' || !Number.isFinite(value) || value < 0,
    )
  ) {
    return null;
  }
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  if (total <= 0) return null;
  return (
    probabilities.reduce((sum, value, band) => sum + value * band * 25, 0) /
    total
  );
}

export function dimensionResult({
  id,
  status,
  score = null,
  rawScore = null,
  probabilities = null,
  reliability = 'unknown',
  methodId,
  evidence = [],
  limitations = [],
  reasonCode = null,
  interval90 = null,
  constructScope = 'productive_speaking',
  reviewRequired = false,
}) {
  if (!DIMENSION_IDS.includes(id)) {
    throw new EvaluationError(500, 'Dimensión desconocida.', 'INVALID_DIMENSION');
  }
  const validScore = finiteScore(score);
  const rawScoreProvided = rawScore !== null && rawScore !== undefined;
  const validRawScore = finiteScore(rawScore);
  const probabilitiesValid =
    probabilities === null ||
    (Array.isArray(probabilities) &&
      probabilities.length === 5 &&
      probabilities.every(
        (value) =>
          typeof value === 'number' &&
          Number.isFinite(value) &&
          value >= 0 &&
          value <= 1,
      ) &&
      Math.abs(
        probabilities.reduce((sum, value) => sum + value, 0) - 1,
      ) <= 0.001);
  const intervalValid =
    interval90 === null ||
    (typeof interval90 === 'object' &&
      typeof interval90.low === 'number' &&
      Number.isFinite(interval90.low) &&
      typeof interval90.high === 'number' &&
      Number.isFinite(interval90.high) &&
      interval90.low >= 0 &&
      interval90.high <= 100 &&
      interval90.low <= interval90.high);
  if (
    status === 'scored' &&
    (validScore === null ||
      (rawScoreProvided && validRawScore === null) ||
      !probabilitiesValid ||
      !intervalValid)
  ) {
    return dimensionResult({
      id,
      status: 'providerError',
      methodId,
      reliability: 'unknown',
      limitations: [
        ...limitations,
        'El proveedor devolvió una puntuación inválida.',
      ],
      reasonCode: 'INVALID_PROVIDER_SCORE',
      constructScope,
      reviewRequired: true,
    });
  }
  return {
    id,
    status,
    score: status === 'scored' ? validScore : null,
    rawScore: validRawScore,
    bandProbabilities:
      status === 'scored' && Array.isArray(probabilities)
        ? probabilities
        : null,
    interval90:
      status === 'scored' && interval90
        ? {
            low: interval90.low,
            high: interval90.high,
          }
        : null,
    reliability,
    methodId,
    calibrationVersion: CALIBRATION_VERSION,
    calibrationStatus: 'provisional',
    constructScope,
    reviewRequired,
    evidence: Array.isArray(evidence) ? evidence : [],
    limitations: Array.isArray(limitations) ? limitations : [],
    reasonCode,
  };
}

export function calculateOverall(dimensions, profile) {
  const missing = DIMENSION_IDS.filter(
    (id) => dimensions[id]?.status !== 'scored',
  );
  if (missing.length) {
    return {
      status: 'insufficientEvidence',
      score: null,
      interval90: null,
      reliability: 'unknown',
      profileId: profile.id,
      profileVersion: profile.version,
      reasonCode: 'REQUIRED_DIMENSIONS_MISSING',
      missingDimensions: missing,
      calibrationVersion: CALIBRATION_VERSION,
    };
  }
  const score = DIMENSION_IDS.reduce(
    (sum, id) => sum + dimensions[id].score * profile.weights[id],
    0,
  );
  const lows = DIMENSION_IDS.map(
    (id) => dimensions[id].interval90?.low ?? dimensions[id].score,
  );
  const highs = DIMENSION_IDS.map(
    (id) => dimensions[id].interval90?.high ?? dimensions[id].score,
  );
  const interval90 = {
    low: DIMENSION_IDS.reduce(
      (sum, id, index) => sum + lows[index] * profile.weights[id],
      0,
    ),
    high: DIMENSION_IDS.reduce(
      (sum, id, index) => sum + highs[index] * profile.weights[id],
      0,
    ),
  };
  const reviewRequired = DIMENSION_IDS.some(
    (id) => dimensions[id].reviewRequired,
  );
  return {
    status: 'scored',
    score,
    interval90,
    reliability: reviewRequired ? 'low' : 'medium',
    profileId: profile.id,
    profileVersion: profile.version,
    calibrationVersion: CALIBRATION_VERSION,
    calibrationStatus: 'provisional',
    reviewRequired: true,
    limitation:
      'Puntaje provisional de apoyo docente; no equivale a una clasificación CEFR global.',
  };
}

export function normalizeTokens(text) {
  return (typeof text === 'string' ? text : '')
    .toLocaleLowerCase('en-US')
    .normalize('NFKD')
    .replace(/\p{M}/gu, '')
    .match(/[\p{L}\p{N}']+/gu) ?? [];
}

export function tokenErrorRate(primary, secondary) {
  const a = normalizeTokens(primary);
  const b = normalizeTokens(secondary);
  if (!a.length && !b.length) return 0;
  if (!a.length || !b.length) return 1;
  const row = Array(b.length + 1)
    .fill(0)
    .map((_, index) => index);
  for (let i = 1; i <= a.length; i++) {
    let previous = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const saved = row[j];
      row[j] = Math.min(
        row[j] + 1,
        row[j - 1] + 1,
        previous + (a[i - 1] === b[j - 1] ? 0 : 1),
      );
      previous = saved;
    }
  }
  return Math.min(1, row[b.length] / Math.max(a.length, b.length));
}

export function newAssessmentId() {
  return randomUUID();
}
