import { createReadStream } from 'node:fs';
import readline from 'node:readline';

import { DIMENSION_IDS, SCORE_PROFILES } from './domain.js';

export const CALIBRATION_REPORT_VERSION = '1.0.0';

const VALID_SPLITS = new Set(['train', 'validation', 'test']);
const VALID_MODES = new Set(['reading', 'spontaneous']);
const VALID_CEFR = new Set(['A1', 'A2', 'B1', 'B2', 'C1', 'C2']);
const VALID_HUMAN_RATING_STATUS = new Set(['consensus', 'adjudicated']);

export class CalibrationError extends Error {
  constructor(message, { code = 'INVALID_CALIBRATION_DATA', line = null } = {}) {
    super(line === null ? message : `Línea ${line}: ${message}`);
    this.name = 'CalibrationError';
    this.code = code;
    this.line = line;
  }
}

function finiteNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function roundMetric(value) {
  return value === null ? null : Math.round(value * 1_000_000) / 1_000_000;
}

export function meanAbsoluteError(predicted, expected) {
  if (
    !Array.isArray(predicted) ||
    !Array.isArray(expected) ||
    predicted.length === 0 ||
    predicted.length !== expected.length ||
    predicted.some((value) => !finiteNumber(value)) ||
    expected.some((value) => !finiteNumber(value))
  ) {
    return null;
  }
  return (
    predicted.reduce(
      (total, value, index) => total + Math.abs(value - expected[index]),
      0,
    ) / predicted.length
  );
}

function ranks(values) {
  const sorted = values
    .map((value, index) => ({ value, index }))
    .sort((a, b) => a.value - b.value);
  const result = Array(values.length);
  let cursor = 0;
  while (cursor < sorted.length) {
    let end = cursor + 1;
    while (end < sorted.length && sorted[end].value === sorted[cursor].value) {
      end++;
    }
    const averageRank = (cursor + 1 + end) / 2;
    for (let index = cursor; index < end; index++) {
      result[sorted[index].index] = averageRank;
    }
    cursor = end;
  }
  return result;
}

function pearsonCorrelation(left, right) {
  const count = left.length;
  const leftMean = left.reduce((sum, value) => sum + value, 0) / count;
  const rightMean = right.reduce((sum, value) => sum + value, 0) / count;
  let covariance = 0;
  let leftVariance = 0;
  let rightVariance = 0;
  for (let index = 0; index < count; index++) {
    const leftDelta = left[index] - leftMean;
    const rightDelta = right[index] - rightMean;
    covariance += leftDelta * rightDelta;
    leftVariance += leftDelta ** 2;
    rightVariance += rightDelta ** 2;
  }
  const denominator = Math.sqrt(leftVariance * rightVariance);
  return denominator === 0 ? null : covariance / denominator;
}

export function spearmanCorrelation(predicted, expected) {
  if (
    !Array.isArray(predicted) ||
    !Array.isArray(expected) ||
    predicted.length < 2 ||
    predicted.length !== expected.length ||
    predicted.some((value) => !finiteNumber(value)) ||
    expected.some((value) => !finiteNumber(value))
  ) {
    return null;
  }
  return pearsonCorrelation(ranks(predicted), ranks(expected));
}

export function quadraticWeightedKappa(predictedBands, expectedBands) {
  if (
    !Array.isArray(predictedBands) ||
    !Array.isArray(expectedBands) ||
    predictedBands.length < 2 ||
    predictedBands.length !== expectedBands.length ||
    [...predictedBands, ...expectedBands].some(
      (value) => !Number.isInteger(value) || value < 0 || value > 4,
    )
  ) {
    return null;
  }

  const categoryCount = 5;
  const observed = Array.from({ length: categoryCount }, () =>
    Array(categoryCount).fill(0),
  );
  const predictedTotals = Array(categoryCount).fill(0);
  const expectedTotals = Array(categoryCount).fill(0);
  for (let index = 0; index < predictedBands.length; index++) {
    const predicted = predictedBands[index];
    const expected = expectedBands[index];
    observed[predicted][expected]++;
    predictedTotals[predicted]++;
    expectedTotals[expected]++;
  }

  let observedDisagreement = 0;
  let expectedDisagreement = 0;
  const count = predictedBands.length;
  const maximumDistance = (categoryCount - 1) ** 2;
  for (let predicted = 0; predicted < categoryCount; predicted++) {
    for (let expected = 0; expected < categoryCount; expected++) {
      const weight = (predicted - expected) ** 2 / maximumDistance;
      observedDisagreement += weight * (observed[predicted][expected] / count);
      expectedDisagreement +=
        weight *
        ((predictedTotals[predicted] * expectedTotals[expected]) / count ** 2);
    }
  }
  return expectedDisagreement === 0
    ? null
    : 1 - observedDisagreement / expectedDisagreement;
}

function requiredText(value, field, line) {
  const text = typeof value === 'string' ? value.trim() : '';
  if (!text) {
    throw new CalibrationError(`${field} es obligatorio.`, { line });
  }
  return text;
}

function normalizeSubgroups(record, line) {
  if (
    record.subgroups !== undefined &&
    (record.subgroups === null ||
      typeof record.subgroups !== 'object' ||
      Array.isArray(record.subgroups))
  ) {
    throw new CalibrationError('subgroups debe ser un objeto.', { line });
  }
  const entries = [
    ...Object.entries(record.subgroups ?? {}).filter(
      ([key]) => key !== 'mode' && key !== 'cefr',
    ),
    ['mode', record.mode],
    ['cefr', record.cefr],
  ];
  const normalized = new Map();
  for (const [rawKey, rawValue] of entries) {
    const key = requiredText(rawKey, 'subgroup key', line);
    if (key.length > 80) {
      throw new CalibrationError('Una clave de subgrupo supera 80 caracteres.', {
        line,
      });
    }
    if (
      !['string', 'number', 'boolean'].includes(typeof rawValue) ||
      (typeof rawValue === 'number' && !Number.isFinite(rawValue))
    ) {
      throw new CalibrationError(
        `El subgrupo ${key} debe tener un valor escalar.`,
        { line },
      );
    }
    const value = String(rawValue).trim();
    if (!value || value.length > 120) {
      throw new CalibrationError(
        `El valor del subgrupo ${key} está vacío o supera 120 caracteres.`,
        { line },
      );
    }
    normalized.set(key, value);
  }
  return Object.fromEntries(normalized);
}

function normalizeDimensionPair(value, dimension, line, raterCount) {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new CalibrationError(
      `La dimensión ${dimension} debe ser un objeto.`,
      { line },
    );
  }
  const { automaticScore, humanBand, raterBands } = value;
  if (automaticScore === null || humanBand === null) return null;
  if (!finiteNumber(automaticScore) || automaticScore < 0 || automaticScore > 100) {
    throw new CalibrationError(
      `automaticScore de ${dimension} debe estar entre 0 y 100.`,
      { line },
    );
  }
  if (!Number.isInteger(humanBand) || humanBand < 0 || humanBand > 4) {
    throw new CalibrationError(
      `humanBand de ${dimension} debe ser un entero entre 0 y 4.`,
      { line },
    );
  }
  if (
    !Array.isArray(raterBands) ||
    raterBands.length !== raterCount ||
    raterBands.some(
      (band) => !Number.isInteger(band) || band < 0 || band > 4,
    )
  ) {
    throw new CalibrationError(
      `raterBands de ${dimension} debe contener una banda 0-4 por evaluador.`,
      { line },
    );
  }
  return { automaticScore, humanBand, raterBands };
}

export function normalizeCalibrationRecord(record, { line = null } = {}) {
  if (!record || typeof record !== 'object' || Array.isArray(record)) {
    throw new CalibrationError('Cada registro debe ser un objeto JSON.', {
      line,
    });
  }
  const assessmentId = requiredText(record.assessmentId, 'assessmentId', line);
  const speakerId = requiredText(record.speakerId, 'speakerId', line);
  if (record.labelSource !== 'human') {
    throw new CalibrationError(
      'labelSource debe ser human; no se aceptan etiquetas de otro modelo.',
      { code: 'HUMAN_LABEL_REQUIRED', line },
    );
  }
  const humanRatingStatus = requiredText(
    record.humanRatingStatus,
    'humanRatingStatus',
    line,
  );
  if (!VALID_HUMAN_RATING_STATUS.has(humanRatingStatus)) {
    throw new CalibrationError(
      'humanRatingStatus debe ser consensus o adjudicated.',
      { line },
    );
  }
  if (!Number.isInteger(record.raterCount) || record.raterCount < 2) {
    throw new CalibrationError(
      'raterCount debe confirmar al menos dos evaluadores humanos.',
      { line },
    );
  }
  const split = requiredText(record.split, 'split', line).toLowerCase();
  if (!VALID_SPLITS.has(split)) {
    throw new CalibrationError(
      'split debe ser train, validation o test.',
      { line },
    );
  }
  const mode = requiredText(record.mode, 'mode', line);
  if (!VALID_MODES.has(mode)) {
    throw new CalibrationError('mode debe ser reading o spontaneous.', {
      line,
    });
  }
  const cefr = requiredText(record.cefr, 'cefr', line).toUpperCase();
  if (!VALID_CEFR.has(cefr)) {
    throw new CalibrationError('cefr debe estar entre A1 y C2.', { line });
  }
  if (
    !record.dimensions ||
    typeof record.dimensions !== 'object' ||
    Array.isArray(record.dimensions)
  ) {
    throw new CalibrationError('dimensions es obligatorio.', { line });
  }

  const dimensions = Object.fromEntries(
    DIMENSION_IDS.map((dimension) => [
      dimension,
      normalizeDimensionPair(
        record.dimensions[dimension],
        dimension,
        line,
        record.raterCount,
      ),
    ]),
  );
  if (Object.values(dimensions).every((value) => value === null)) {
    throw new CalibrationError(
      'El registro no contiene pares automático-humano utilizables.',
      { line },
    );
  }

  return {
    assessmentId,
    speakerId,
    labelSource: 'human',
    humanRatingStatus,
    raterCount: record.raterCount,
    split,
    mode,
    cefr,
    subgroups: normalizeSubgroups(
      { ...record, mode, cefr },
      line,
    ),
    dimensions,
  };
}

function assertNoLeakage(records) {
  const speakerSplits = new Map();
  const assessmentIds = new Set();
  for (const record of records) {
    if (assessmentIds.has(record.assessmentId)) {
      throw new CalibrationError(
        `assessmentId duplicado: ${record.assessmentId}.`,
        { code: 'DUPLICATE_ASSESSMENT' },
      );
    }
    assessmentIds.add(record.assessmentId);
    const previous = speakerSplits.get(record.speakerId);
    if (previous && previous !== record.split) {
      throw new CalibrationError(
        `El hablante ${record.speakerId} aparece en ${previous} y ${record.split}.`,
        { code: 'SPEAKER_SPLIT_LEAKAGE' },
      );
    }
    speakerSplits.set(record.speakerId, record.split);
  }
}

function automaticBand(score) {
  return Math.max(0, Math.min(4, Math.round(score / 25)));
}

function dimensionMetrics(records, dimension) {
  const pairs = records
    .map((record) => record.dimensions[dimension])
    .filter(Boolean);
  const automaticScores = pairs.map((pair) => pair.automaticScore);
  const humanScores = pairs.map((pair) => pair.humanBand * 25);
  const predictedBands = pairs.map((pair) => automaticBand(pair.automaticScore));
  const humanBands = pairs.map((pair) => pair.humanBand);
  const firstRaterBands = pairs.map((pair) => pair.raterBands[0]);
  const secondRaterBands = pairs.map((pair) => pair.raterBands[1]);
  return {
    pairs: pairs.length,
    coverage:
      records.length === 0
        ? 0
        : roundMetric(pairs.length / records.length),
    mae: roundMetric(meanAbsoluteError(automaticScores, humanScores)),
    spearman: roundMetric(
      spearmanCorrelation(automaticScores, humanScores),
    ),
    quadraticWeightedKappa: roundMetric(
      quadraticWeightedKappa(predictedBands, humanBands),
    ),
    humanWeightedKappa: roundMetric(
      quadraticWeightedKappa(firstRaterBands, secondRaterBands),
    ),
  };
}

function overallPairs(records) {
  return records
    .map((record) => {
      if (
        DIMENSION_IDS.some(
          (dimension) => record.dimensions[dimension] === null,
        )
      ) {
        return null;
      }
      const weights = SCORE_PROFILES[record.mode].weights;
      const automaticScore = DIMENSION_IDS.reduce(
        (sum, dimension) =>
          sum +
          record.dimensions[dimension].automaticScore * weights[dimension],
        0,
      );
      const humanScore = DIMENSION_IDS.reduce(
        (sum, dimension) =>
          sum +
          record.dimensions[dimension].humanBand *
            25 *
            weights[dimension],
        0,
      );
      const firstRaterScore = DIMENSION_IDS.reduce(
        (sum, dimension) =>
          sum +
          record.dimensions[dimension].raterBands[0] *
            25 *
            weights[dimension],
        0,
      );
      const secondRaterScore = DIMENSION_IDS.reduce(
        (sum, dimension) =>
          sum +
          record.dimensions[dimension].raterBands[1] *
            25 *
            weights[dimension],
        0,
      );
      return {
        automaticScore,
        humanScore,
        firstRaterScore,
        secondRaterScore,
      };
    })
    .filter(Boolean);
}

function overallMetrics(records) {
  const pairs = overallPairs(records);
  const automaticScores = pairs.map((pair) => pair.automaticScore);
  const humanScores = pairs.map((pair) => pair.humanScore);
  return {
    pairs: pairs.length,
    coverage:
      records.length === 0
        ? 0
        : roundMetric(pairs.length / records.length),
    mae: roundMetric(meanAbsoluteError(automaticScores, humanScores)),
    spearman: roundMetric(
      spearmanCorrelation(automaticScores, humanScores),
    ),
    quadraticWeightedKappa: roundMetric(
      quadraticWeightedKappa(
        automaticScores.map(automaticBand),
        humanScores.map(automaticBand),
      ),
    ),
    humanWeightedKappa: roundMetric(
      quadraticWeightedKappa(
        pairs.map((pair) => automaticBand(pair.firstRaterScore)),
        pairs.map((pair) => automaticBand(pair.secondRaterScore)),
      ),
    ),
  };
}

function cohortSummary(records) {
  return {
    records: records.length,
    speakers: new Set(records.map((record) => record.speakerId)).size,
    overall: overallMetrics(records),
    dimensions: Object.fromEntries(
      DIMENSION_IDS.map((dimension) => [
        dimension,
        dimensionMetrics(records, dimension),
      ]),
    ),
  };
}

function subgroupMaeGaps(subgroups) {
  const metrics = ['overall', ...DIMENSION_IDS];
  return Object.fromEntries(
    metrics.map((metric) => {
      const comparisons = [];
      for (const [subgroup, values] of Object.entries(subgroups)) {
        const cohorts = Object.entries(values)
          .map(([value, summary]) => ({
            value,
            mae:
              metric === 'overall'
                ? summary.overall.mae
                : summary.dimensions[metric].mae,
          }))
          .filter((entry) => entry.mae !== null);
        if (cohorts.length < 2) continue;
        const sorted = cohorts.sort((left, right) => left.mae - right.mae);
        comparisons.push({
          subgroup,
          best: sorted[0],
          worst: sorted[sorted.length - 1],
          gap: roundMetric(
            sorted[sorted.length - 1].mae - sorted[0].mae,
          ),
        });
      }
      comparisons.sort((left, right) => right.gap - left.gap);
      return [
        metric,
        {
          maximumGap: comparisons[0]?.gap ?? null,
          worstComparison: comparisons[0] ?? null,
          comparisons,
        },
      ];
    }),
  );
}

function subgroupSummaries(records) {
  const groups = new Map();
  for (const record of records) {
    for (const [key, value] of Object.entries(record.subgroups)) {
      if (!groups.has(key)) groups.set(key, new Map());
      const values = groups.get(key);
      if (!values.has(value)) values.set(value, []);
      values.get(value).push(record);
    }
  }
  return Object.fromEntries(
    [...groups.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, values]) => [
        key,
        Object.fromEntries(
          [...values.entries()]
            .sort(([left], [right]) => left.localeCompare(right))
            .map(([value, groupRecords]) => [
              value,
              cohortSummary(groupRecords),
            ]),
        ),
      ]),
  );
}

export function analyzeCalibrationRecords(records) {
  if (!Array.isArray(records) || records.length === 0) {
    throw new CalibrationError('El conjunto de calibración está vacío.', {
      code: 'EMPTY_CALIBRATION_DATASET',
    });
  }
  const normalized = records.map((record, index) =>
    normalizeCalibrationRecord(record, { line: index + 1 }),
  );
  assertNoLeakage(normalized);

  const splits = Object.fromEntries(
    [...VALID_SPLITS].map((split) => [
      split,
      cohortSummary(normalized.filter((record) => record.split === split)),
    ]),
  );
  const subgroups = subgroupSummaries(normalized);
  const fullCohort = cohortSummary(normalized);
  return {
    schemaVersion: CALIBRATION_REPORT_VERSION,
    generatedAt: new Date().toISOString(),
    records: normalized.length,
    speakers: new Set(normalized.map((record) => record.speakerId)).size,
    splits,
    overall: fullCohort.overall,
    dimensions: fullCohort.dimensions,
    subgroups,
    fairness: {
      maeGaps: subgroupMaeGaps(subgroups),
    },
    methodology: {
      humanScale: 'bands 0-4 converted to 0-100 for MAE and Spearman',
      predictedBand:
        'automaticScore rounded to nearest 25-point band for kappa',
      kappa: 'quadratic weighted Cohen kappa',
      humanKappa:
        'quadratic weighted Cohen kappa between the first two human raters',
    },
  };
}

export async function readCalibrationJsonl(filePath) {
  const records = [];
  const input = createReadStream(filePath, { encoding: 'utf8' });
  const lines = readline.createInterface({
    input,
    crlfDelay: Infinity,
  });
  let lineNumber = 0;
  try {
    for await (const rawLine of lines) {
      lineNumber++;
      const line = rawLine.trim();
      if (!line) continue;
      let record;
      try {
        record = JSON.parse(line);
      } catch {
        throw new CalibrationError('JSON inválido.', { line: lineNumber });
      }
      records.push(normalizeCalibrationRecord(record, { line: lineNumber }));
    }
  } catch (error) {
    input.destroy();
    throw error;
  }
  return records;
}
