import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  analyzeCalibrationRecords,
  meanAbsoluteError,
  quadraticWeightedKappa,
  readCalibrationJsonl,
  spearmanCorrelation,
} from '../src/evaluation/calibration.js';

function record({
  assessmentId,
  speakerId,
  split,
  automaticScore,
  humanBand,
  mode = 'reading',
  cefr = 'B1',
  l1 = 'es',
}) {
  return {
    assessmentId,
    speakerId,
    labelSource: 'human',
    humanRatingStatus: 'consensus',
    raterCount: 2,
    split,
    mode,
    cefr,
    subgroups: { l1, device: 'mobile' },
    dimensions: {
      pronunciation: {
        automaticScore,
        humanBand,
        raterBands: [humanBand, humanBand],
      },
      fluency: {
        automaticScore: Math.max(0, automaticScore - 5),
        humanBand,
        raterBands: [humanBand, humanBand],
      },
    },
  };
}

test('calcula MAE, Spearman con empates y kappa cuadrático', () => {
  assert.equal(meanAbsoluteError([0, 25, 50], [0, 50, 50]), 25 / 3);
  assert.equal(spearmanCorrelation([10, 20, 30], [25, 50, 75]), 1);
  assert.equal(spearmanCorrelation([10, 20, 30], [75, 50, 25]), -1);
  assert.equal(quadraticWeightedKappa([0, 1, 2, 3, 4], [0, 1, 2, 3, 4]), 1);
  assert.equal(spearmanCorrelation([10, 10], [25, 50]), null);
});

test('resume métricas por dimensión, split y subgrupo', () => {
  const report = analyzeCalibrationRecords([
    record({
      assessmentId: 'a-1',
      speakerId: 'speaker-1',
      split: 'train',
      automaticScore: 70,
      humanBand: 3,
    }),
    record({
      assessmentId: 'a-2',
      speakerId: 'speaker-2',
      split: 'validation',
      automaticScore: 50,
      humanBand: 2,
      l1: 'fr',
    }),
    record({
      assessmentId: 'a-3',
      speakerId: 'speaker-3',
      split: 'test',
      automaticScore: 90,
      humanBand: 4,
      mode: 'spontaneous',
      cefr: 'B2',
    }),
  ]);

  assert.equal(report.records, 3);
  assert.equal(report.speakers, 3);
  assert.equal(report.dimensions.pronunciation.pairs, 3);
  assert.equal(report.dimensions.pronunciation.mae, 5);
  assert.equal(report.dimensions.pronunciation.spearman, 1);
  assert.equal(report.dimensions.pronunciation.quadraticWeightedKappa, 1);
  assert.equal(report.dimensions.pronunciation.humanWeightedKappa, 1);
  assert.equal(report.dimensions.grammar.pairs, 0);
  assert.equal(report.dimensions.grammar.mae, null);
  assert.equal(report.splits.train.records, 1);
  assert.equal(report.subgroups.l1.es.records, 2);
  assert.equal(report.subgroups.mode.spontaneous.records, 1);
  assert.equal(report.overall.pairs, 0);
  assert.equal(report.fairness.maeGaps.pronunciation.maximumGap, 7.5);
});

test('mode y CEFR del registro prevalecen sobre subgrupos reservados', () => {
  const input = record({
    assessmentId: 'a-1',
    speakerId: 'speaker-1',
    split: 'train',
    automaticScore: 75,
    humanBand: 3,
  });
  input.subgroups.mode = 'altered';
  input.subgroups.cefr = 'C2';
  const report = analyzeCalibrationRecords([input]);

  assert.equal(report.subgroups.mode.reading.records, 1);
  assert.equal(report.subgroups.mode.altered, undefined);
  assert.equal(report.subgroups.cefr.B1.records, 1);
});

test('rechaza fuga de un mismo hablante entre splits', () => {
  assert.throws(
    () =>
      analyzeCalibrationRecords([
        record({
          assessmentId: 'a-1',
          speakerId: 'speaker-1',
          split: 'train',
          automaticScore: 70,
          humanBand: 3,
        }),
        record({
          assessmentId: 'a-2',
          speakerId: 'speaker-1',
          split: 'test',
          automaticScore: 75,
          humanBand: 3,
        }),
      ]),
    (error) =>
      error.code === 'SPEAKER_SPLIT_LEAKAGE' &&
      /speaker-1.*train.*test/.test(error.message),
  );
});

test('rechaza etiquetas automáticas como verdad de calibración', () => {
  const input = record({
    assessmentId: 'a-auto',
    speakerId: 'speaker-auto',
    split: 'train',
    automaticScore: 75,
    humanBand: 3,
  });
  input.labelSource = 'gpt-oss';
  assert.throws(
    () => analyzeCalibrationRecords([input]),
    (error) => error.code === 'HUMAN_LABEL_REQUIRED',
  );
});

test('lee JSONL, omite líneas vacías y reporta la línea inválida', async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), 'gordon-calibration-'));
  const validPath = path.join(directory, 'valid.jsonl');
  const invalidPath = path.join(directory, 'invalid.jsonl');
  const validRecords = [
    record({
      assessmentId: 'a-1',
      speakerId: 'speaker-1',
      split: 'train',
      automaticScore: 75,
      humanBand: 3,
    }),
    record({
      assessmentId: 'a-2',
      speakerId: 'speaker-2',
      split: 'test',
      automaticScore: 50,
      humanBand: 2,
    }),
  ];
  await Promise.all([
    writeFile(
      validPath,
      `${JSON.stringify(validRecords[0])}\n\n${JSON.stringify(validRecords[1])}\n`,
    ),
    writeFile(invalidPath, `${JSON.stringify(validRecords[0])}\n{no-json}\n`),
  ]);

  try {
    const parsed = await readCalibrationJsonl(validPath);
    assert.equal(parsed.length, 2);
    assert.equal(analyzeCalibrationRecords(parsed).records, 2);
    await assert.rejects(
      readCalibrationJsonl(invalidPath),
      /Línea 2: JSON inválido/,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
