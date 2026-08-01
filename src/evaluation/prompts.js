import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const promptDirectory = new URL('../../prompts/v1/', import.meta.url);

function readText(name) {
  const text = readFileSync(new URL(name, promptDirectory), 'utf8').trim();
  if (!text) {
    throw new Error(`El prompt versionado ${name} está vacío.`);
  }
  return text;
}

function readJson(name) {
  return JSON.parse(readFileSync(new URL(name, promptDirectory), 'utf8'));
}

const manifest = readJson('manifest.json');
const examplesManifest = readJson(manifest.examples);

export const PROMPT_MANIFEST_VERSION = manifest.promptVersion;

export const PROMPT_EXAMPLES_STATUS = Object.freeze({
  version: examplesManifest.version,
  reviewStatus: examplesManifest.reviewStatus,
  approved:
    examplesManifest.reviewStatus === 'approved_human' &&
    Array.isArray(examplesManifest.reviewedBy) &&
    examplesManifest.reviewedBy.length > 0,
  count: Array.isArray(examplesManifest.examples)
    ? examplesManifest.examples.length
    : 0,
});

function approvedExamplesFor(role) {
  if (!PROMPT_EXAMPLES_STATUS.approved) return '';
  const examples = examplesManifest.examples.filter(
    (example) => example?.role === role,
  );
  return examples.length
    ? `\n\nEjemplos CEFR revisados por humanos:\n${JSON.stringify(examples)}`
    : '';
}

function prompt(role) {
  const file = manifest.files?.[role];
  if (typeof file !== 'string' || !file.trim()) {
    throw new Error(`Falta el archivo del prompt ${role}.`);
  }
  return `${readText(file)}${approvedExamplesFor(role)}`;
}

export const INTERNAL_PROMPTS = Object.freeze({
  securityEnvelope: prompt('securityEnvelope'),
  instructionImprover: prompt('instructionImprover'),
  phoneticLiteralizer: prompt('phoneticLiteralizer'),
  expectedPhonetic: prompt('expectedPhonetic'),
  rubricCompiler: prompt('rubricCompiler'),
  evidenceExtractor: prompt('evidenceExtractor'),
  analyticJudge: prompt('analyticJudge'),
  holisticJudge: prompt('holisticJudge'),
  adjudicator: prompt('adjudicator'),
  feedbackGenerator: prompt('feedbackGenerator'),
});

export const PROMPT_MANIFEST_HASH = createHash('sha256')
  .update(
    JSON.stringify({
      manifest,
      prompts: INTERNAL_PROMPTS,
      examples: PROMPT_EXAMPLES_STATUS.approved
        ? examplesManifest
        : {
            version: examplesManifest.version,
            reviewStatus: examplesManifest.reviewStatus,
          },
    }),
  )
  .digest('hex');
