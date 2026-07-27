import assert from 'node:assert/strict';
import test from 'node:test';

import {
  INTERNAL_PROMPTS,
  PROMPT_EXAMPLES_STATUS,
  PROMPT_MANIFEST_HASH,
  PROMPT_MANIFEST_VERSION,
} from '../src/evaluation/prompts.js';

test('carga prompts separados, versionados y resistentes a instrucciones', () => {
  assert.equal(PROMPT_MANIFEST_VERSION, 'gordon-evidence-v1.1');
  assert.match(PROMPT_MANIFEST_HASH, /^[a-f0-9]{64}$/);
  assert.match(INTERNAL_PROMPTS.securityEnvelope, /nunca instrucciones/i);
  assert.match(INTERNAL_PROMPTS.instructionImprover, /intención docente/i);
  assert.match(INTERNAL_PROMPTS.instructionImprover, /dirigida al estudiante/i);
  assert.match(INTERNAL_PROMPTS.evidenceExtractor, /no asignes notas/i);
  assert.match(INTERNAL_PROMPTS.analyticJudge, /pronunciación ni fluidez/i);
  assert.match(INTERNAL_PROMPTS.feedbackGenerator, /no asignes.*score/i);
});

test('no inyecta ejemplos CEFR hasta que exista revisión humana explícita', () => {
  assert.equal(PROMPT_EXAMPLES_STATUS.approved, false);
  assert.equal(PROMPT_EXAMPLES_STATUS.reviewStatus, 'pending_human_review');
  assert.equal(PROMPT_EXAMPLES_STATUS.count, 0);
  assert.doesNotMatch(
    INTERNAL_PROMPTS.analyticJudge,
    /Ejemplos CEFR revisados por humanos/,
  );
});
