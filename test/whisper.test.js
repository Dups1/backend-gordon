import assert from 'node:assert/strict';
import test from 'node:test';

import {
  WHISPER_LITERAL_POLICY_VERSION,
  whisperLiteralOptions,
  whisperLiteralPrompt,
} from '../src/whisper.js';

test('configura Whisper con temperatura cero y política literal en inglés', () => {
  const options = whisperLiteralOptions('EN');

  assert.equal(WHISPER_LITERAL_POLICY_VERSION, 'whisper-literal-v1');
  assert.equal(options.temperature, 0);
  assert.equal(options.language, 'en');
  assert.equal(options.response_format, 'verbose_json');
  assert.deepEqual(options.timestamp_granularities, ['word', 'segment']);
  assert.match(options.prompt, /Preserve fillers/);
  assert.match(options.prompt, /ungrammatical wording exactly as spoken/);
  assert.match(options.prompt, /Do not rewrite, summarize, or correct/);
});

test('usa un prompt literal en español sin inventarlo para otros idiomas', () => {
  const spanish = whisperLiteralOptions('es');
  const unknown = whisperLiteralOptions('fr');

  assert.equal(spanish.prompt, whisperLiteralPrompt('es'));
  assert.match(spanish.prompt, /errores gramaticales exactamente/);
  assert.equal(unknown.language, 'fr');
  assert.equal(unknown.prompt, undefined);
});
