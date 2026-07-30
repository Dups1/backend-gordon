export const WHISPER_LITERAL_POLICY_VERSION = 'whisper-literal-v1';

const literalPrompts = {
  en: [
    'Verbatim, disfluent speech.',
    'Preserve fillers, false starts, repeated words, incomplete phrases,',
    'stutters, and ungrammatical wording exactly as spoken.',
    'Do not rewrite, summarize, or correct the speaker.',
  ].join(' '),
  es: [
    'Transcripción literal y espontánea.',
    'Conserva las muletillas, falsos comienzos, palabras repetidas, frases',
    'incompletas, tartamudeos y errores gramaticales exactamente como se',
    'dijeron. No reescribas, resumas ni corrijas al hablante.',
  ].join(' '),
};

export function whisperLiteralPrompt(language) {
  const normalized =
    typeof language === 'string' ? language.trim().toLowerCase() : '';
  return literalPrompts[normalized] ?? null;
}

export function whisperLiteralOptions(language) {
  const normalized =
    typeof language === 'string' ? language.trim().toLowerCase() : '';
  const prompt = whisperLiteralPrompt(normalized);
  const options = {
    response_format: 'verbose_json',
    timestamp_granularities: ['word', 'segment'],
    temperature: 0,
  };
  if (normalized) options.language = normalized;
  if (prompt) options.prompt = prompt;
  return options;
}
