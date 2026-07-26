import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';

import {
  cleanAudioChunks,
  splitNormalizedAudio,
} from './audio.js';
import {
  EvaluationError,
  finiteScore,
  normalizeTokens,
  tokenErrorRate,
} from './domain.js';
import { withCircuitBreaker } from './resilience.js';

function finite(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function strictProviderScore(value, field) {
  if (value === null || value === undefined) return null;
  const score = finiteScore(value);
  if (score === null) {
    throw new EvaluationError(
      502,
      `Azure devolvió un valor inválido para ${field}.`,
      'INVALID_PROVIDER_SCORE',
      { field },
    );
  }
  return score;
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function azureSeconds(value) {
  const number = finite(value);
  return number === null ? null : number / 10_000_000;
}

function assessmentOf(value) {
  return value?.PronunciationAssessment ?? value ?? {};
}

function normalizePhoneme(phoneme) {
  const assessment = assessmentOf(phoneme);
  return {
    phoneme: typeof phoneme?.Phoneme === 'string' ? phoneme.Phoneme : '',
    start: azureSeconds(phoneme?.Offset),
    duration: azureSeconds(phoneme?.Duration),
    accuracyScore: strictProviderScore(
      assessment.AccuracyScore,
      'phoneme.accuracyScore',
    ),
    nBestPhonemes: Array.isArray(assessment.NBestPhonemes)
      ? assessment.NBestPhonemes.map((candidate) => ({
          phoneme:
            typeof candidate?.Phoneme === 'string' ? candidate.Phoneme : '',
          score: strictProviderScore(
            candidate?.Score,
            'phoneme.nBestScore',
          ),
        }))
      : [],
  };
}

function normalizeSyllable(syllable) {
  const assessment = assessmentOf(syllable);
  return {
    syllable: typeof syllable?.Syllable === 'string' ? syllable.Syllable : '',
    grapheme: typeof syllable?.Grapheme === 'string' ? syllable.Grapheme : '',
    start: azureSeconds(syllable?.Offset),
    duration: azureSeconds(syllable?.Duration),
    accuracyScore: strictProviderScore(
      assessment.AccuracyScore,
      'syllable.accuracyScore',
    ),
  };
}

function normalizeWord(word) {
  const assessment = assessmentOf(word);
  return {
    word: typeof word?.Word === 'string' ? word.Word : '',
    start: azureSeconds(word?.Offset),
    duration: azureSeconds(word?.Duration),
    accuracyScore: strictProviderScore(
      assessment.AccuracyScore,
      'word.accuracyScore',
    ),
    errorType:
      typeof assessment.ErrorType === 'string' ? assessment.ErrorType : null,
    feedback: assessment.Feedback ?? null,
    syllables: Array.isArray(word?.Syllables)
      ? word.Syllables.map(normalizeSyllable)
      : [],
    phonemes: Array.isArray(word?.Phonemes)
      ? word.Phonemes.map(normalizePhoneme)
      : [],
  };
}

export function normalizeAzureResponse(raw, { mode, locale, requestId } = {}) {
  const candidates = Array.isArray(raw?.NBest) ? raw.NBest : [];
  const best = candidates[0] ?? null;
  if (!best) {
    throw new EvaluationError(
      502,
      'Azure no devolvió candidatos de pronunciación.',
      'AZURE_NO_ASSESSMENT',
    );
  }
  const assessment = assessmentOf(best);
  return {
    provider: 'azure-speech',
    mode,
    locale,
    recognitionStatus:
      typeof raw?.RecognitionStatus === 'string' ? raw.RecognitionStatus : null,
    requestId: requestId ?? null,
    offset: azureSeconds(raw?.Offset),
    duration: azureSeconds(raw?.Duration),
    displayText:
      typeof raw?.DisplayText === 'string'
        ? raw.DisplayText
        : typeof best?.Display === 'string'
          ? best.Display
          : '',
    lexicalText: typeof best?.Lexical === 'string' ? best.Lexical : '',
    itnText: typeof best?.ITN === 'string' ? best.ITN : '',
    maskedItnText:
      typeof best?.MaskedITN === 'string' ? best.MaskedITN : '',
    confidence: finite(best?.Confidence),
    pronunciationScore: strictProviderScore(
      assessment.PronScore,
      'pronunciationScore',
    ),
    accuracyScore: strictProviderScore(
      assessment.AccuracyScore,
      'accuracyScore',
    ),
    fluencyScore: strictProviderScore(
      assessment.FluencyScore,
      'fluencyScore',
    ),
    completenessScore:
      mode === 'reading'
        ? strictProviderScore(
            assessment.CompletenessScore,
            'completenessScore',
          )
        : null,
    prosodyScore: strictProviderScore(
      assessment.ProsodyScore,
      'prosodyScore',
    ),
    feedback: assessment.Feedback ?? null,
    words: Array.isArray(best?.Words) ? best.Words.map(normalizeWord) : [],
    candidates: candidates.map((candidate) => ({
      confidence: finite(candidate?.Confidence),
      lexicalText:
        typeof candidate?.Lexical === 'string' ? candidate.Lexical : '',
      displayText:
        typeof candidate?.Display === 'string' ? candidate.Display : '',
      assessment: assessmentOf(candidate),
      words: Array.isArray(candidate?.Words)
        ? candidate.Words.map(normalizeWord)
        : [],
    })),
    rawResponse: raw,
  };
}

function normalizeWhisperChunk(raw, offset) {
  const words = Array.isArray(raw?.words)
    ? raw.words
        .map((word) => ({
          word: typeof word?.word === 'string' ? word.word.trim() : '',
          start: finite(word?.start),
          end: finite(word?.end),
        }))
        .filter(
          (word) =>
            word.word &&
            word.start !== null &&
            word.end !== null &&
            word.end >= word.start,
        )
        .map((word) => ({
          ...word,
          start: word.start + offset,
          end: word.end + offset,
        }))
    : [];
  const segments = Array.isArray(raw?.segments)
    ? raw.segments
        .map((segment) => ({
          id: Number.isInteger(segment?.id) ? segment.id : null,
          text: typeof segment?.text === 'string' ? segment.text.trim() : '',
          start: finite(segment?.start),
          end: finite(segment?.end),
          avgLogprob: finite(segment?.avg_logprob),
          compressionRatio: finite(segment?.compression_ratio),
          noSpeechProb: finite(segment?.no_speech_prob),
        }))
        .filter(
          (segment) => segment.start !== null && segment.end !== null,
        )
        .map((segment) => ({
          ...segment,
          start: segment.start + offset,
          end: segment.end + offset,
        }))
    : [];
  return {
    text: typeof raw?.text === 'string' ? raw.text.trim() : '',
    language: typeof raw?.language === 'string' ? raw.language : null,
    duration: finite(raw?.duration),
    words,
    segments,
    raw,
  };
}

function stitchWhisperChunks(chunks, overlapSeconds) {
  const words = [];
  const segments = [];
  for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
    const chunk = chunks[chunkIndex];
    const ownedStart =
      chunkIndex === 0 ? chunk.start : chunk.start + overlapSeconds / 2;
    for (const word of chunk.result.words) {
      const center = (word.start + word.end) / 2;
      if (center < ownedStart) continue;
      const duplicate = words
        .slice(-6)
        .some(
          (existing) =>
            normalizeTokens(existing.word)[0] === normalizeTokens(word.word)[0] &&
            Math.abs(existing.start - word.start) < overlapSeconds,
        );
      if (!duplicate) words.push(word);
    }
    for (const segment of chunk.result.segments) {
      const center = (segment.start + segment.end) / 2;
      if (center >= ownedStart) segments.push(segment);
    }
  }
  const text = words.length
    ? words.map((word) => word.word).join(' ').replace(/\s+([,.!?;:])/g, '$1')
    : chunks.map((chunk) => chunk.result.text).filter(Boolean).join(' ');
  return {
    text,
    words,
    segments,
    language: chunks.find((chunk) => chunk.result.language)?.result.language ?? null,
  };
}

function whisperHint(rubric) {
  const hints = Array.isArray(rubric.spec.vocabularyHints)
    ? rubric.spec.vocabularyHints
    : [];
  if (!hints.length) return null;
  return `Vocabulary and proper nouns: ${hints.join(', ')}`.slice(0, 800);
}

export async function transcribeWhisper({
  client,
  normalizedPath,
  durationSeconds,
  rubric,
  quality,
  model = 'whisper-large-v3',
}) {
  const overlapSeconds = 2;
  const audioChunks = await splitNormalizedAudio(normalizedPath, {
    durationSeconds,
    chunkSeconds: 28,
    overlapSeconds,
    preferredSilences: quality?.activity?.silenceIntervals ?? [],
  });
  try {
    const results = [];
    for (const chunk of audioChunks) {
      const hint = whisperHint(rubric);
      let raw;
      let lastError;
      for (let attempt = 0; attempt < 3; attempt++) {
        const options = {
          file: createReadStream(chunk.path),
          model,
          response_format: 'verbose_json',
          timestamp_granularities: ['word', 'segment'],
          temperature: 0,
          language: rubric.spec.asrLanguage,
        };
        if (hint) options.prompt = hint;
        try {
          raw = await withCircuitBreaker('groq-whisper', () =>
            client.audio.transcriptions.create(options, {
              timeout: 90_000,
            }),
          );
          break;
        } catch (error) {
          lastError = error;
          const retryable =
            error?.status === 429 ||
            (Number.isInteger(error?.status) && error.status >= 500);
          if (!retryable || attempt === 2) break;
          await delay(250 * 3 ** attempt + Math.floor(Math.random() * 150));
        }
      }
      if (!raw) {
        throw new EvaluationError(
          lastError?.status === 429 ? 429 : 502,
          lastError?.status === 429
            ? 'Groq alcanzó temporalmente su límite.'
            : 'Whisper no pudo transcribir el audio.',
          lastError?.status === 429
            ? 'GROQ_RATE_LIMIT'
            : 'WHISPER_PROVIDER_ERROR',
        );
      }
      results.push({
        ...chunk,
        result: normalizeWhisperChunk(raw, chunk.start),
      });
    }
    const stitched = stitchWhisperChunks(results, overlapSeconds);
    return {
      provider: 'groq',
      model,
      language: stitched.language,
      text: stitched.text,
      words: stitched.words,
      segments: stitched.segments,
      chunks: results.map((chunk) => ({
        start: chunk.start,
        end: chunk.end,
        text: chunk.result.text,
        language: chunk.result.language,
        raw: chunk.result.raw,
      })),
      chunking:
        results.length > 1
          ? {
              method: 'vad-aware-window-v1',
              activityMethod: quality?.activity?.method ?? 'unavailable',
              chunkSeconds: 28,
              overlapSeconds,
              chunkCount: results.length,
            }
          : { method: 'single', chunkCount: 1 },
    };
  } finally {
    await cleanAudioChunks(audioChunks);
  }
}

function alignReference(referenceText, recognizedWords) {
  const reference = normalizeTokens(referenceText);
  const recognized = recognizedWords.map((word) => normalizeTokens(word.word)[0] ?? '');
  const rows = Array.from({ length: reference.length + 1 }, () =>
    Array(recognized.length + 1).fill(0),
  );
  for (let i = 0; i <= reference.length; i++) rows[i][0] = i;
  for (let j = 0; j <= recognized.length; j++) rows[0][j] = j;
  for (let i = 1; i <= reference.length; i++) {
    for (let j = 1; j <= recognized.length; j++) {
      rows[i][j] = Math.min(
        rows[i - 1][j] + 1,
        rows[i][j - 1] + 1,
        rows[i - 1][j - 1] + (reference[i - 1] === recognized[j - 1] ? 0 : 1),
      );
    }
  }
  const mapping = new Map();
  let i = reference.length;
  let j = recognized.length;
  while (i > 0 || j > 0) {
    if (
      i > 0 &&
      j > 0 &&
      rows[i][j] ===
        rows[i - 1][j - 1] + (reference[i - 1] === recognized[j - 1] ? 0 : 1)
    ) {
      mapping.set(j - 1, i - 1);
      i--;
      j--;
    } else if (i > 0 && rows[i][j] === rows[i - 1][j] + 1) {
      i--;
    } else {
      j--;
    }
  }
  return { reference, mapping };
}

function referenceSlices(referenceText, whisperWords, chunks, durationSeconds) {
  const { reference, mapping } = alignReference(referenceText, whisperWords);
  let previousEnd = 0;
  return chunks.map((chunk, chunkIndex) => {
    const recognizedIndexes = whisperWords
      .map((word, index) => ({ word, index }))
      .filter(({ word }) => {
        const center = (word.start + word.end) / 2;
        return center >= chunk.start && center < chunk.end;
      })
      .map(({ index }) => index);
    const mapped = recognizedIndexes
      .map((index) => mapping.get(index))
      .filter(Number.isInteger);
    const proportionalStart = Math.floor(
      (chunk.start / durationSeconds) * reference.length,
    );
    const proportionalEnd = Math.ceil(
      (chunk.end / durationSeconds) * reference.length,
    );
    const start = Math.max(
      previousEnd,
      mapped.length ? Math.min(...mapped) : proportionalStart,
    );
    const end =
      chunkIndex === chunks.length - 1
        ? reference.length
        : Math.max(
            start + 1,
            mapped.length ? Math.max(...mapped) + 1 : proportionalEnd,
          );
    previousEnd = Math.min(reference.length, end);
    return reference.slice(start, previousEnd).join(' ');
  });
}

function weightedAverage(items, valueSelector, weightSelector) {
  const valid = items
    .map((item) => ({
      value: finiteScore(valueSelector(item)),
      weight: Math.max(0.001, finite(weightSelector(item)) ?? 1),
    }))
    .filter((item) => item.value !== null);
  if (!valid.length) return null;
  const weight = valid.reduce((sum, item) => sum + item.weight, 0);
  return valid.reduce((sum, item) => sum + item.value * item.weight, 0) / weight;
}

function aggregateAzureChunks(chunks, { mode, locale, referenceText }) {
  const words = chunks.flatMap((chunk) =>
    chunk.result.words.map((word) => ({
      ...word,
      start: word.start === null ? null : word.start + chunk.start,
    })),
  );
  const phonemes = words.flatMap((word) => word.phonemes);
  const eligibleDuration = chunks.reduce(
    (sum, chunk) => sum + (chunk.end - chunk.start),
    0,
  );
  const accuracyScore =
    weightedAverage(
      phonemes,
      (phoneme) => phoneme.accuracyScore,
      (phoneme) => phoneme.duration,
    ) ??
    weightedAverage(
      words,
      (word) => word.accuracyScore,
      (word) => word.duration,
    );
  const fluencyScore = weightedAverage(
    chunks,
    (chunk) => chunk.result.fluencyScore,
    (chunk) => chunk.end - chunk.start,
  );
  const prosodyScore = weightedAverage(
    chunks,
    (chunk) => chunk.result.prosodyScore,
    (chunk) => chunk.end - chunk.start,
  );
  const referenceWords = normalizeTokens(referenceText).length;
  const pronouncedWords = words.filter(
    (word) => !['Omission'].includes(word.errorType),
  ).length;
  const completenessScore =
    mode === 'reading' && referenceWords
      ? Math.min(100, (pronouncedWords / referenceWords) * 100)
      : null;
  return {
    provider: 'azure-speech',
    mode,
    locale,
    recognitionStatus: chunks.every(
      (chunk) => chunk.result.recognitionStatus === 'Success',
    )
      ? 'Success'
      : 'Partial',
    displayText: chunks.map((chunk) => chunk.result.displayText).join(' ').trim(),
    lexicalText: chunks.map((chunk) => chunk.result.lexicalText).join(' ').trim(),
    pronunciationScore: accuracyScore,
    accuracyScore,
    fluencyScore,
    completenessScore,
    prosodyScore,
    words,
    candidates: [],
    rawResponse: chunks.map((chunk) => ({
      start: chunk.start,
      end: chunk.end,
      referenceText: chunk.referenceText,
      response: chunk.result.rawResponse,
    })),
    aggregation: {
      method: 'eligible-phoneme-word-duration-v1',
      chunkCount: chunks.length,
      eligibleDuration,
      referenceWordCount: referenceWords,
      pronouncedWordCount: pronouncedWords,
    },
  };
}

export function createContinuousPronunciationConfig(sdk, locale) {
  const pronunciation = new sdk.PronunciationAssessmentConfig(
    '',
    sdk.PronunciationAssessmentGradingSystem.HundredMark,
    sdk.PronunciationAssessmentGranularity.Phoneme,
    false,
  );
  pronunciation.enableProsodyAssessment = locale === 'en-US';
  pronunciation.phonemeAlphabet = 'IPA';
  pronunciation.nbestPhonemeCount = 5;
  return pronunciation;
}

async function recognizeContinuousWithKey({
  key,
  region,
  normalizedPath,
  locale,
  timeoutMs = 7 * 60 * 1000,
}) {
  let module;
  try {
    module = await import('microsoft-cognitiveservices-speech-sdk');
  } catch {
    throw new EvaluationError(
      503,
      'Azure Speech SDK no está instalado.',
      'AZURE_SDK_UNAVAILABLE',
    );
  }
  const sdk = module.default ?? module;
  const speechConfig = sdk.SpeechConfig.fromSubscription(key, region);
  speechConfig.speechRecognitionLanguage = locale;
  speechConfig.outputFormat = sdk.OutputFormat.Detailed;
  const audioConfig = sdk.AudioConfig.fromWavFileInput(
    await readFile(normalizedPath),
  );
  const recognizer = new sdk.SpeechRecognizer(speechConfig, audioConfig);
  const pronunciation = createContinuousPronunciationConfig(sdk, locale);
  pronunciation.applyTo(recognizer);
  return new Promise((resolve, reject) => {
    const rawResults = [];
    let settled = false;
    const timer = setTimeout(() => {
      recognizer.stopContinuousRecognitionAsync(
        () =>
          finish(
            new EvaluationError(
              504,
              'Azure continuo agotó el tiempo de evaluación.',
              'AZURE_CONTINUOUS_TIMEOUT',
            ),
          ),
        () =>
          finish(
            new EvaluationError(
              504,
              'Azure continuo agotó el tiempo de evaluación.',
              'AZURE_CONTINUOUS_TIMEOUT',
            ),
          ),
      );
    }, timeoutMs);
    const finish = (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      recognizer.close();
      if (error) reject(error);
      else resolve(rawResults);
    };
    recognizer.recognized = (_sender, event) => {
      if (event.result.reason !== sdk.ResultReason.RecognizedSpeech) return;
      const json = event.result.properties.getProperty(
        sdk.PropertyId.SpeechServiceResponse_JsonResult,
      );
      try {
        rawResults.push(JSON.parse(json));
      } catch {
        // A result without valid detailed JSON cannot be scored.
      }
    };
    recognizer.canceled = (_sender, event) => {
      finish(
        new EvaluationError(
          502,
          'Azure canceló la evaluación continua.',
          event.errorCode === 1
            ? 'AZURE_AUTHENTICATION'
            : 'AZURE_CONTINUOUS_ERROR',
          { cancellationReason: event.reason },
        ),
      );
    };
    recognizer.sessionStopped = () => finish();
    recognizer.startContinuousRecognitionAsync(
      undefined,
      (error) =>
        finish(
          new EvaluationError(
            502,
            'No fue posible iniciar Azure continuo.',
            'AZURE_CONTINUOUS_ERROR',
            { error: String(error) },
          ),
        ),
    );
  });
}

export async function evaluateAzureV2({
  normalizedPath,
  durationSeconds,
  rubric,
  whisper,
  quality,
  azureConfig,
  evaluateRest,
  evaluateContinuous = recognizeContinuousWithKey,
}) {
  if (!azureConfig) {
    throw new EvaluationError(
      503,
      'Azure Speech no está configurado.',
      'AZURE_NOT_CONFIGURED',
    );
  }
  if (rubric.spec.mode === 'spontaneous') {
    const keys = [azureConfig.clavePrimaria, azureConfig.claveSecundaria].filter(
      (value, index, values) => value && values.indexOf(value) === index,
    );
    let lastError;
    for (const key of keys) {
      try {
        const rawResults = await withCircuitBreaker('azure-speech', () =>
          evaluateContinuous({
            key,
            region: azureConfig.region,
            normalizedPath,
            locale: rubric.spec.targetLocale,
          }),
        );
        const chunks = rawResults.map((raw, index) => ({
          start: azureSeconds(raw.Offset) ?? 0,
          end:
            (azureSeconds(raw.Offset) ?? 0) +
            (azureSeconds(raw.Duration) ?? durationSeconds / rawResults.length),
          referenceText: '',
          result: normalizeAzureResponse(raw, {
            mode: 'spontaneous',
            locale: rubric.spec.targetLocale,
            requestId: `continuous-${index}`,
          }),
        }));
        return aggregateAzureChunks(chunks, {
          mode: 'spontaneous',
          locale: rubric.spec.targetLocale,
          referenceText: '',
        });
      } catch (error) {
        lastError = error;
        if (error?.code !== 'AZURE_AUTHENTICATION') break;
      }
    }
    throw lastError;
  }
  const chunks = await splitNormalizedAudio(normalizedPath, {
    durationSeconds,
    chunkSeconds: 25,
    overlapSeconds: 0,
    preferredSilences: quality?.activity?.silenceIntervals ?? [],
  });
  try {
    const references =
      rubric.spec.mode === 'reading'
        ? referenceSlices(
            rubric.spec.referenceText,
            whisper.words,
            chunks,
            durationSeconds,
          )
        : chunks.map(() => '');
    const results = [];
    for (let index = 0; index < chunks.length; index++) {
      const rawNormalized = await withCircuitBreaker('azure-speech', () =>
        evaluateRest({
          rutaWav: chunks[index].path,
          textoReferencia: references[index],
          idioma: rubric.spec.targetLocale,
          configuracion: azureConfig,
          mode: rubric.spec.mode,
          preserveRaw: true,
        }),
      );
      const normalized =
        rawNormalized?.rawResponse || rawNormalized?.recognitionStatus
          ? rawNormalized
          : rawNormalized;
      results.push({
        ...chunks[index],
        referenceText: references[index],
        result: normalized,
      });
    }
    return results.length === 1
      ? {
          ...results[0].result,
          mode: rubric.spec.mode,
          locale: rubric.spec.targetLocale,
          completenessScore:
            rubric.spec.mode === 'reading'
              ? results[0].result.completenessScore
              : null,
        }
      : aggregateAzureChunks(results, {
          mode: rubric.spec.mode,
          locale: rubric.spec.targetLocale,
          referenceText: rubric.spec.referenceText,
        });
  } finally {
    await cleanAudioChunks(chunks);
  }
}

export function transcriptComparison(whisper, azure) {
  const azureText = azure?.lexicalText || azure?.displayText || '';
  const disagreement = azureText ? tokenErrorRate(whisper?.text ?? '', azureText) : null;
  return {
    primary: {
      provider: 'groq',
      text: whisper?.text ?? '',
      language: whisper?.language ?? null,
      words: whisper?.words ?? [],
      segments: whisper?.segments ?? [],
      chunking: whisper?.chunking ?? null,
    },
    secondary: azureText
      ? {
          provider: 'azure-speech',
          text: azureText,
        }
      : null,
    providerDisagreement: disagreement,
    reviewRequired: disagreement !== null && disagreement > 0.15,
  };
}
