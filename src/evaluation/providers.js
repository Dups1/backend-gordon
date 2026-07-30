import { createReadStream } from 'node:fs';
import { finished } from 'node:stream/promises';

import { cleanAudioChunks, splitNormalizedAudio } from './audio.js';
import { EvaluationError, normalizeTokens } from './domain.js';
import { withCircuitBreaker } from './resilience.js';
import {
  WHISPER_LITERAL_POLICY_VERSION,
  whisperLiteralOptions,
} from '../whisper.js';

function finite(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function delay(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
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
            normalizeTokens(existing.word)[0] ===
              normalizeTokens(word.word)[0] &&
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
    language:
      chunks.find((chunk) => chunk.result.language)?.result.language ?? null,
  };
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
      let raw;
      let lastError;
      for (let attempt = 0; attempt < 3; attempt++) {
        const fileStream = createReadStream(chunk.path);
        const options = {
          file: fileStream,
          model,
          ...whisperLiteralOptions(rubric.spec.asrLanguage),
        };
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
        } finally {
          if (!fileStream.destroyed) fileStream.destroy();
          await finished(fileStream).catch(() => {});
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
      transcriptionPolicy: WHISPER_LITERAL_POLICY_VERSION,
    };
  } finally {
    await cleanAudioChunks(audioChunks);
  }
}
