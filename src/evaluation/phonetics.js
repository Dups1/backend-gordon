import { readFile } from 'node:fs/promises';

import { EvaluationError } from './domain.js';

export const PHONEME_ENDPOINT =
  process.env.GORDON_PHONEME_URL?.trim() || '';
export const PHONEME_API_KEY =
  process.env.GORDON_PHONEME_API_KEY?.trim() || '';
export const PHONEME_TIMEOUT_MS = Number.parseInt(
  process.env.GORDON_PHONEME_TIMEOUT_MS ?? String(4 * 60 * 1000),
  10,
);

export function phonemeAnalysisEndpoint(value) {
  const configured = String(value ?? '').trim().replace(/\/+$/u, '');
  if (!configured) return '';
  return /\/api\/v1\/phonemes$/u.test(configured)
    ? configured
    : `${configured}/api/v1/phonemes`;
}

function responseMessage(body, status) {
  if (body && typeof body === 'object') {
    const detail = body.detail;
    if (typeof detail === 'string' && detail.trim()) return detail.trim();
    const message = body.error?.message;
    if (typeof message === 'string' && message.trim()) return message.trim();
  }
  return `El endpoint fonético remoto respondió con HTTP ${status}.`;
}

function parseResponseBody(text) {
  try {
    const parsed = JSON.parse(text);
    return parsed && typeof parsed === 'object' ? parsed : null;
  } catch {
    return null;
  }
}

function normalizeEvents(value) {
  if (!Array.isArray(value)) return [];
  return value
    .map((event) => {
      const type = typeof event?.type === 'string' ? event.type.trim() : '';
      const phoneme =
        typeof event?.phoneme === 'string' ? event.phoneme.trim() : '';
      const startSec = Number(event?.startSec);
      const endSec = Number(event?.endSec);
      const confidence = Number(event?.confidence);
      if (!phoneme || !['phoneme', 'boundary'].includes(type)) return null;
      return {
        type: type || 'phoneme',
        phoneme: phoneme.slice(0, 30),
        startSec: Number.isFinite(startSec) ? startSec : null,
        endSec: Number.isFinite(endSec) ? endSec : null,
        confidence: Number.isFinite(confidence) ? confidence : null,
      };
    })
    .filter(Boolean)
    .slice(0, 12000);
}

export function normalizeRemotePhonemeResponse(value, endpoint) {
  const transcript =
    typeof value?.phoneticTranscript === 'string'
      ? value.phoneticTranscript.trim().slice(0, 12000)
      : '';
  const events = normalizeEvents(value?.events);
  if (!transcript && !events.length) {
    throw new EvaluationError(
      502,
      'El endpoint fonético remoto no devolvió fonemas utilizables.',
      'PHONEME_REMOTE_EMPTY',
      { endpoint },
    );
  }
  const confidence = Number(value?.meanConfidence ?? value?.confidence);
  const durationSeconds = Number(value?.durationSeconds);
  return {
    provider: 'remote-wav2vec2',
    endpoint,
    model:
      typeof value?.model === 'string' && value.model.trim()
        ? value.model.trim().slice(0, 200)
        : 'wav2vec2-remote',
    transcript,
    durationSeconds: Number.isFinite(durationSeconds) && durationSeconds >= 0
      ? durationSeconds
      : null,
    confidence: Number.isFinite(confidence) ? confidence : null,
    events,
    normalization: value?.normalization ?? null,
    requestId:
      typeof value?.requestId === 'string' ? value.requestId.slice(0, 120) : null,
  };
}

export async function requestRemotePhonemeEvidence({
  audioPath,
  endpoint = PHONEME_ENDPOINT,
  apiKey = PHONEME_API_KEY,
  timeoutMs = PHONEME_TIMEOUT_MS,
  requestId = null,
  fetchImpl = globalThis.fetch,
}) {
  const url = phonemeAnalysisEndpoint(endpoint);
  if (!url) return null;
  if (typeof fetchImpl !== 'function') {
    throw new EvaluationError(
      503,
      'El runtime no dispone de fetch para llamar al endpoint fonético remoto.',
      'PHONEME_REMOTE_FETCH_UNAVAILABLE',
    );
  }

  const bytes = await readFile(audioPath);
  const form = new FormData();
  form.append(
    'audio',
    new Blob([bytes], { type: 'audio/wav' }),
    'normalized.wav',
  );
  const headers = { Accept: 'application/json' };
  if (requestId) headers['X-Request-Id'] = requestId;
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;

  const controller = new AbortController();
  const timer = setTimeout(
    () => controller.abort(),
    Math.max(1000, Number.isFinite(timeoutMs) ? timeoutMs : PHONEME_TIMEOUT_MS),
  );
  try {
    const response = await fetchImpl(url, {
      method: 'POST',
      headers,
      body: form,
      signal: controller.signal,
    });
    const bodyText = await response.text();
    const body = parseResponseBody(bodyText);
    if (!response.ok) {
      throw new EvaluationError(
        response.status === 429 ? 429 : 502,
        responseMessage(body, response.status),
        response.status === 429
          ? 'PHONEME_REMOTE_RATE_LIMIT'
          : 'PHONEME_REMOTE_HTTP_ERROR',
        { endpoint: url, providerStatus: response.status },
      );
    }
    return normalizeRemotePhonemeResponse(body, url);
  } catch (error) {
    if (error instanceof EvaluationError) throw error;
    if (error?.name === 'AbortError') {
      throw new EvaluationError(
        504,
        'El endpoint fonético remoto agotó el tiempo de respuesta.',
        'PHONEME_REMOTE_TIMEOUT',
        { endpoint: url, timeoutMs },
      );
    }
    throw new EvaluationError(
      502,
      'No se pudo conectar con el endpoint fonético remoto.',
      'PHONEME_REMOTE_UNAVAILABLE',
      { endpoint: url, providerMessage: String(error?.message ?? '').slice(0, 300) },
    );
  } finally {
    clearTimeout(timer);
  }
}
