import { randomUUID } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { basename } from 'node:path';

const baseUrl = (process.argv[2] || 'https://backend-gordon.onrender.com')
  .replace(/\/+$/, '');
const audioPath = process.argv[3];

if (!audioPath) {
  console.error(
    'Uso: node scripts/diagnose-deployed.mjs BASE_URL /ruta/audio.wav',
  );
  process.exitCode = 2;
} else {
  const requestJson = async (path, init) => {
    const response = await fetch(`${baseUrl}${path}`, init);
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      throw new Error(
        `${path} respondió ${response.status}: ${JSON.stringify(body)}`,
      );
    }
    return { body, response };
  };

  const rubricSpec = {
    mode: 'spontaneous',
    targetLocale: 'en-US',
    cefr: 'B2',
    instruction:
      'Describe a past trip, compare transportation options, and explain the reasons for your decisions.',
    communicativePurpose: 'Narrate and explain a sequence of past events',
  };

  const { body: drafted } = await requestJson('/api/v2/rubrics/draft', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(rubricSpec),
  });
  const { body: confirmed } = await requestJson('/api/v2/rubrics/confirm', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ draft: drafted.draft }),
  });

  const audioBytes = await readFile(audioPath);
  const form = new FormData();
  form.append(
    'audio',
    new Blob([audioBytes], { type: 'audio/wav' }),
    basename(audioPath),
  );
  form.append('confirmedRubricToken', confirmed.confirmedRubricToken);
  form.append('consentToStore', 'false');

  const assessmentResponse = await fetch(`${baseUrl}/api/v2/assessments`, {
    method: 'POST',
    headers: { 'idempotency-key': `diagnostic-${randomUUID()}` },
    body: form,
  });
  const report = await assessmentResponse.json().catch(() => null);

  const summary = {
    httpStatus: assessmentResponse.status,
    requestId:
      assessmentResponse.headers.get('x-request-id') ?? report?.requestId ?? null,
    assessmentId: report?.assessmentId ?? null,
    reportStatus: report?.status ?? null,
    quality: report?.quality
      ? {
          status: report.quality.status,
          durationSeconds: report.quality.metrics?.durationSeconds ?? null,
          voicedSeconds: report.quality.metrics?.voicedSeconds ?? null,
          speechRatio: report.quality.metrics?.speechRatio ?? null,
        }
      : null,
    transcriptWordCount:
      typeof report?.transcript?.primary?.text === 'string'
        ? report.transcript.primary.text.trim().split(/\s+/).filter(Boolean).length
        : null,
    linguisticError: report?.providerEvidence?.linguistic?.error ?? null,
    linguisticSufficiency:
      report?.providerEvidence?.linguistic?.evidence?.sufficiency ?? null,
    dimensions: Object.fromEntries(
      Object.entries(report?.dimensions ?? {}).map(([id, dimension]) => [
        id,
        {
          status: dimension.status,
          score: dimension.score,
          reasonCode: dimension.reasonCode,
        },
      ]),
    ),
  };

  console.log(JSON.stringify(summary, null, 2));
}
