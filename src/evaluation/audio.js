import { createHash, randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { readFile, stat, unlink } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import ffmpegPath from 'ffmpeg-static';

import { EvaluationError } from './domain.js';

const tempDirectory = path.join(os.tmpdir(), 'backend-gordon-audio');
const allowedSignatures = new Map([
  ['wav', new Set(['.wav'])],
  ['flac', new Set(['.flac'])],
  ['ogg', new Set(['.ogg', '.opus'])],
  ['mp3', new Set(['.mp3', '.mpeg', '.mpga'])],
  ['mp4', new Set(['.m4a', '.mp4'])],
  ['webm', new Set(['.webm'])],
]);

function detectContainer(header) {
  if (
    header.length >= 12 &&
    header.subarray(0, 4).toString('ascii') === 'RIFF' &&
    header.subarray(8, 12).toString('ascii') === 'WAVE'
  ) {
    return 'wav';
  }
  if (header.subarray(0, 4).toString('ascii') === 'fLaC') return 'flac';
  if (header.subarray(0, 4).toString('ascii') === 'OggS') return 'ogg';
  if (
    header.subarray(0, 3).toString('ascii') === 'ID3' ||
    (header[0] === 0xff && (header[1] & 0xe0) === 0xe0)
  ) {
    return 'mp3';
  }
  if (header.length >= 12 && header.subarray(4, 8).toString('ascii') === 'ftyp') {
    return 'mp4';
  }
  if (
    header.length >= 4 &&
    header[0] === 0x1a &&
    header[1] === 0x45 &&
    header[2] === 0xdf &&
    header[3] === 0xa3
  ) {
    return 'webm';
  }
  return null;
}

function runFfmpeg(args, { timeoutMs = 120000, bin = ffmpegPath } = {}) {
  if (!bin) {
    throw new EvaluationError(
      503,
      'FFmpeg no está disponible.',
      'FFMPEG_UNAVAILABLE',
    );
  }
  return new Promise((resolve, reject) => {
    const child = spawn(bin, args, { stdio: ['ignore', 'ignore', 'pipe'] });
    let diagnostics = '';
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill('SIGKILL');
    }, timeoutMs);
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', (chunk) => {
      if (diagnostics.length < 1024 * 1024) diagnostics += chunk;
    });
    child.once('error', () => {
      clearTimeout(timer);
      reject(
        new EvaluationError(
          503,
          'No fue posible iniciar FFmpeg.',
          'FFMPEG_UNAVAILABLE',
        ),
      );
    });
    child.once('close', (code) => {
      clearTimeout(timer);
      if (timedOut) {
        reject(
          new EvaluationError(
            504,
            'El procesamiento de audio agotó el tiempo.',
            'AUDIO_PROCESSING_TIMEOUT',
          ),
        );
      } else if (code !== 0) {
        reject(
          new EvaluationError(
            422,
            'El archivo no contiene audio decodificable.',
            'INVALID_AUDIO',
            { diagnostics: diagnostics.slice(-500) },
          ),
        );
      } else {
        resolve(diagnostics);
      }
    });
  });
}

export async function inspectAudioSignature(filePath, originalName) {
  const bytes = await readFile(filePath);
  const container = detectContainer(bytes.subarray(0, 32));
  if (!container) {
    throw new EvaluationError(
      415,
      'La firma del archivo no corresponde a un audio admitido.',
      'AUDIO_SIGNATURE_INVALID',
    );
  }
  const extension = path.extname(originalName).toLowerCase();
  if (!allowedSignatures.get(container)?.has(extension)) {
    throw new EvaluationError(
      415,
      'La extensión no coincide con el contenido real del audio.',
      'AUDIO_CONTAINER_MISMATCH',
      { detectedContainer: container },
    );
  }
  return {
    container,
    originalBytes: bytes.length,
    originalSha256: createHash('sha256').update(bytes).digest('hex'),
  };
}

export async function normalizeAudio(
  inputPath,
  { maxBytes = 25 * 1024 * 1024, bin = ffmpegPath } = {},
) {
  const outputPath = path.join(tempDirectory, `${randomUUID()}.wav`);
  try {
    await runFfmpeg(
      [
        '-nostdin',
        '-hide_banner',
        '-loglevel',
        'error',
        '-y',
        '-i',
        inputPath,
        '-map',
        '0:a:0',
        '-vn',
        '-ac',
        '1',
        '-ar',
        '16000',
        '-c:a',
        'pcm_s16le',
        outputPath,
      ],
      { bin },
    );
    const info = await stat(outputPath);
    if (!info.size || info.size > maxBytes) {
      throw new EvaluationError(
        info.size > maxBytes ? 413 : 422,
        info.size > maxBytes
          ? 'El audio normalizado supera 25 MB.'
          : 'La normalización no produjo audio.',
        info.size > maxBytes ? 'NORMALIZED_AUDIO_TOO_LARGE' : 'INVALID_AUDIO',
      );
    }
    return outputPath;
  } catch (error) {
    await unlink(outputPath).catch(() => {});
    throw error;
  }
}

function parsePcm16Wav(buffer) {
  if (
    buffer.length < 44 ||
    buffer.subarray(0, 4).toString('ascii') !== 'RIFF' ||
    buffer.subarray(8, 12).toString('ascii') !== 'WAVE'
  ) {
    throw new EvaluationError(422, 'WAV normalizado inválido.', 'INVALID_WAV');
  }
  let offset = 12;
  let format;
  let data;
  while (offset + 8 <= buffer.length) {
    const id = buffer.subarray(offset, offset + 4).toString('ascii');
    const size = buffer.readUInt32LE(offset + 4);
    const start = offset + 8;
    const end = Math.min(buffer.length, start + size);
    if (id === 'fmt ' && end - start >= 16) {
      format = {
        audioFormat: buffer.readUInt16LE(start),
        channels: buffer.readUInt16LE(start + 2),
        sampleRate: buffer.readUInt32LE(start + 4),
        bitsPerSample: buffer.readUInt16LE(start + 14),
      };
    } else if (id === 'data') {
      data = buffer.subarray(start, end);
    }
    offset = end + (size % 2);
  }
  if (
    !format ||
    !data ||
    format.audioFormat !== 1 ||
    format.channels !== 1 ||
    format.sampleRate !== 16000 ||
    format.bitsPerSample !== 16
  ) {
    throw new EvaluationError(
      422,
      'El WAV normalizado no es PCM16 mono a 16 kHz.',
      'INVALID_WAV_FORMAT',
    );
  }
  return { ...format, data };
}

function percentile(values, fraction) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
}

function dbfs(amplitude) {
  return amplitude > 0 ? 20 * Math.log10(amplitude) : -120;
}

export async function analyzeAudioQuality(
  normalizedPath,
  { mode = 'spontaneous', referenceWordCount = 0 } = {},
) {
  const buffer = await readFile(normalizedPath);
  const { data, sampleRate } = parsePcm16Wav(buffer);
  const sampleCount = Math.floor(data.length / 2);
  const durationSeconds = sampleCount / sampleRate;
  const frameSamples = Math.round(sampleRate * 0.02);
  const frameRms = [];
  let sumSquares = 0;
  let clipped = 0;
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
    const value = data.readInt16LE(sampleIndex * 2) / 32768;
    sumSquares += value * value;
    if (Math.abs(value) >= 0.99) clipped++;
  }
  for (let start = 0; start < sampleCount; start += frameSamples) {
    const end = Math.min(sampleCount, start + frameSamples);
    let squares = 0;
    for (let index = start; index < end; index++) {
      const value = data.readInt16LE(index * 2) / 32768;
      squares += value * value;
    }
    frameRms.push(Math.sqrt(squares / Math.max(1, end - start)));
  }
  const noiseRms = Math.max(percentile(frameRms, 0.2), 1 / 32768);
  const voiceThreshold = Math.max(noiseRms * 3.1623, 0.003);
  const maximumFrame = Math.max(...frameRms, 0);
  const positiveFrames = frameRms.filter((value) => value > 0);
  const minimumFrame = positiveFrames.length
    ? Math.min(...positiveFrames)
    : 0;
  const uniformlyVoiced =
    maximumFrame >= 0.01 &&
    minimumFrame > 0 &&
    maximumFrame / minimumFrame < 1.5;
  const speechFrames = uniformlyVoiced
    ? frameRms
    : frameRms.filter((value) => value >= voiceThreshold);
  const voicedMask = uniformlyVoiced
    ? frameRms.map(() => true)
    : frameRms.map((value) => value >= voiceThreshold);
  const silenceIntervals = [];
  let silenceStart = null;
  for (let index = 0; index <= voicedMask.length; index++) {
    const silent = index < voicedMask.length && !voicedMask[index];
    if (silent && silenceStart === null) silenceStart = index * 0.02;
    if (!silent && silenceStart !== null) {
      const end = Math.min(durationSeconds, index * 0.02);
      if (end - silenceStart >= 0.15) {
        silenceIntervals.push({ start: silenceStart, end });
      }
      silenceStart = null;
    }
  }
  const speechRatio = frameRms.length
    ? speechFrames.length / frameRms.length
    : 0;
  const voicedSeconds = durationSeconds * speechRatio;
  const speechRms =
    speechFrames.length > 0
      ? Math.sqrt(
          speechFrames.reduce((sum, value) => sum + value * value, 0) /
            speechFrames.length,
        )
      : 0;
  const snrDb =
    speechRms > noiseRms ? 20 * Math.log10(speechRms / noiseRms) : 0;
  const rms = Math.sqrt(sumSquares / Math.max(1, sampleCount));
  const clippingRatio = sampleCount ? clipped / sampleCount : 0;
  const hardReasons = [];
  const warnings = [];
  if (durationSeconds < 1) hardReasons.push('AUDIO_TOO_SHORT');
  if (durationSeconds > 300) hardReasons.push('AUDIO_TOO_LONG');
  if (voicedSeconds < 1 || speechRatio < 0.02) hardReasons.push('NO_SPEECH');
  if (mode === 'reading') {
    if (referenceWordCount < 10 || voicedSeconds < 5) {
      warnings.push('INSUFFICIENT_READING_SAMPLE');
    }
  } else if (voicedSeconds < 10) {
    warnings.push('INSUFFICIENT_ACOUSTIC_SAMPLE');
  }
  if (snrDb < 15) warnings.push('LOW_SNR');
  if (clippingRatio > 0.01) warnings.push('CLIPPING');
  if (speechRatio < 0.35) warnings.push('LOW_SPEECH_RATIO');
  const normalizedSha256 = createHash('sha256').update(buffer).digest('hex');
  return {
    status:
      hardReasons.length > 0
        ? 'needsRetake'
        : warnings.length > 0
          ? 'warning'
          : 'accepted',
    scorable: hardReasons.length === 0,
    reasons: hardReasons,
    warnings,
    policyVersion: 'audio-quality-enUS-v1',
    metrics: {
      durationSeconds,
      voicedSeconds,
      speechRatio,
      sampleRate,
      channels: 1,
      bitsPerSample: 16,
      rmsDbfs: dbfs(rms),
      noiseFloorDbfs: dbfs(noiseRms),
      estimatedSnrDb: snrDb,
      clippingRatio,
    },
    normalizedSha256,
    activity: {
      method: 'adaptive-frame-energy-v1',
      frameSeconds: 0.02,
      silenceIntervals,
    },
  };
}

export async function prepareAudio({
  filePath,
  originalName,
  mode,
  referenceWordCount,
  maxBytes,
}) {
  const signature = await inspectAudioSignature(filePath, originalName);
  const normalizedPath = await normalizeAudio(filePath, { maxBytes });
  try {
    const quality = await analyzeAudioQuality(normalizedPath, {
      mode,
      referenceWordCount,
    });
    return { normalizedPath, signature, quality };
  } catch (error) {
    await unlink(normalizedPath).catch(() => {});
    throw error;
  }
}

export async function splitNormalizedAudio(
  normalizedPath,
  {
    durationSeconds,
    chunkSeconds = 28,
    overlapSeconds = 2,
    preferredSilences = [],
    bin = ffmpegPath,
  } = {},
) {
  if (
    !Number.isFinite(durationSeconds) ||
    durationSeconds <= 0 ||
    durationSeconds <= chunkSeconds
  ) {
    return [
      {
        path: normalizedPath,
        start: 0,
        end: Math.max(0, durationSeconds ?? 0),
        owned: false,
      },
    ];
  }
  const chunks = [];
  let start = 0;
  while (start < durationSeconds) {
    const targetEnd = Math.min(durationSeconds, start + chunkSeconds);
    const candidates = (Array.isArray(preferredSilences)
      ? preferredSilences
      : []
    )
      .filter(
        (silence) =>
          silence?.end > silence?.start &&
          (silence.start + silence.end) / 2 >= start + chunkSeconds * 0.7 &&
          (silence.start + silence.end) / 2 <= targetEnd + 1,
      )
      .map((silence) => (silence.start + silence.end) / 2)
      .sort(
        (a, b) => Math.abs(a - targetEnd) - Math.abs(b - targetEnd),
      );
    const end =
      targetEnd >= durationSeconds
        ? durationSeconds
        : Math.max(start + 1, candidates[0] ?? targetEnd);
    const outputPath = path.join(tempDirectory, `${randomUUID()}.wav`);
    try {
      await runFfmpeg(
        [
          '-nostdin',
          '-hide_banner',
          '-loglevel',
          'error',
          '-y',
          '-ss',
          String(start),
          '-i',
          normalizedPath,
          '-t',
          String(end - start),
          '-ac',
          '1',
          '-ar',
          '16000',
          '-c:a',
          'pcm_s16le',
          outputPath,
        ],
        { bin },
      );
      chunks.push({ path: outputPath, start, end, owned: true });
    } catch (error) {
      await Promise.all(chunks.map((chunk) => unlink(chunk.path).catch(() => {})));
      await unlink(outputPath).catch(() => {});
      throw error;
    }
    if (end >= durationSeconds) break;
    start = Math.max(start + 1, end - overlapSeconds);
  }
  return chunks;
}

export async function cleanAudioChunks(chunks) {
  await Promise.all(
    (Array.isArray(chunks) ? chunks : [])
      .filter((chunk) => chunk?.owned === true && typeof chunk.path === 'string')
      .map((chunk) => unlink(chunk.path).catch(() => {})),
  );
}
