import {
  CALIBRATION_VERSION,
  DIMENSION_IDS,
  PROMPT_VERSION,
  SCHEMA_VERSION,
  SCORE_PROFILES,
  dimensionResult,
  calculateOverall,
  expectedScore,
  newAssessmentId,
  normalizeTokens,
  probabilitiesForBand,
  probabilitiesForScore,
} from './domain.js';
import {
  extractLinguisticEvidence,
  generateExpectedPhoneticFromWhisper,
  generatePedagogicalFeedback,
  judgePronunciationFromPhonetics,
  runDoubleLinguisticJudging,
  transcribePhonemesLiterally,
} from './linguistic.js';
import {
  transcribeWhisper,
} from './providers.js';
import {
  PROMPT_EXAMPLES_STATUS,
  PROMPT_MANIFEST_HASH,
} from './prompts.js';
import { WHISPER_LITERAL_POLICY_VERSION } from '../whisper.js';

function limitationForMode(rubric, id) {
  if (
    rubric.spec.mode === 'reading' &&
    ['communication', 'grammar', 'vocabulary'].includes(id)
  ) {
    return [
      `${id} se limita a la realización oral del texto de referencia; no mide producción libre.`,
    ];
  }
  return [];
}

function attachLinguisticTimestamps(evidence, whisper) {
  if (!evidence || !Array.isArray(evidence.findings)) return evidence;
  const timeline = (whisper?.words ?? []).flatMap((word) =>
    normalizeTokens(word.word).map(() => ({
      startSec:
        typeof word.start === 'number' && Number.isFinite(word.start)
          ? word.start
          : null,
      endSec:
        typeof word.end === 'number' && Number.isFinite(word.end)
          ? word.end
          : null,
    })),
  );
  return {
    ...evidence,
    findings: evidence.findings.map((finding) => {
      const start = timeline[finding.transcriptStart];
      const end = timeline[finding.transcriptEnd];
      return {
        ...finding,
        tokenIds: Array.from(
          {
            length:
              Math.max(
                finding.transcriptStart,
                finding.transcriptEnd,
              ) -
              finding.transcriptStart +
              1,
          },
          (_value, index) => `t${finding.transcriptStart + index}`,
        ),
        startSec: start?.startSec ?? null,
        endSec: end?.endSec ?? null,
      };
    }),
  };
}

function unavailableAcoustic(id, constructScope) {
  return dimensionResult({
    id,
    status: 'unavailable',
    methodId: 'remote-phonetic-alignment-pending',
    reliability: 'unknown',
    reasonCode: 'REMOTE_PHONETIC_SCORING_PENDING',
    limitations: [
      'La puntuación acústica está pendiente de recibir evidencia del endpoint fonético remoto y alinearla contra la pronunciación esperada.',
    ],
    constructScope,
    reviewRequired: true,
  });
}

export function pronunciationFromPhoneticJudge({
  judged,
  error,
  phoneticEvidence,
  constructScope,
}) {
  if (error) {
    return dimensionResult({
      id: 'pronunciation',
      status: 'providerError',
      methodId: 'deepseek-v4-phonetic-judge-v1',
      reliability: 'unknown',
      reasonCode: error.code ?? 'PHONETIC_JUDGE_ERROR',
      limitations: [error.message],
      constructScope,
      reviewRequired: true,
    });
  }
  if (!judged) return unavailableAcoustic('pronunciation', constructScope);
  const band =
    Number.isInteger(judged.band) && judged.band >= 1 && judged.band <= 4
      ? judged.band
      : 1;
  const probabilities = probabilitiesForBand(band, 0.3);
  const score = expectedScore(probabilities);
  const reliability =
    typeof phoneticEvidence?.confidence === 'number' &&
    phoneticEvidence.confidence >= 0.65
      ? 'medium'
      : 'low';
  const alignments = new Map(
    (judged.wordAlignments ?? []).map((alignment) => [
      alignment.id,
      alignment,
    ]),
  );
  return dimensionResult({
    id: 'pronunciation',
    status: 'scored',
    score,
    rawScore: null,
    probabilities,
    interval90: {
      low: Math.max(0, score - 15),
      high: Math.min(100, score + 15),
    },
    reliability,
    methodId: 'deepseek-v4-phonetic-judge-v1',
    evidence: (judged.observations ?? []).map((observation, index) => {
      const alignment = alignments.get(observation.alignmentId);
      return {
        id: `phonetic-observation-${index}`,
        kind: 'phoneticComparison',
        claim:
          `Esperado: ${observation.expected}. Observado: ${observation.observed}. ` +
          observation.explanation,
        expected: observation.expected,
        observed: observation.observed,
        affectsIntelligibility: observation.affectsIntelligibility,
        startSec: alignment?.startSec ?? null,
        endSec: alignment?.endSec ?? null,
        source: 'wav2vec2-local+deepseek-v4',
      };
    }),
    limitations: [
      judged.rationale,
      ifPronunciationWasForced(judged),
      'Resultado provisional: la secuencia IPA local y el juez todavía requieren calibración humana.',
    ].filter(Boolean),
    constructScope,
    reviewRequired: true,
  });
}

function ifPronunciationWasForced(judged) {
  return judged.status === 'scored' && judged.band >= 1
    ? null
    : 'La evidencia disponible se puntuó con confiabilidad baja en lugar de descartarse.';
}

function clampScore(value) {
  return Math.max(0, Math.min(100, value));
}

export function provisionalFluencyFromTimings({
  phoneticEvidence,
  whisperEvidence,
  quality,
  constructScope,
}) {
  const durationSeconds = Number(quality?.metrics?.durationSeconds);
  const voicedSeconds = Number(quality?.metrics?.voicedSeconds);
  const events = (Array.isArray(phoneticEvidence?.events)
    ? phoneticEvidence.events
    : []
  )
    .filter(
      (event) =>
        Number.isFinite(event?.startSec) &&
        Number.isFinite(event?.endSec) &&
        event.endSec >= event.startSec,
    )
    .sort((a, b) => a.startSec - b.startSec);
  let score;
  let claim;
  let methodId;
  if (events.length >= 2) {
    const articulatedSeconds = events.reduce(
      (total, event) => total + Math.max(0, event.endSec - event.startSec),
      0,
    );
    let pauseSeconds = 0;
    let pauseCount = 0;
    for (let index = 1; index < events.length; index++) {
      const gap = Math.max(0, events[index].startSec - events[index - 1].endSec);
      if (gap >= 0.25) {
        pauseSeconds += gap;
        pauseCount++;
      }
    }
    const effectiveVoice =
      Number.isFinite(voicedSeconds) && voicedSeconds > 0
        ? voicedSeconds
        : Math.max(
            0.001,
            (events.at(-1)?.endSec ?? 0) -
              (events[0]?.startSec ?? 0) -
              pauseSeconds,
          );
    const phonemesPerSecond = events.length / effectiveVoice;
    const paceScore = clampScore(100 - Math.abs(phonemesPerSecond - 12) * 8);
    const continuityBase =
      Number.isFinite(durationSeconds) && durationSeconds > 0
        ? 1 - pauseSeconds / durationSeconds
        : articulatedSeconds / Math.max(0.001, articulatedSeconds + pauseSeconds);
    const continuityScore = clampScore(continuityBase * 100);
    score = clampScore(paceScore * 0.45 + continuityScore * 0.55);
    claim =
      `${events.length} fonemas en ${effectiveVoice.toFixed(1)} s de voz; ritmo ${phonemesPerSecond.toFixed(1)} fonemas/s; ` +
      `${pauseCount} pausas internas de al menos 0.25 s.`;
    methodId = 'wav2vec2-timing-provisional-v1';
  } else {
    const words = normalizeTokens(whisperEvidence?.text ?? '');
    if (
      words.length === 0 ||
      !Number.isFinite(durationSeconds) ||
      durationSeconds <= 0
    ) {
      return unavailableAcoustic('fluency', constructScope);
    }
    const effectiveVoice =
      Number.isFinite(voicedSeconds) && voicedSeconds > 0
        ? voicedSeconds
        : durationSeconds;
    const wordsPerMinute = (words.length / effectiveVoice) * 60;
    const paceScore = clampScore(100 - Math.abs(wordsPerMinute - 130) * 0.8);
    const continuityScore = clampScore((effectiveVoice / durationSeconds) * 100);
    score = clampScore(paceScore * 0.55 + continuityScore * 0.45);
    claim =
      `${words.length} palabras en ${effectiveVoice.toFixed(1)} s de voz; ` +
      `ritmo aproximado ${wordsPerMinute.toFixed(0)} palabras/min.`;
    methodId = 'whisper-timing-provisional-v1';
  }
  const confidence = phoneticEvidence?.confidence;
  const reliability =
    typeof confidence === 'number' && confidence >= 0.65 ? 'medium' : 'low';
  const halfWidth = reliability === 'medium' ? 18 : 25;
  return dimensionResult({
    id: 'fluency',
    status: 'scored',
    score,
    rawScore: score,
    probabilities: probabilitiesForScore(score),
    interval90: {
      low: Math.max(0, score - halfWidth),
      high: Math.min(100, score + halfWidth),
    },
    reliability,
    methodId,
    evidence: [
      {
        id: 'fluency-timing-summary',
        kind: 'timingSummary',
        claim,
        value: score,
        unit: '0-100',
        source:
          events.length >= 2 ? 'wav2vec2-local' : 'whisper+audio-quality',
      },
    ],
    limitations: [
      'Puntuación provisional basada en continuidad y ritmo observables; la evidencia parcial reduce la confiabilidad, no elimina el indicador.',
    ],
    constructScope,
    reviewRequired: true,
  });
}

function linguisticDimension({
  id,
  judged,
  evidence,
  rubric,
  providerError,
}) {
  const constructScope =
    rubric.spec.mode === 'reading' ? 'reading_realization' : 'productive_speaking';
  if (providerError) {
    return dimensionResult({
      id,
      status: 'providerError',
      methodId: 'deepseek-v4-double-judge',
      reliability: 'unknown',
      reasonCode: providerError.code ?? 'LINGUISTIC_PROVIDER_ERROR',
      limitations: [providerError.message],
      constructScope,
      reviewRequired: true,
    });
  }
  const item = judged?.dimensions?.[id];
  if (!item || item.status !== 'scored') {
    return dimensionResult({
      id,
      status: 'insufficientEvidence',
      methodId: 'deepseek-v4-double-judge',
      reliability: 'unknown',
      reasonCode:
        item?.reasonCode ?? 'INSUFFICIENT_LINGUISTIC_EVIDENCE',
      evidence: (evidence?.findings ?? []).filter(
        (finding) => finding.dimension === id,
      ),
      limitations: [
        ...limitationForMode(rubric, id),
        item?.rationale ??
          'La evidencia no permite asignar una banda verificable.',
      ],
      constructScope,
      reviewRequired: true,
    });
  }
  const selectedEvidence = new Set(item.evidenceIds);
  return dimensionResult({
    id,
    status: 'scored',
    score: item.score,
    probabilities: item.probabilities,
    interval90: item.interval90,
    reliability:
      item.judgeAgreement && !item.reviewRequired ? 'medium' : 'low',
    methodId: 'deepseek-v4-double-judge',
    evidence: (evidence?.findings ?? []).filter((finding) =>
      selectedEvidence.has(finding.id),
    ),
    limitations: [
      ...limitationForMode(rubric, id),
      'Resultado provisional pendiente de calibración con evaluadores humanos.',
    ],
    constructScope,
    reviewRequired: true,
  });
}

function emptyDimensionsForRetake(rubric, quality) {
  return Object.fromEntries(
    DIMENSION_IDS.map((id) => [
      id,
      dimensionResult({
        id,
        status: 'invalidInput',
        methodId: 'audio-quality-gate',
        reliability: 'unknown',
        reasonCode: quality.reasons[0] ?? 'AUDIO_NOT_SCORABLE',
        limitations: [
          'El audio debe repetirse antes de evaluar el desempeño.',
          ...quality.reasons,
        ],
        constructScope:
          rubric.spec.mode === 'reading' &&
          ['communication', 'grammar', 'vocabulary'].includes(id)
            ? 'reading_realization'
            : 'productive_speaking',
        reviewRequired: true,
      }),
    ]),
  );
}

function reportStatus({ quality, dimensions }) {
  if (!quality.scorable) return 'needsRetake';
  const scored = Object.values(dimensions).filter(
    (dimension) => dimension.status === 'scored',
  ).length;
  if (scored === DIMENSION_IDS.length) {
    return Object.values(dimensions).some(
      (dimension) => dimension.reviewRequired,
    )
      ? 'needsReview'
      : 'complete';
  }
  return scored > 0 ? 'partial' : 'failed';
}

export async function runAssessment({
  rubric,
  originalAudio,
  normalizedAudio,
  signature,
  quality,
  groqClient,
  linguisticClient = groqClient,
  linguisticModel,
  whisperModel,
  analyzeSpeech,
  phoneticEvidence,
  phoneticError = null,
  requestId,
}) {
  const assessmentId = newAssessmentId();
  const base = {
    schemaVersion: SCHEMA_VERSION,
    requestId,
    assessmentId,
    createdAt: new Date().toISOString(),
    taskSnapshot: {
      mode: rubric.spec.mode,
      targetLocale: rubric.spec.targetLocale,
      cefr: rubric.spec.cefr,
      instruction: rubric.spec.instruction,
      referenceText:
        rubric.spec.mode === 'reading' ? rubric.spec.referenceText : null,
      communicativePurpose: rubric.spec.communicativePurpose,
      nativeLanguage: rubric.spec.nativeLanguage,
    },
    rubric: {
      id: rubric.id,
      version: rubric.version,
      dimensions: rubric.dimensions,
      scoreProfile: rubric.scoreProfile,
      limitations: rubric.limitations,
    },
    quality: {
      ...quality,
      original: signature,
    },
  };
  if (!quality.scorable) {
    const dimensions = emptyDimensionsForRetake(rubric, quality);
    return {
      ...base,
      status: 'needsRetake',
      transcript: null,
      speechEvidence: {
        pauses: { status: 'unavailable', items: null },
        elongations: { status: 'unavailable', items: null },
        phonetic: {
          status: phoneticError ? 'providerError' : 'unavailable',
          transcript: null,
          model: null,
          durationSeconds: null,
          confidence: null,
          events: null,
          error: phoneticError,
        },
        expectedPhonetic: {
          status: 'unavailable',
          transcript: null,
          model: null,
          targetLocale: rubric.spec.targetLocale,
          source: null,
          words: null,
          error: null,
        },
        annotatedTranscript: null,
        annotatedTranscriptStatus: 'unavailable',
        annotatedTranscriptMethod: null,
        annotatedTranscriptError: null,
      },
      dimensions,
      overall: calculateOverall(dimensions, rubric.scoreProfile),
      feedback: {
        priority: 'Repetir la grabación',
        activity:
          'Graba nuevamente en un lugar silencioso y mantén el micrófono a una distancia constante.',
        rationale: quality.reasons.join(', '),
        byDimension: [],
      },
      provenance: {
        calibrationVersion: CALIBRATION_VERSION,
        promptVersion: PROMPT_VERSION,
        promptHash: PROMPT_MANIFEST_HASH,
        promptExamples: PROMPT_EXAMPLES_STATUS,
        scoreProfileId: rubric.scoreProfile.id,
        scoreProfileVersion: rubric.scoreProfile.version,
      },
      consentStorage: { requested: false, stored: false },
      providerEvidence: null,
    };
  }

  let whisper = null;
  let whisperError = null;
  try {
    whisper = await transcribeWhisper({
      client: groqClient,
      normalizedPath: normalizedAudio,
      durationSeconds: quality.metrics.durationSeconds,
      rubric,
      quality,
      model: whisperModel,
    });
  } catch (error) {
    whisperError = {
      code: error?.code ?? 'WHISPER_PROVIDER_ERROR',
      message:
        error?.message ??
        'Whisper no pudo producir una transcripción utilizable.',
    };
  }
  const whisperEvidence =
    whisper ??
    {
      provider: 'groq',
      model: whisperModel,
      language: null,
      text: '',
      words: [],
      segments: [],
      chunks: [],
      chunking: null,
    };
  const transcript = {
    primary: {
      provider: 'groq',
      text: whisperEvidence.text,
      language: whisperEvidence.language,
      words: whisperEvidence.words,
      segments: whisperEvidence.segments,
      chunking: whisperEvidence.chunking,
      status: whisper ? 'complete' : 'unavailable',
      error: whisperError,
    },
    secondary: null,
    providerDisagreement: null,
    reviewRequired: false,
  };
  let expectedPhonetic = null;
  let expectedPhoneticError = null;
  if (whisper && whisperEvidence.text?.trim()) {
    try {
      expectedPhonetic = await generateExpectedPhoneticFromWhisper({
        client: linguisticClient,
        model: linguisticModel,
        targetLocale: rubric.spec.targetLocale,
        nativeLanguage: rubric.spec.nativeLanguage,
        transcript: whisperEvidence.text,
        words: whisperEvidence.words,
      });
    } catch (error) {
      expectedPhoneticError = {
        code: error?.code ?? 'EXPECTED_PHONETIC_PROVIDER_ERROR',
        message:
          error?.message ??
          'La pronunciación IPA esperada no pudo generarse desde Whisper.',
        details: error?.details ?? null,
      };
    }
  }
  let speechEvidence = {
    pauses: { status: 'unavailable', items: null },
    elongations: { status: 'unavailable', items: null },
    phonetic: {
      status: phoneticEvidence?.transcript
        ? 'complete'
        : phoneticError
          ? 'providerError'
          : 'unavailable',
      transcript: phoneticEvidence?.transcript ?? null,
      model: phoneticEvidence?.model ?? null,
      durationSeconds:
        phoneticEvidence?.durationSeconds ?? quality.metrics.durationSeconds,
      confidence: phoneticEvidence?.confidence ?? null,
      events: phoneticEvidence?.events ?? null,
      error: phoneticError,
    },
    expectedPhonetic: {
      status: expectedPhonetic
        ? 'complete'
        : expectedPhoneticError
          ? 'providerError'
          : 'unavailable',
      transcript: expectedPhonetic?.transcript ?? null,
      model: expectedPhonetic?.model ?? null,
      targetLocale:
        expectedPhonetic?.targetLocale ?? rubric.spec.targetLocale,
      source: expectedPhonetic?.source ?? null,
      words: expectedPhonetic?.words ?? null,
      error: expectedPhoneticError,
    },
    annotatedTranscript: null,
    annotatedTranscriptStatus: 'unavailable',
    annotatedTranscriptMethod: null,
    annotatedTranscriptError: null,
  };
  if (typeof analyzeSpeech === 'function' && whisperEvidence.words.length) {
    try {
      const legacy = await analyzeSpeech({
        rutaAudio: normalizedAudio,
        palabras: whisperEvidence.words,
        duracionSegundos: quality.metrics.durationSeconds,
        pronunciacion: null,
      });
      speechEvidence = {
        ...speechEvidence,
        pauses: {
          status: legacy ? 'complete' : 'unavailable',
          items: legacy?.pauses ?? null,
        },
        elongations: {
          status: legacy ? 'complete' : 'unavailable',
          items: legacy?.elongations ?? null,
        },
        method: legacy?.method ?? null,
      };
    } catch (error) {
      speechEvidence.error = {
        code: error?.code ?? 'SPEECH_EVIDENCE_ERROR',
        message: error?.message ?? 'No se pudo medir pausas y duraciones.',
      };
    }
  }
  let phoneticLiteralTranscription = null;
  if (phoneticEvidence?.transcript) {
    try {
      phoneticLiteralTranscription = await transcribePhonemesLiterally({
        client: linguisticClient,
        model: linguisticModel,
        targetLocale: rubric.spec.targetLocale,
        phoneticEvidence,
      });
      speechEvidence = {
        ...speechEvidence,
        annotatedTranscript: phoneticLiteralTranscription?.text ?? null,
        annotatedTranscriptStatus: phoneticLiteralTranscription
          ? 'complete'
          : 'unavailable',
        annotatedTranscriptMethod:
          phoneticLiteralTranscription?.methodId ?? null,
        annotatedTranscriptError: null,
      };
    } catch (error) {
      speechEvidence = {
        ...speechEvidence,
        annotatedTranscript: null,
        annotatedTranscriptStatus: 'providerError',
        annotatedTranscriptMethod:
          'deepseek-phoneme-literal-transcription-v3',
        annotatedTranscriptError: {
          code: error?.code ?? 'PHONETIC_LITERAL_TRANSCRIPT_ERROR',
          message:
            error?.message ??
            'No se pudo convertir la secuencia fonética en texto literal.',
          details: error?.details ?? null,
        },
      };
    }
  }

  let linguisticEvidence = null;
  let judging = null;
  let linguisticError = null;
  if (!whisper) {
    linguisticError = {
      code: whisperError.code,
      message:
        'Sin la transcripción independiente de Whisper no se asignan puntuaciones lingüísticas.',
    };
  } else {
    try {
      linguisticEvidence = attachLinguisticTimestamps(
        await extractLinguisticEvidence({
          client: linguisticClient,
          model: linguisticModel,
          rubric,
          transcript: transcript.primary.text,
          secondaryTranscript: '',
          secondaryRecognitionStatus: null,
          secondaryConfidence: null,
          quality,
          providerDisagreement: transcript.providerDisagreement,
        }),
        whisperEvidence,
      );
      judging = await runDoubleLinguisticJudging({
        client: linguisticClient,
        model: linguisticModel,
        rubric,
        evidence: linguisticEvidence,
      });
    } catch (error) {
      linguisticError = {
        code: error?.code ?? 'LINGUISTIC_PROVIDER_ERROR',
        message:
          error?.message ??
          'El análisis lingüístico no pudo completarse con doble evaluación.',
        details: error?.details ?? null,
      };
    }
  }
  let pronunciationJudging = null;
  let pronunciationError = null;
  if (whisper && phoneticEvidence?.transcript) {
    try {
      pronunciationJudging = await judgePronunciationFromPhonetics({
        client: linguisticClient,
        model: linguisticModel,
        rubric,
        transcript: whisperEvidence.text,
        words: whisperEvidence.words,
        phoneticEvidence,
        expectedPhonetic,
      });
    } catch (error) {
      pronunciationError = {
        code: error?.code ?? 'PHONETIC_JUDGE_ERROR',
        message:
          error?.message ??
          'El juez fonético no pudo completar la comparación.',
        details: error?.details ?? null,
      };
    }
  }
  const scopes = Object.fromEntries(
    rubric.dimensions.map((dimension) => [
      dimension.id,
      dimension.constructScope,
    ]),
  );
  const dimensions = {
    communication: linguisticDimension({
      id: 'communication',
      judged: judging,
      evidence: linguisticEvidence,
      rubric,
      providerError: linguisticError,
    }),
    pronunciation: pronunciationFromPhoneticJudge({
      judged: pronunciationJudging,
      error: pronunciationError,
      phoneticEvidence,
      constructScope: scopes.pronunciation,
    }),
    grammar: linguisticDimension({
      id: 'grammar',
      judged: judging,
      evidence: linguisticEvidence,
      rubric,
      providerError: linguisticError,
    }),
    vocabulary: linguisticDimension({
      id: 'vocabulary',
      judged: judging,
      evidence: linguisticEvidence,
      rubric,
      providerError: linguisticError,
    }),
    fluency: provisionalFluencyFromTimings({
      phoneticEvidence,
      whisperEvidence,
      quality,
      constructScope: scopes.fluency,
    }),
  };
  const overall = calculateOverall(
    dimensions,
    SCORE_PROFILES[rubric.spec.mode],
  );
  let feedback = {
    priority: 'Revisión docente',
    activity:
      'Escucha la grabación junto con la evidencia disponible antes de asignar una decisión final.',
    rationale:
      linguisticError?.message ??
      'El resultado automático es provisional y requiere confirmación docente.',
    byDimension: [],
  };
  if (linguisticEvidence && judging) {
    try {
      feedback = await generatePedagogicalFeedback({
        client: linguisticClient,
        model: linguisticModel,
        rubric,
        dimensions,
        evidence: linguisticEvidence,
      });
    } catch (error) {
      feedback.error = {
        code: error?.code ?? 'FEEDBACK_PROVIDER_ERROR',
        message:
          error?.message ??
          'No se pudo generar la recomendación pedagógica.',
      };
    }
  }
  return {
    ...base,
    status: reportStatus({ quality, dimensions }),
    transcript,
    speechEvidence,
    dimensions,
    overall,
    feedback,
    provenance: {
      providers: {
        whisper: {
          provider: 'groq',
          model: whisperModel,
          status: whisper ? 'complete' : 'providerError',
          chunking: whisperEvidence.chunking,
          error: whisperError,
          parameters: {
            language: rubric.spec.asrLanguage,
            temperature: 0,
            responseFormat: 'verbose_json',
            timestampGranularities: ['word', 'segment'],
            transcriptionPolicy: WHISPER_LITERAL_POLICY_VERSION,
            vocabularyHintsApplied: false,
          },
        },
        linguistic: {
          provider: 'opencode-zen',
          model: linguisticModel,
          promptVersion: PROMPT_VERSION,
          doubleJudge: true,
          adjudicated: judging?.adjudicated ?? false,
          parameters: {
            temperature: 0,
            responseFormat: 'json_object',
            thinking: 'disabled',
          },
        },
        phonetic: {
          provider: phoneticEvidence?.provider ?? 'remote-wav2vec2',
          model: phoneticEvidence?.model ?? null,
          status: phoneticEvidence?.transcript
            ? 'complete'
            : phoneticError
              ? 'providerError'
              : 'unavailable',
          confidence: phoneticEvidence?.confidence ?? null,
          nativeLanguage: rubric.spec.nativeLanguage,
          judgeModel: linguisticModel,
          literalTranscriptModel: linguisticModel,
          literalTranscriptMethod:
            phoneticLiteralTranscription?.methodId ?? null,
        },
        expectedPhonetic: {
          provider: expectedPhonetic?.provider ?? 'opencode-zen',
          model: expectedPhonetic?.model ?? linguisticModel,
          status: expectedPhonetic
            ? 'complete'
            : expectedPhoneticError
              ? 'providerError'
              : 'unavailable',
          methodId:
            expectedPhonetic?.methodId ?? 'deepseek-whisper-expected-ipa-v2',
          source: 'whisper-primary',
          targetLocale: rubric.spec.targetLocale,
          error: expectedPhoneticError,
        },
      },
      calibrationVersion: CALIBRATION_VERSION,
      calibrationStatus: 'provisional',
      promptVersion: PROMPT_VERSION,
      promptHash: PROMPT_MANIFEST_HASH,
      promptExamples: PROMPT_EXAMPLES_STATUS,
      scoreProfileId: rubric.scoreProfile.id,
      scoreProfileVersion: rubric.scoreProfile.version,
    },
    consentStorage: { requested: false, stored: false },
    providerEvidence: {
      whisper: {
        status: whisper ? 'complete' : 'providerError',
        chunks: whisperEvidence.chunks,
        error: whisperError,
      },
      linguistic: {
        evidence: linguisticEvidence,
        judging,
        error: linguisticError,
      },
      phonetic: {
        input: phoneticEvidence ?? null,
        inputError: phoneticError,
        literalTranscription: phoneticLiteralTranscription,
        literalTranscriptionError:
          speechEvidence.annotatedTranscriptError ?? null,
        judging: pronunciationJudging,
        error: pronunciationError,
      },
      expectedPhonetic: {
        output: expectedPhonetic,
        error: expectedPhoneticError,
      },
    },
    _private: {
      originalAudio,
      normalizedAudio,
    },
  };
}
