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
} from './domain.js';
import {
  extractLinguisticEvidence,
  generatePedagogicalFeedback,
  judgePronunciationFromPhonetics,
  runDoubleLinguisticJudging,
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
    methodId: 'local-phonetic-alignment-pending',
    reliability: 'unknown',
    reasonCode: 'LOCAL_PHONETIC_SCORING_PENDING',
    limitations: [
      'La puntuación acústica está pendiente de integrar el motor fonético local con alineación contra la pronunciación esperada.',
    ],
    constructScope,
    reviewRequired: true,
  });
}

function pronunciationFromPhoneticJudge({
  judged,
  error,
  phoneticEvidence,
  constructScope,
}) {
  if (error) {
    return dimensionResult({
      id: 'pronunciation',
      status: 'providerError',
      methodId: 'gpt-oss-phonetic-judge-v1',
      reliability: 'unknown',
      reasonCode: error.code ?? 'PHONETIC_JUDGE_ERROR',
      limitations: [error.message],
      constructScope,
      reviewRequired: true,
    });
  }
  if (!judged) return unavailableAcoustic('pronunciation', constructScope);
  if (judged.status !== 'scored' || judged.band === 0) {
    return dimensionResult({
      id: 'pronunciation',
      status: 'insufficientEvidence',
      methodId: 'gpt-oss-phonetic-judge-v1',
      reliability: 'low',
      reasonCode: 'PHONETIC_EVIDENCE_INSUFFICIENT',
      evidence: [],
      limitations: [judged.rationale],
      constructScope,
      reviewRequired: true,
    });
  }
  const probabilities = probabilitiesForBand(judged.band, 0.3);
  const score = expectedScore(probabilities);
  const reliability =
    typeof phoneticEvidence?.confidence === 'number' &&
    phoneticEvidence.confidence >= 0.65
      ? 'medium'
      : 'low';
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
    methodId: 'gpt-oss-phonetic-judge-v1',
    evidence: (judged.observations ?? []).map((observation, index) => ({
      id: `phonetic-observation-${index}`,
      kind: 'phoneticComparison',
      claim:
        `Esperado: ${observation.expected}. Observado: ${observation.observed}. ` +
        observation.explanation,
      expected: observation.expected,
      observed: observation.observed,
      affectsIntelligibility: observation.affectsIntelligibility,
      source: 'wav2vec2-local+gpt-oss',
    })),
    limitations: [
      judged.rationale,
      'Resultado provisional: la secuencia IPA local y el juez todavía requieren calibración humana.',
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
      methodId: 'gpt-oss-double-judge',
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
      methodId: 'gpt-oss-double-judge',
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
    methodId: 'gpt-oss-double-judge',
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
  linguisticModel,
  whisperModel,
  analyzeSpeech,
  phoneticEvidence,
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
        annotatedTranscript: null,
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
  let speechEvidence = {
    pauses: { status: 'unavailable', items: null },
    elongations: { status: 'unavailable', items: null },
    annotatedTranscript: null,
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
        pauses: {
          status: legacy ? 'complete' : 'unavailable',
          items: legacy?.pauses ?? null,
        },
        elongations: {
          status: legacy ? 'complete' : 'unavailable',
          items: legacy?.elongations ?? null,
        },
        annotatedTranscript: legacy?.annotatedTranscript ?? null,
        method: legacy?.method ?? null,
      };
    } catch (error) {
      speechEvidence.error = {
        code: error?.code ?? 'SPEECH_EVIDENCE_ERROR',
        message: error?.message ?? 'No se pudo medir pausas y duraciones.',
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
          client: groqClient,
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
        client: groqClient,
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
        client: groqClient,
        model: linguisticModel,
        rubric,
        transcript: whisperEvidence.text,
        phoneticEvidence,
      });
    } catch (error) {
      pronunciationError = {
        code: error?.code ?? 'PHONETIC_JUDGE_ERROR',
        message:
          error?.message ??
          'El juez fonético no pudo completar la comparación.',
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
    fluency: unavailableAcoustic('fluency', scopes.fluency),
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
        client: groqClient,
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
          provider: 'groq',
          model: linguisticModel,
          promptVersion: PROMPT_VERSION,
          doubleJudge: true,
          adjudicated: judging?.adjudicated ?? false,
          parameters: {
            temperature: 0,
            reasoningEffort: 'low',
            responseFormat: 'json_schema_strict',
          },
        },
        phonetic: {
          provider: 'wav2vec2-local',
          model: phoneticEvidence?.model ?? null,
          status: phoneticEvidence?.transcript
            ? 'complete'
            : 'unavailable',
          confidence: phoneticEvidence?.confidence ?? null,
          nativeLanguage: rubric.spec.nativeLanguage,
          judgeModel: linguisticModel,
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
        judging: pronunciationJudging,
        error: pronunciationError,
      },
    },
    _private: {
      originalAudio,
      normalizedAudio,
    },
  };
}
