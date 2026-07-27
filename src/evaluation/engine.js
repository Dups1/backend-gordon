import {
  CALIBRATION_VERSION,
  DIMENSION_IDS,
  PROMPT_VERSION,
  SCHEMA_VERSION,
  SCORE_PROFILES,
  dimensionResult,
  calculateOverall,
  finiteScore,
  newAssessmentId,
  normalizeTokens,
  probabilitiesForScore,
} from './domain.js';
import {
  extractLinguisticEvidence,
  generatePedagogicalFeedback,
  runDoubleLinguisticJudging,
} from './linguistic.js';
import {
  evaluateAzureV2,
  transcriptComparison,
  transcribeWhisper,
} from './providers.js';
import {
  PROMPT_EXAMPLES_STATUS,
  PROMPT_MANIFEST_HASH,
} from './prompts.js';

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

function evidenceForAzure(azure, id) {
  if (!azure) return [];
  if (id === 'pronunciation') {
    return (azure.words ?? [])
      .filter((word) => finiteScore(word.accuracyScore) !== null)
      .sort((a, b) => a.accuracyScore - b.accuracyScore)
      .slice(0, 8)
      .map((word, index) => ({
        id: `azure-word-${index}`,
        kind: 'word',
        claim: `Realización acústica de “${word.word}”.`,
        metric: 'accuracyScore',
        value: word.accuracyScore,
        unit: '0-100',
        startSec: word.start,
        endSec:
          word.start !== null && word.duration !== null
            ? word.start + word.duration
            : null,
        source: 'azure-speech',
        errorType: word.errorType,
        phonemes: word.phonemes,
      }));
  }
  return [
    {
      id: 'azure-fluency',
      kind: 'acousticMetric',
      claim: 'Fluidez acústica medida por Azure Speech.',
      metric: 'fluencyScore',
      value: azure.fluencyScore,
      unit: '0-100',
      source: 'azure-speech',
    },
    ...(finiteScore(azure.prosodyScore) !== null
      ? [
          {
            id: 'azure-prosody',
            kind: 'acousticMetric',
            claim: 'Prosodia medida por Azure Speech.',
            metric: 'prosodyScore',
            value: azure.prosodyScore,
            unit: '0-100',
            source: 'azure-speech',
          },
        ]
      : []),
  ];
}

function unavailableAcoustic(id, error, constructScope) {
  const providerRejected = error?.code === 'INVALID_PROVIDER_SCORE';
  return dimensionResult({
    id,
    status: providerRejected ? 'providerError' : 'unavailable',
    methodId: 'azure-pronunciation-assessment',
    reliability: 'unknown',
    reasonCode: error?.code ?? 'AZURE_UNAVAILABLE',
    limitations: [
      error?.message ??
        'Azure Speech no entregó evidencia para esta dimensión.',
    ],
    constructScope,
    reviewRequired: true,
  });
}

function acousticDimension({
  id,
  score,
  rawScore,
  azure,
  quality,
  transcriptReview,
  constructScope,
}) {
  const valid = finiteScore(score);
  if (score !== null && score !== undefined && valid === null) {
    return dimensionResult({
      id,
      status: 'providerError',
      methodId: 'azure-pronunciation-assessment',
      reliability: 'unknown',
      reasonCode: 'INVALID_PROVIDER_SCORE',
      limitations: [
        `Azure devolvió un valor inválido para ${id}; no fue limitado ni reinterpretado.`,
      ],
      constructScope,
      reviewRequired: true,
    });
  }
  if (valid === null) {
    return unavailableAcoustic(
      id,
      {
        code: 'AZURE_SCORE_MISSING',
        message: `Azure no entregó una puntuación válida de ${id}.`,
      },
      constructScope,
    );
  }
  const lowEvidence =
    quality.warnings.includes('INSUFFICIENT_ACOUSTIC_SAMPLE') ||
    quality.warnings.includes('INSUFFICIENT_READING_SAMPLE');
  if (lowEvidence) {
    return dimensionResult({
      id,
      status: 'insufficientEvidence',
      methodId: 'azure-provisional-calibration',
      reliability: 'low',
      rawScore,
      evidence: evidenceForAzure(azure, id),
      limitations: ['La muestra no alcanza la duración acústica recomendada.'],
      reasonCode: 'INSUFFICIENT_ACOUSTIC_SAMPLE',
      constructScope,
      reviewRequired: true,
    });
  }
  const reviewRequired =
    quality.status === 'warning' || transcriptReview === true;
  const halfWidth = reviewRequired ? 15 : 10;
  return dimensionResult({
    id,
    status: 'scored',
    score: valid,
    rawScore,
    probabilities: probabilitiesForScore(valid),
    interval90: {
      low: Math.max(0, valid - halfWidth),
      high: Math.min(100, valid + halfWidth),
    },
    reliability: reviewRequired ? 'low' : 'medium',
    methodId: 'azure-provisional-calibration',
    evidence: evidenceForAzure(azure, id),
    limitations: [
      'Las probabilidades de banda son una proyección ordinal provisional del score de Azure, no una calibración humana final.',
      'Resultado provisional pendiente de calibración con evaluadores humanos.',
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
  azureConfig,
  evaluateAzureRest,
  evaluateAzureContinuous,
  analyzeSpeech,
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
  let azure = null;
  let azureError = null;
  try {
    azure = await evaluateAzureV2({
      normalizedPath: normalizedAudio,
      durationSeconds: quality.metrics.durationSeconds,
      rubric,
      whisper: whisperEvidence,
      quality,
      azureConfig,
      evaluateRest: evaluateAzureRest,
      evaluateContinuous: evaluateAzureContinuous,
    });
  } catch (error) {
    azureError = {
      code: error?.code ?? 'AZURE_PROVIDER_ERROR',
      message:
        error?.message ??
        'Azure Speech no pudo completar la evaluación acústica.',
    };
  }
  const comparedTranscript = transcriptComparison(whisperEvidence, azure);
  const transcript = {
    ...comparedTranscript,
    primary: {
      ...comparedTranscript.primary,
      status: whisper ? 'complete' : 'unavailable',
      error: whisperError,
    },
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
        pronunciacion: azure,
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
          secondaryTranscript: transcript.secondary?.text ?? '',
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
    pronunciation: azure
      ? acousticDimension({
          id: 'pronunciation',
          score: azure.accuracyScore ?? azure.pronunciationScore,
          rawScore: azure.pronunciationScore,
          azure,
          quality,
          transcriptReview: transcript.reviewRequired,
          constructScope: scopes.pronunciation,
        })
      : unavailableAcoustic('pronunciation', azureError, scopes.pronunciation),
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
    fluency: azure
      ? acousticDimension({
          id: 'fluency',
          score: azure.fluencyScore,
          rawScore: azure.fluencyScore,
          azure,
          quality,
          transcriptReview: transcript.reviewRequired,
          constructScope: scopes.fluency,
        })
      : unavailableAcoustic('fluency', azureError, scopes.fluency),
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
          },
        },
        azure: azure
          ? {
              provider: 'azure-speech',
              locale: rubric.spec.targetLocale,
              mode: rubric.spec.mode,
              region: azureConfig?.region ?? null,
              requestId: azure.requestId ?? null,
              aggregation: azure.aggregation ?? { method: 'single' },
              parameters: {
                referenceText:
                  rubric.spec.mode === 'reading' ? 'canonical' : 'none',
                granularity: 'phoneme',
                prosody: rubric.spec.targetLocale === 'en-US',
                recognitionMode: azure.recognitionMode,
                continuous: azure.recognitionMode === 'continuous',
                continuousThresholdSeconds:
                  azure.recognitionThresholdSeconds,
              },
            }
          : { provider: 'azure-speech', error: azureError },
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
      azure,
      linguistic: {
        evidence: linguisticEvidence,
        judging,
        error: linguisticError,
      },
    },
    _private: {
      originalAudio,
      normalizedAudio,
    },
  };
}
