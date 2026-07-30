import { EvaluationError } from './domain.js';

function unavailable() {
  throw new EvaluationError(
    503,
    'El almacenamiento consentido está deshabilitado hasta configurar un proveedor nuevo.',
    'PILOT_STORAGE_NOT_CONFIGURED',
  );
}

export function createPilotStorage() {
  return {
    configured: false,
    container: null,
    saveAssessment: unavailable,
    saveHumanRating: unavailable,
    deleteAssessment: unavailable,
    async purgeExpiredAudio() {
      return { deleted: 0 };
    },
  };
}
