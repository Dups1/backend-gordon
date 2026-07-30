import { EvaluationError } from './domain.js';

const circuits = new Map();

export async function withCircuitBreaker(
  provider,
  operation,
  { failureThreshold = 5, coolDownMs = 60_000 } = {},
) {
  const now = Date.now();
  const state = circuits.get(provider) ?? {
    failures: 0,
    openedAt: null,
  };
  circuits.set(provider, state);
  if (
    state.openedAt !== null &&
    now - state.openedAt < coolDownMs
  ) {
    throw new EvaluationError(
      503,
      `${provider} está temporalmente aislado tras errores consecutivos.`,
      'PROVIDER_CIRCUIT_OPEN',
      { provider, retryAfterMs: coolDownMs - (now - state.openedAt) },
    );
  }
  if (state.openedAt !== null) {
    state.openedAt = null;
    state.failures = 0;
  }
  try {
    const result = await operation();
    circuits.set(provider, { failures: 0, openedAt: null });
    return result;
  } catch (error) {
    const providerStatus = Number.isInteger(error?.details?.providerStatus)
      ? error.details.providerStatus
      : error?.status;
    const providerFailure =
      !Number.isInteger(providerStatus) ||
      providerStatus >= 500 ||
      providerStatus === 408;
    if (providerFailure) {
      state.failures++;
      if (state.failures >= failureThreshold) state.openedAt = Date.now();
      circuits.set(provider, state);
    }
    throw error;
  }
}

export function circuitSnapshot() {
  return Object.fromEntries(
    [...circuits.entries()].map(([provider, state]) => [
      provider,
      {
        failures: state.failures,
        open: state.openedAt !== null,
      },
    ]),
  );
}
