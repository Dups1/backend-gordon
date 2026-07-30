import assert from 'node:assert/strict';
import test from 'node:test';

import {
  circuitSnapshot,
  withCircuitBreaker,
} from '../src/evaluation/resilience.js';

function statusError(status) {
  const error = new Error(`provider status ${status}`);
  error.status = status;
  return error;
}

test('el circuito acumula fallos transitorios concurrentes sin perder incrementos', async () => {
  const provider = `concurrent-408-${Date.now()}`;
  const results = await Promise.allSettled(
    Array.from({ length: 5 }, () =>
      withCircuitBreaker(
        provider,
        async () => {
          throw statusError(408);
        },
        { failureThreshold: 5 },
      ),
    ),
  );

  assert.ok(results.every((result) => result.status === 'rejected'));
  assert.deepEqual(circuitSnapshot()[provider], {
    failures: 5,
    open: true,
  });
});

test('un límite 429 no se interpreta como caída del proveedor', async () => {
  const provider = `rate-limit-${Date.now()}`;
  for (let attempt = 0; attempt < 6; attempt++) {
    await assert.rejects(
      withCircuitBreaker(provider, async () => {
        throw statusError(429);
      }),
    );
  }

  assert.deepEqual(circuitSnapshot()[provider], {
    failures: 0,
    open: false,
  });
});
