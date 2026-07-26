import { readFile } from 'node:fs/promises';

import { EvaluationError } from './domain.js';

function blobConfiguration(override = {}) {
  const retentionDays = Number.parseInt(
    override.retentionDays ??
      process.env.AUDIO_RETENTION_DAYS ??
      '90',
    10,
  );
  return {
    connectionString:
      override.connectionString ??
      process.env.AZURE_STORAGE_CONNECTION_STRING?.trim() ??
      '',
    container:
      override.container ??
      process.env.AZURE_STORAGE_CONTAINER?.trim() ??
      'gordon-pilot-private',
    retentionDays:
      Number.isFinite(retentionDays) && retentionDays > 0
        ? Math.min(retentionDays, 3650)
        : 90,
  };
}

async function containerClient(configuration) {
  if (!configuration.connectionString) return null;
  let sdk;
  try {
    sdk = await import('@azure/storage-blob');
  } catch {
    throw new EvaluationError(
      503,
      'El cliente de Azure Blob Storage no está instalado.',
      'PILOT_STORAGE_SDK_UNAVAILABLE',
    );
  }
  const service =
    sdk.BlobServiceClient.fromConnectionString(configuration.connectionString);
  const container = service.getContainerClient(configuration.container);
  await container.createIfNotExists({ access: undefined });
  const properties = await container.getProperties();
  if (properties.blobPublicAccess) {
    await container.setAccessPolicy(undefined);
  }
  return container;
}

function metadata(record) {
  return {
    assessmentId: record.assessmentId,
    schemaVersion: String(record.report.schemaVersion),
    locale: String(record.report.taskSnapshot.targetLocale),
    mode: String(record.report.taskSnapshot.mode),
    cefr: String(record.report.taskSnapshot.cefr),
    consentVersion: String(record.consent.version),
  };
}

function audioMetadata(record, retentionDays) {
  return {
    ...metadata(record),
    retentionClass: 'audio',
    deleteAfterUtc: new Date(
      Date.now() + retentionDays * 24 * 60 * 60 * 1000,
    ).toISOString(),
  };
}

export function createPilotStorage(override) {
  const configuration = blobConfiguration(override);
  return {
    configured: Boolean(configuration.connectionString),
    container: configuration.container,

    async saveAssessment(record) {
      const container = await containerClient(configuration);
      if (!container) {
        throw new EvaluationError(
          503,
          'El almacenamiento consentido no está configurado.',
          'PILOT_STORAGE_NOT_CONFIGURED',
        );
      }
      const prefix = `${record.assessmentId}/`;
      const common = {
        metadata: metadata(record),
        conditions: { ifNoneMatch: '*' },
      };
      const audioCommon = {
        ...common,
        metadata: audioMetadata(record, configuration.retentionDays),
      };
      const originalAudio = await readFile(record.originalAudioPath);
      const normalizedAudio = await readFile(record.normalizedAudioPath);
      const reportBuffer = Buffer.from(JSON.stringify(record.report));
      const providersBuffer = Buffer.from(JSON.stringify(record.providers));
      const consentBuffer = Buffer.from(JSON.stringify(record.consent));
      await Promise.all([
        container
          .getBlockBlobClient(`${prefix}audio-original`)
          .uploadData(originalAudio, {
            ...audioCommon,
            blobHTTPHeaders: {
              blobContentType:
                record.originalContentType || 'application/octet-stream',
            },
          }),
        container
          .getBlockBlobClient(`${prefix}audio-normalized.wav`)
          .uploadData(normalizedAudio, {
            ...audioCommon,
            blobHTTPHeaders: { blobContentType: 'audio/wav' },
          }),
        container
          .getBlockBlobClient(`${prefix}report.json`)
          .upload(reportBuffer, reportBuffer.length, {
            ...common,
            blobHTTPHeaders: { blobContentType: 'application/json' },
          }),
        container
          .getBlockBlobClient(`${prefix}providers.json`)
          .upload(
            providersBuffer,
            providersBuffer.length,
            {
              ...common,
              blobHTTPHeaders: { blobContentType: 'application/json' },
            },
          ),
        container
          .getBlockBlobClient(`${prefix}consent.json`)
          .upload(
            consentBuffer,
            consentBuffer.length,
            {
              ...common,
              blobHTTPHeaders: { blobContentType: 'application/json' },
            },
          ),
      ]);
      return {
        stored: true,
        recordId: record.assessmentId,
        container: configuration.container,
        publicUrl: null,
        audioRetentionDays: configuration.retentionDays,
      };
    },

    async purgeExpiredAudio({ now = Date.now() } = {}) {
      const container = await containerClient(configuration);
      if (!container) return { deleted: 0 };
      let deleted = 0;
      for await (const blob of container.listBlobsFlat({
        includeMetadata: true,
      })) {
        const metadata = blob.metadata ?? {};
        if (
          (metadata.retentionClass ?? metadata.retentionclass) !== 'audio'
        ) {
          continue;
        }
        const expiry = Date.parse(
          metadata.deleteAfterUtc ?? metadata.deleteafterutc ?? '',
        );
        if (!Number.isFinite(expiry) || expiry > now) continue;
        const result = await container.deleteBlob(blob.name, {
          deleteSnapshots: 'include',
        });
        if (result.succeeded) deleted++;
      }
      return { deleted };
    },

    async saveHumanRating(assessmentId, rating) {
      const container = await containerClient(configuration);
      if (!container) {
        throw new EvaluationError(
          503,
          'El almacenamiento consentido no está configurado.',
          'PILOT_STORAGE_NOT_CONFIGURED',
        );
      }
      const consentBlob = container.getBlobClient(
        `${assessmentId}/consent.json`,
      );
      if (!(await consentBlob.exists())) {
        throw new EvaluationError(
          409,
          'No existe consentimiento almacenado para esta evaluación.',
          'PILOT_CONSENT_NOT_FOUND',
        );
      }
      const name = `${assessmentId}/human-ratings/${rating.ratingId}.json`;
      const serialized = Buffer.from(JSON.stringify(rating));
      await container.getBlockBlobClient(name).upload(serialized, serialized.length, {
        conditions: { ifNoneMatch: '*' },
        blobHTTPHeaders: { blobContentType: 'application/json' },
      });
      return { stored: true, ratingId: rating.ratingId };
    },

    async deleteAssessment(assessmentId) {
      const container = await containerClient(configuration);
      if (!container) {
        throw new EvaluationError(
          503,
          'El almacenamiento consentido no está configurado.',
          'PILOT_STORAGE_NOT_CONFIGURED',
        );
      }
      let deleted = 0;
      for await (const blob of container.listBlobsFlat({
        prefix: `${assessmentId}/`,
      })) {
        const result = await container.deleteBlob(blob.name, {
          deleteSnapshots: 'include',
        });
        if (result.succeeded) deleted++;
      }
      return { deleted: true, deletedObjects: deleted };
    },
  };
}
