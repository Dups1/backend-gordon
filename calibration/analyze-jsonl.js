import process from 'node:process';
import path from 'node:path';

import {
  analyzeCalibrationRecords,
  readCalibrationJsonl,
} from '../src/evaluation/calibration.js';

async function main() {
  const input = process.argv[2];
  if (!input || input === '--help' || input === '-h') {
    process.stderr.write(
      'Uso: npm run calibration:analyze -- /ruta/al/dataset.jsonl\n',
    );
    process.exitCode = input ? 0 : 2;
    return;
  }

  try {
    const records = await readCalibrationJsonl(path.resolve(input));
    const report = analyzeCalibrationRecords(records);
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  } catch (error) {
    process.stderr.write(
      `No fue posible analizar la calibración: ${error.message}\n`,
    );
    process.exitCode = 1;
  }
}

await main();
