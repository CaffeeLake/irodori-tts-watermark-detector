import { execFileSync } from 'node:child_process';
import path from 'node:path';
import * as ort from 'onnxruntime-web';

import { getVariant } from '../web/src/detector.js';

export const ROOT = path.resolve(import.meta.dirname, '..');
export const MODELS = path.join(ROOT, 'web', 'models');

/** Decode any media file to mono float32 PCM at `rate` using ffmpeg. */
export function decodePcm(file, rate = 48000) {
  const buf = execFileSync('ffmpeg', [
    '-v', 'error', '-i', file, '-f', 'f32le', '-ac', '1', '-ar', String(rate), '-',
  ], { maxBuffer: 1 << 30 });
  return new Float32Array(buf.buffer, buf.byteOffset, buf.length / 4);
}

/** Encode mono float32 PCM as a 32-bit float WAV (handy for Opus round trips). */
export function wavBytes(pcm, rate) {
  const n = pcm.length;
  const b = Buffer.alloc(44 + n * 4);
  b.write('RIFF', 0); b.writeUInt32LE(36 + n * 4, 4); b.write('WAVEfmt ', 8);
  b.writeUInt32LE(16, 16); b.writeUInt16LE(3, 20); b.writeUInt16LE(1, 22);
  b.writeUInt32LE(rate, 24); b.writeUInt32LE(rate * 4, 28);
  b.writeUInt16LE(4, 32); b.writeUInt16LE(32, 34);
  b.write('data', 36); b.writeUInt32LE(n * 4, 40);
  for (let i = 0; i < n; i++) b.writeFloatLE(pcm[i], 44 + i * 4);
  return b;
}

let envReady = false;
const sessions = new Map();

/** Lazily create (and cache) the ONNX session for a SilentCipher variant. */
export function getSession(variant = '44.1k') {
  if (!envReady) {
    ort.env.wasm.wasmPaths = path.join(ROOT, 'node_modules', 'onnxruntime-web', 'dist') + path.sep;
    ort.env.wasm.numThreads = Number(process.env.THREADS || 4);
    ort.env.logLevel = 'error';
    envReady = true;
  }
  if (!sessions.has(variant)) {
    const cfg = getVariant(variant);
    sessions.set(variant, ort.InferenceSession.create(path.join(MODELS, cfg.model), {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    }));
  }
  return sessions.get(variant);
}

export const Tensor = ort.Tensor;
