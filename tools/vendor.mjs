#!/usr/bin/env node
/** Copy the onnxruntime-web runtime into web/vendor/ so the page needs no bundler. */
import { cp, mkdir, readdir } from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const src = path.join(root, 'node_modules', 'onnxruntime-web', 'dist');
const dst = path.join(root, 'web', 'vendor');

// Only the plain (non-jsep) SIMD+threads build is needed: inference runs on the
// WebAssembly execution provider, so the 29 MB WebGPU-enabled binary is skipped.
const WANTED = [
  'ort.wasm.min.mjs',
  'ort-wasm-simd-threaded.mjs',
  'ort-wasm-simd-threaded.wasm',
];

await mkdir(dst, { recursive: true });
const have = new Set(await readdir(src));
for (const f of WANTED) {
  if (!have.has(f)) {
    console.warn(`skip (missing): ${f}`);
    continue;
  }
  await cp(path.join(src, f), path.join(dst, f));
  console.log(`vendored ${f}`);
}
