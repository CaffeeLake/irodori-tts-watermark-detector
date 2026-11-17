/**
 * Whole-chain numerical parity: raw 48 kHz PCM -> resample -> STFT -> ONNX/WASM decoder
 * must reproduce the PyTorch (eval-mode) reference logits produced by
 * tools/make_fixtures.py.
 *
 * This is what pins the browser build to the reference implementation; the statistics
 * layered on top are only as trustworthy as these logits.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';

import { prepare } from '../web/src/pipeline.js';
import { getSession, Tensor } from './helpers.mjs';

const FX = path.resolve(import.meta.dirname, 'fixtures');
const manifest = JSON.parse(readFileSync(path.join(FX, 'manifest.json'), 'utf8'));
const readF32 = (p) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.length / 4);
};

/** index of the largest of `D` values stored as x[d * F + f] */
const argmaxAt = (x, D, F, f) => {
  let best = 0;
  for (let d = 1; d < D; d++) if (x[d * F + f] > x[best * F + f]) best = d;
  return best;
};

for (const c of manifest.cases) {
  test(`JS front-end + ONNX/WASM reproduces PyTorch logits (${c.name})`, { timeout: 600000 }, async () => {
    const input = readF32(path.join(FX, `${c.name}.input48k.f32`));
    const want = readF32(path.join(FX, `${c.name}.logits.f32`)); // (message_dim, frames)

    const spec = prepare(input, c.input_rate);
    assert.equal(spec.frames, c.frames, 'frame count');

    const session = await getSession();
    const out = await session.run({
      magnitude: new Tensor('float32', spec.mag, [1, 1, manifest.bands, c.frames]),
    });
    const got = out[Object.keys(out)[0]].data;
    assert.equal(got.length, want.length);

    const D = c.message_dim, F = c.frames;
    let maxAbs = 0, maxRel = 0, agree = 0;
    for (let i = 0; i < want.length; i++) {
      const d = Math.abs(got[i] - want[i]);
      maxAbs = Math.max(maxAbs, d);
      maxRel = Math.max(maxRel, d / (Math.abs(want[i]) + 1e-6));
    }
    for (let f = 0; f < F; f++) {
      if (argmaxAt(got, D, F, f) === argmaxAt(want, D, F, f)) agree++;
    }
    console.log(`    max|diff|=${maxAbs.toExponential(2)} maxRel=${maxRel.toExponential(2)} `
      + `argmax agreement=${(agree / F).toFixed(4)} (${F} frames)`);

    // What has to hold exactly is the decision: every frame must choose the symbol
    // PyTorch chose, since every statistic the detector reports is built from argmax.
    assert.equal(agree, F, 'every frame must pick the same symbol as PyTorch');

    // The logits themselves carry a small, deliberate offset.  The reference scales the
    // signal by the mean power of the *resampled* waveform; we measure it at the input
    // rate so that long files do not have to be resampled in full (see prepare()).  The
    // two powers differ by the energy above the new Nyquist frequency -- 0.077 % here,
    // i.e. 0.039 % in gain -- which moves the logits by ~2e-3 relative and was measured
    // to leave verdict, match, margin, structure, shift, p-value and the recovered
    // payload bit-identical on watermarked, Opus-coded and clean audio alike.
    assert.ok(maxRel < 5e-3, `relative logit error ${maxRel}`);
  });
}
