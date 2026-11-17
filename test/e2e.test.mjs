/**
 * Negative controls and invariants for the 44.1 kHz detector, driven by the generated
 * sample audio in test/samples/ (see tools/make_samples.py).
 *
 * Positive detection and cross-variant behaviour live in test/variants.test.mjs; this
 * file covers the things that need synthesised or derived signals: noise, tones,
 * time-reversal, and the guarantee that chunking does not change the answer.
 *
 *   node --test test/e2e.test.mjs
 *   FULL=1 node --test test/e2e.test.mjs   # analyse whole files instead of a window
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import path from 'node:path';

import { prepare, runDetector } from '../web/src/pipeline.js';
import { DEFAULTS } from '../web/src/detector.js';
import { ROOT, decodePcm, getSession, Tensor } from './helpers.mjs';

const SAMPLES = path.join(ROOT, 'test', 'samples');
const WATERMARKED = path.join(SAMPLES, '44k1_watermarked.wav');
const haveSamples = existsSync(WATERMARKED);
if (!haveSamples) {
  console.log('    note: test/samples is empty — run `npm run samples` first.');
}

const FULL = process.env.FULL === '1';
const windowFrames = (sec) => Math.round((sec * DEFAULTS.sampleRate) / DEFAULTS.hopLength);
const SECONDS = Number(process.env.SECONDS || 10);

async function analyse(pcm, rate, opts = {}) {
  const spec = prepare(pcm, rate);
  const session = await getSession();
  const { result } = await runDetector(spec, session, Tensor, {
    chunkFrames: 43,
    maxFrames: FULL ? spec.frames : Math.min(spec.frames, windowFrames(SECONDS)),
    earlyStop: false,
    inputName: session.inputNames[0],
    outputName: session.outputNames[0],
    ...opts,
  });
  return result;
}

const fmt = (r) => `verdict=${r.verdict} match=${r.match.toFixed(3)} null=${r.nullMatch.toFixed(3)} `
  + `margin=${r.margin.toFixed(3)} p=${r.pValue.toExponential(2)} struct=${r.structure.toFixed(3)} `
  + `frames=${r.frames} shift=${r.shift} blind=${r.blind.ascii}`;

test('does not fire on white noise', { timeout: 1800000 }, async () => {
  let seed = 7;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
  const pcm = Float32Array.from({ length: 48000 * (SECONDS + 2) }, () => 0.1 * rnd());
  const r = await analyse(pcm, 48000);
  console.log(`    white noise: ${fmt(r)}`);
  assert.notEqual(r.verdict, 'detected');
  assert.ok(!r.structurePresent, `structure ${r.structure}`);
});

test('does not fire on a pure tone (constant decoder output)', { timeout: 1800000 }, async () => {
  const n = 48000 * (SECONDS + 2);
  const pcm = Float32Array.from({ length: n }, (_, i) => 0.3 * Math.sin((2 * Math.PI * 440 * i) / 48000));
  const r = await analyse(pcm, 48000);
  console.log(`    sine 440: ${fmt(r)}`);
  assert.notEqual(r.verdict, 'detected');
  assert.ok(!r.structurePresent, `structure ${r.structure} — constant output must not look periodic`);
});

test('does not fire on the unwatermarked carrier', { timeout: 1800000, skip: !haveSamples }, async () => {
  const r = await analyse(decodePcm(path.join(SAMPLES, '44k1_clean.wav'), 48000), 48000);
  console.log(`    clean carrier: ${fmt(r)}`);
  assert.notEqual(r.verdict, 'detected');
  assert.notEqual(r.blind.ascii, 'IRDTS');
});

test('does not fire on time-reversed watermarked audio',
  { timeout: 1800000, skip: !haveSamples }, async () => {
    const rev = Float32Array.from(decodePcm(WATERMARKED, 48000)).reverse();
    const r = await analyse(rev, 48000);
    console.log(`    reversed: ${fmt(r)}`);
    assert.notEqual(r.verdict, 'detected');
    // Reversal preserves SilentCipher-like periodic structure but destroys the payload.
    assert.notEqual(r.blind.ascii, 'IRDTS');
  });

test('chunked inference equals a single full-length forward pass',
  { timeout: 1800000, skip: !haveSamples }, async () => {
    const pcm = decodePcm(WATERMARKED, 48000).subarray(0, 48000 * 4);
    const spec = prepare(pcm, 48000);
    const session = await getSession();
    const common = {
      maxFrames: spec.frames,
      earlyStop: false,
      inputName: session.inputNames[0],
      outputName: session.outputNames[0],
    };
    const a = await runDetector(spec, session, Tensor, { ...common, chunkFrames: spec.frames });
    const b = await runDetector(spec, session, Tensor, { ...common, chunkFrames: 11 });
    assert.equal(a.result.frames, b.result.frames);
    // The halo context makes chunking exact up to float32 conv-tiling non-determinism.
    assert.ok(Math.abs(a.result.match - b.result.match) < 1e-9,
      `${a.result.match} vs ${b.result.match}`);
    assert.ok(Math.abs(a.result.logLik - b.result.logLik) < 1e-4,
      `${a.result.logLik} vs ${b.result.logLik}`);
    assert.equal(a.result.shift, b.result.shift);
    assert.equal(a.result.blind.ascii, b.result.blind.ascii);
  });

test('the front end only transforms the frames it needs', { timeout: 1800000, skip: !haveSamples }, async () => {
  // A verdict needs ~129 frames, so transforming a long file end to end is pure waste:
  // measured on 341 s of audio the front end cost 7.2 s instead of 0.2 s. Capping it
  // must not change a single number.
  const pcm = decodePcm(WATERMARKED, 48000);
  const budget = DEFAULTS.thresholds.minFrames;

  const full = prepare(pcm, 48000, DEFAULTS);
  const capped = prepare(pcm, 48000, DEFAULTS, { maxFrames: budget });
  assert.equal(full.truncated, false);
  assert.equal(capped.truncated, true, 'the sample is long enough to be capped');
  assert.ok(capped.frames >= budget + DEFAULTS.halo,
    `kept ${capped.frames} frames, need ${budget} + ${DEFAULTS.halo} halo`);
  assert.ok(capped.frames < full.frames, `${capped.frames} vs ${full.frames}`);
  assert.equal(capped.fileFrames, full.fileFrames, 'the reported file length must not change');

  // Every frame we will actually use must be bit-identical to the full transform.
  const bands = DEFAULTS.messageBandSize;
  const used = budget + DEFAULTS.halo;
  for (let band = 0; band < bands; band += 97) {
    for (let f = 0; f < used; f += 13) {
      assert.equal(capped.mag[band * capped.frames + f], full.mag[band * full.frames + f],
        `band ${band} frame ${f}`);
    }
  }

  const session = await getSession();
  const common = {
    chunkFrames: 43, maxFrames: budget, earlyStop: false,
    inputName: session.inputNames[0], outputName: session.outputNames[0],
  };
  const a = (await runDetector(full, session, Tensor, common)).result;
  const b = (await runDetector(capped, session, Tensor, common)).result;
  console.log(`    full ${full.frames} fr -> capped ${capped.frames} fr; `
    + `margin ${a.margin.toFixed(9)} vs ${b.margin.toFixed(9)}`);
  assert.equal(a.verdict, b.verdict);
  assert.equal(a.match, b.match);
  assert.equal(a.margin, b.margin);
  assert.equal(a.structure, b.structure);
  assert.equal(a.shift, b.shift);
  assert.equal(a.blind.ascii, b.blind.ascii);
});

test('analysing the whole file recovers the payload byte-exactly',
  { timeout: 1800000, skip: !(haveSamples && FULL) }, async () => {
    const r = await analyse(decodePcm(WATERMARKED, 48000), 48000);
    console.log(`    full file: ${fmt(r)}`);
    assert.equal(r.verdict, 'detected');
    assert.deepEqual(r.blind.message, DEFAULTS.payload);
    assert.equal(r.blind.ascii, 'IRDTS');
    assert.equal(r.payloadExact, true);
    assert.equal(r.symbolAgreement, DEFAULTS.messageLen);
  });
