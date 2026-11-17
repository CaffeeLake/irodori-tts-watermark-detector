/**
 * Arbitrary payloads.
 *
 * The detection statistic scores one rotation of the expected word against the other
 * rotations, so its null distribution depends on `messageLen` and `messageDim` — not on
 * which word is being looked for. The shipped thresholds should therefore work for any
 * payload, which is what these tests check against a sample watermarked with "HELLO".
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

import { prepare, runDetector } from '../web/src/pipeline.js';
import {
  VARIANTS, getVariant, withPayload, parsePayload, payloadCapacity, bytesToAscii,
  Accumulator, payloadToSymbols,
} from '../web/src/detector.js';
import { ROOT, decodePcm, getSession, Tensor } from './helpers.mjs';

const SAMPLES = path.join(ROOT, 'test', 'samples');
const HELLO = path.join(SAMPLES, '44k1_hello_watermarked.wav');
const IRDTS = path.join(SAMPLES, '44k1_watermarked.wav');
const have = existsSync(HELLO) && existsSync(IRDTS);
if (!have) console.log('    note: run `npm run samples` (and the HELLO sample) first.');

const V44 = VARIANTS['44.1k'];

async function analyse(file, payload) {
  const cfg = withPayload(getVariant('44.1k'), payload);
  const budget = cfg.thresholds.minFrames;
  const spec = prepare(decodePcm(file, 48000), 48000, cfg, { maxFrames: budget });
  const session = await getSession('44.1k');
  const { result } = await runDetector(spec, session, Tensor, {
    cfg, chunkFrames: 43, maxFrames: Math.min(spec.frames, budget), earlyStop: false,
    inputName: session.inputNames[0], outputName: session.outputNames[0],
  });
  return result;
}

/* ------------------------------------------------------------- parsing ------ */

test('payload parsing accepts text, decimal and hex', () => {
  const want = [73, 82, 68, 84, 83];
  for (const form of ['IRDTS', '73,82,68,84,83', '73 82 68 84 83', '0x49 0x52 0x44 0x54 0x53']) {
    const r = parsePayload(form, V44);
    assert.ok(r.ok, `${form}: ${r.error}`);
    assert.deepEqual(r.bytes, want, form);
  }
});

test('payload parsing rejects the wrong length and out-of-range bytes', () => {
  const cap = payloadCapacity(V44);
  assert.equal(cap.bytes, 5);
  for (const bad of ['', 'ABC', 'TOOLONG', '1,2,3', '300,1,2,3,4', '-1,1,2,3,4']) {
    const r = parsePayload(bad, V44);
    assert.equal(r.ok, false, `${bad} should be rejected`);
    assert.ok(r.error && r.error.length > 0);
  }
});

test('the 16 kHz variant takes a symbol word, not text', () => {
  const v16 = VARIANTS['16k'];
  const cap = payloadCapacity(v16);
  assert.equal(cap.bytes, null, 'no whole-byte payload fits');
  assert.equal(parsePayload('IRDTS', v16).ok, false);
  const word = new Array(cap.slots).fill(0).map((_, i) => i % (cap.maxSymbol + 1));
  const r = parsePayload(word.join(','), v16);
  assert.ok(r.ok, r.error);
  assert.deepEqual(r.symbols, word);
  assert.equal(r.word.length, v16.messageLen);
  assert.equal(r.word[r.word.length - 1], 0, 'terminator');
  // One past the allowed range must be refused: the one-hot index would overflow.
  assert.equal(parsePayload(word.map(() => cap.maxSymbol + 1).join(','), v16).ok, false);
});

test('withPayload swaps the word the accumulator looks for', () => {
  const cfg = withPayload(V44, 'HELLO');
  assert.deepEqual(cfg.payload, [72, 69, 76, 76, 79]);
  assert.deepEqual(cfg.expectedWord, payloadToSymbols([72, 69, 76, 76, 79], V44.messageLen));
  assert.deepEqual(new Accumulator(cfg).expected, cfg.expectedWord);
  // The default is untouched.
  assert.deepEqual(new Accumulator(V44).expected, payloadToSymbols(V44.payload, V44.messageLen));
  assert.throws(() => withPayload(V44, 'NOPE'), /exactly 5 bytes/);
});

/* --------------------------------------------------------- end to end ------- */

test('detects a custom payload with the shipped thresholds',
  { timeout: 1800000, skip: !have }, async () => {
    const th = V44.thresholds;
    const r = await analyse(HELLO, 'HELLO');
    console.log(`    HELLO sample, searching HELLO: verdict=${r.verdict} `
      + `margin=${r.margin.toFixed(3)} p=${r.pValue.toExponential(2)} blind=${r.blind.ascii}`);
    assert.equal(r.verdict, 'detected');
    assert.ok(r.margin >= th.detectMargin, `margin ${r.margin}`);
    assert.ok(r.pValue <= th.detectP, `p ${r.pValue}`);
    assert.equal(r.blind.ascii, 'HELLO');
    assert.equal(r.payloadExact, true);
    assert.equal(r.expectedAscii, 'HELLO');
    assert.equal(r.customPayload, true);
  });

test('a watermark carrying a different payload is not reported as a match',
  { timeout: 1800000, skip: !have }, async () => {
    // The HELLO sample really is watermarked, so structure is present; but looking for
    // the Irodori payload must not come back "detected".
    const r = await analyse(HELLO, null);
    console.log(`    HELLO sample, searching IRDTS: verdict=${r.verdict} `
      + `margin=${r.margin.toFixed(3)} struct=${r.structure.toFixed(3)} blind=${r.blind.ascii}`);
    assert.notEqual(r.verdict, 'detected');
    assert.equal(r.expectedAscii, bytesToAscii(V44.payload));
    assert.equal(r.customPayload, false);
    assert.ok(r.structurePresent, 'a watermark IS present, just not this payload');
    assert.equal(r.blind.ascii, 'HELLO', 'and the blind decode says which');
  });

test('searching for the wrong payload on the Irodori sample also fails',
  { timeout: 1800000, skip: !have }, async () => {
    const r = await analyse(IRDTS, 'HELLO');
    console.log(`    IRDTS sample, searching HELLO: verdict=${r.verdict} `
      + `margin=${r.margin.toFixed(3)} struct=${r.structure.toFixed(3)} blind=${r.blind.ascii}`);
    assert.notEqual(r.verdict, 'detected');
    assert.ok(r.structurePresent);
    assert.equal(r.blind.ascii, 'IRDTS');
  });

test('the default payload still works unchanged', { timeout: 1800000, skip: !have }, async () => {
  const r = await analyse(IRDTS, null);
  console.log(`    IRDTS sample, default: verdict=${r.verdict} margin=${r.margin.toFixed(3)}`);
  assert.equal(r.verdict, 'detected');
  assert.equal(r.blind.ascii, 'IRDTS');
  assert.equal(r.customPayload, false);
});

test('an equivalent payload spelling gives identical results',
  { timeout: 1800000, skip: !have }, async () => {
    const a = await analyse(HELLO, 'HELLO');
    const b = await analyse(HELLO, '72,69,76,76,79');
    assert.equal(a.margin, b.margin);
    assert.equal(a.pValue, b.pValue);
    assert.equal(a.shift, b.shift);
    assert.deepEqual(a.expectedPayload, b.expectedPayload);
  });
