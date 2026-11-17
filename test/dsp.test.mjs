/**
 * Parity tests: the JS resampler / STFT must match torchaudio + silentcipher.
 * Fixtures are produced by tools/make_fixtures.py.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';

import { resampleSinc, stftMagnitude, frameCount, meanPower } from '../web/src/dsp.js';
import {
  payloadToSymbols, binomSurvival, Accumulator, DEFAULTS, THRESHOLDS, notes,
  VARIANTS, nullPeriods, verdictLabel, windowFrames, canReachVerdict, minSeconds,
} from '../web/src/detector.js';

const FX = path.resolve(import.meta.dirname, 'fixtures');
const manifest = JSON.parse(readFileSync(path.join(FX, 'manifest.json'), 'utf8'));
const AVG_E = 0.002837200844477648;

const readF32 = (p) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.length / 4);
};

const relErr = (a, b) => {
  let num = 0, den = 0;
  for (let i = 0; i < a.length; i++) { num += (a[i] - b[i]) ** 2; den += b[i] ** 2; }
  return Math.sqrt(num / den);
};
const maxAbs = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
};

for (const c of manifest.cases) {
  test(`resampleSinc matches torchaudio (${c.name})`, () => {
    const input = readF32(path.join(FX, `${c.name}.input48k.f32`));
    const want = readF32(path.join(FX, `${c.name}.resampled.f32`));
    const got = resampleSinc(input, c.input_rate, manifest.sample_rate);
    assert.equal(got.length, want.length, 'output length');
    assert.equal(got.length, c.resampled_samples);
    // torchaudio rounds its sinc kernel to float32 before convolving while we
    // accumulate in float64, so agreement is limited by float32 rounding (~1e-6).
    const rel = relErr(got, want);
    assert.ok(rel < 2e-5, `relative error ${rel}`);
  });

  test(`stftMagnitude matches silentcipher (${c.name})`, () => {
    const res = readF32(path.join(FX, `${c.name}.resampled.f32`));
    const want = readF32(path.join(FX, `${c.name}.mag.f32`));
    const gain = Math.sqrt(AVG_E / meanPower(res));
    const scaled = Float32Array.from(res, (v) => v * gain);
    const { mag, frames } = stftMagnitude(scaled, {
      nFft: manifest.n_fft, hop: manifest.hop_length, bands: manifest.bands,
    });
    assert.equal(frames, c.frames, 'frame count');
    assert.equal(frameCount(res.length, manifest.n_fft, manifest.hop_length), c.frames);
    assert.equal(mag.length, want.length);
    const rel = relErr(mag, want);
    assert.ok(rel < 1e-5, `relative error ${rel} (max abs ${maxAbs(mag, want)})`);
  });
}

test('payloadToSymbols encodes "IRDTS" the way the encoder tiles it', () => {
  const syms = payloadToSymbols([73, 82, 68, 84, 83]);
  assert.equal(syms.length, 21);
  assert.equal(syms[20], 0, 'terminator symbol');
  // 73 = 0b01001001 -> 2-bit groups 01,00,10,01 -> +1 -> 2,1,3,2
  assert.deepEqual(syms.slice(0, 4), [2, 1, 3, 2]);
  assert.ok(syms.slice(0, 20).every((v) => v >= 1 && v <= 4));
});

test('Accumulator round-trips a synthetic perfect watermark', () => {
  const payload = [73, 82, 68, 84, 83];
  const syms = payloadToSymbols(payload);
  const acc = new Accumulator();
  const D = 5, n = 21 * 8;
  const offset = 7; // arbitrary rotation, as if the clip started mid-word
  const logits = new Float32Array(D * n);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < D; d++) logits[d * n + i] = -8;
    logits[syms[(i + offset) % 21] * n + i] = 8;
  }
  acc.push(logits, n, 0);
  const r = acc.result();
  assert.equal(r.verdict, 'detected');
  assert.equal(r.match, 1);
  assert.equal(r.shift, offset);
  assert.deepEqual(r.blind.message, payload);
  assert.equal(r.blind.ascii, 'IRDTS');
  assert.equal(r.blind.confidence, 1);
});

test('Accumulator rejects uniform-random symbol streams', () => {
  let seed = 42;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const acc = new Accumulator();
  const D = 5, n = 21 * 20;
  const logits = new Float32Array(D * n);
  for (let i = 0; i < n; i++) for (let d = 0; d < D; d++) logits[d * n + i] = rnd();
  acc.push(logits, n, 0);
  const r = acc.result();
  assert.equal(r.verdict, 'not_detected');
  assert.ok(r.match < 0.45, `match=${r.match}`);
});

/** Build an Accumulator from an explicit per-frame symbol sequence. */
function accFromSymbols(symbols, confident = 8, cfg = DEFAULTS) {
  const D = cfg.messageDim;
  const n = symbols.length;
  const logits = new Float32Array(D * n);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < D; d++) logits[d * n + i] = -confident;
    logits[symbols[i] * n + i] = confident;
  }
  const acc = new Accumulator(cfg);
  acc.push(logits, n, 0);
  return acc;
}

test('structure metric: watermark-like period-21 pattern scores high', () => {
  const syms = payloadToSymbols([73, 82, 68, 84, 83]);
  const seq = Array.from({ length: 21 * 10 }, (_, i) => syms[i % 21]);
  const s = accFromSymbols(seq).structure();
  assert.ok(s > 0.9, `structure=${s}`);
});

test('structure metric is not fooled by a constant prediction', () => {
  // A pure tone makes the decoder emit the same symbol everywhere; that folds
  // perfectly at *every* period and must therefore score ~0, not ~1.
  const s = accFromSymbols(new Array(21 * 10).fill(3)).structure();
  assert.equal(s, 0, `structure=${s}`);
});

test('structure metric is not fooled by a near-constant prediction', () => {
  // What a 440 Hz sine actually produces: one symbol everywhere except a handful of
  // frames at the file edges (verbatim from the decoder, 12 s at 44.1 kHz).  Those few
  // frames must not become "structure" -- period 21 and the null periods have to be
  // measured over the same frames, and with only ~7 frames left unexplained there is
  // nothing to normalise against anyway.  Folding whole rows only scored this 0.38.
  const seq = new Array(261).fill(3);
  for (const [i, v] of [[0, 1], [2, 4], [254, 4], [256, 1], [257, 1], [258, 1], [259, 2]]) seq[i] = v;
  const s = accFromSymbols(seq).structure();
  assert.ok(s < THRESHOLDS.structurePresent, `structure=${s}`);
});

test('structure metric is low for unstructured predictions', () => {
  let seed = 11;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) % 5;
  const seq = Array.from({ length: 21 * 20 }, rnd);
  const s = accFromSymbols(seq).structure();
  assert.ok(s < 0.25, `structure=${s}`);
});

test('structure metric detects a wrong-payload watermark (payload independent)', () => {
  const other = payloadToSymbols([1, 2, 3, 4, 5]);
  const seq = Array.from({ length: 21 * 10 }, (_, i) => other[i % 21]);
  const acc = accFromSymbols(seq);
  const r = acc.result();
  assert.ok(r.structure > 0.9, `structure=${r.structure}`);
  assert.notEqual(r.verdict, 'detected');
  assert.ok(r.structurePresent);
  assert.ok(notes(r).some((n) => /different payload/.test(n)), notes(r).join(' | '));
});

test('a resolved payload mismatch is reported as such', () => {
  // "IRDTX" differs from "IRDTS" in one byte: strongly correlated, but not Irodori's.
  const near = payloadToSymbols([73, 82, 68, 84, 88]);
  const seq = Array.from({ length: 21 * 12 }, (_, i) => near[i % 21]);
  const r = accFromSymbols(seq).result();
  assert.equal(r.blind.ascii, 'IRDTX');
  assert.equal(r.payloadExact, false);
  assert.ok(r.blind.confidence >= THRESHOLDS.payloadResolved);
  assert.ok(notes(r).some((n) => n.includes('"IRDTX"')), notes(r).join(' | '));
});

/* ------------------------------------------------------- 16 kHz variant ----- */

test('null periods are coprime to the message length', () => {
  // gcd(p, L) > 1 would leak real period-L structure into the null estimate.
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  for (const L of [16, 21]) {
    const nulls = nullPeriods(L);
    assert.equal(nulls.length, 4);
    for (const p of nulls) assert.equal(gcd(p, L), 1, `gcd(${p},${L})`);
  }
  // The 44.1k set must stay exactly as calibrated.
  assert.deepEqual(nullPeriods(21), [19, 20, 22, 23]);
  // 14 and 18 share a factor of 2 with 16 and must be skipped.
  assert.deepEqual(nullPeriods(16), [13, 15, 17, 19]);
});

test('16k variant exposes no byte payload', () => {
  const v = VARIANTS['16k'];
  assert.equal(v.bytePayload, false);
  assert.equal(v.payload, null);
  // 15 symbol slots cannot hold a whole number of bytes at 4 symbols per byte.
  assert.equal((v.messageLen - 1) % 4 !== 0, true);
  const acc = new Accumulator(v);
  assert.equal(acc.expected, null, 'no expected word to correlate against');
});

test('16k: a period-16 watermark is detected payload-independently', () => {
  const v = VARIANTS['16k'];
  // 15 slots of values 0..2 (one-hot index value+1 must stay below message_dim=4),
  // plus the terminator; this is the only message layout the 16k model can carry.
  const msg = [0, 1, 2, 2, 1, 0, 1, 2, 0, 0, 2, 1, 1, 2, 0];
  const word = [...msg.map((s) => s + 1), 0];
  assert.equal(word.length, v.messageLen);
  const seq = Array.from({ length: v.messageLen * 10 }, (_, i) => word[i % v.messageLen]);
  const r = accFromSymbols(seq, 8, v).result();
  assert.equal(r.variant, '16k');
  assert.equal(r.bytePayload, false);
  assert.equal(r.verdict, 'detected');
  assert.ok(r.structure > 0.9, `structure=${r.structure}`);
  assert.deepEqual(r.blind.symbols, msg, 'raw symbol word recovered');
  assert.equal(r.blind.message, null, 'no bytes are derived for this variant');
  assert.equal(r.match, null, 'no payload-matched statistic exists');
  assert.ok(verdictLabel(r).includes('SilentCipher'));
  assert.ok(!verdictLabel(r).includes('Irodori'), verdictLabel(r));
  assert.ok(notes(r, v).some((n) => /payload-independent/.test(n)), notes(r, v).join(' | '));
});

test('16k: unstructured predictions are rejected', () => {
  const v = VARIANTS['16k'];
  let seed = 3;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) % v.messageDim;
  const seq = Array.from({ length: v.messageLen * 20 }, rnd);
  const r = accFromSymbols(seq, 8, v).result();
  assert.equal(r.verdict, 'not_detected');
  assert.ok(r.structure < v.thresholds.rejectStructure, `structure=${r.structure}`);
});

test('16k: a constant prediction is rejected', () => {
  const v = VARIANTS['16k'];
  const r = accFromSymbols(new Array(v.messageLen * 20).fill(2), 8, v).result();
  assert.equal(r.verdict, 'not_detected');
  assert.equal(r.structure, 0);
});

test('16k: too little audio yields no verdict', () => {
  const v = VARIANTS['16k'];
  const word = [1, 2, 3, 3, 2, 1, 2, 3, 1, 1, 3, 2, 2, 3, 1, 0];
  const seq = Array.from({ length: 32 }, (_, i) => word[i % v.messageLen]);
  const r = accFromSymbols(seq, 8, v).result();
  assert.equal(r.verdict, 'insufficient');
  assert.ok(notes(r, v).some((n) => /at least/i.test(n)), notes(r, v).join(' | '));
});

/* ------------------------------------------- window / minimum-length logic --- */

test('windowFrames never returns less than the verdict minimum', () => {
  for (const v of Object.values(VARIANTS)) {
    const min = v.thresholds.minFrames;
    assert.equal(windowFrames('min', v), min);
    assert.equal(windowFrames(undefined, v), min);
    assert.equal(windowFrames('all', v), Infinity);
    // A window shorter than the minimum must be clamped up, not honoured, otherwise
    // the analysis could never reach a verdict.
    assert.equal(windowFrames('1', v), min, `${v.id} with a 1 s window`);
    assert.equal(windowFrames('0', v), min);
    assert.equal(windowFrames('nonsense', v), min);
    // A longer window is honoured.
    const long = windowFrames('60', v);
    assert.ok(long > min, `${v.id}: ${long} vs ${min}`);
    assert.equal(long, Math.round((60 * v.sampleRate) / v.hopLength));
  }
});

test('canReachVerdict matches the frame arithmetic', () => {
  for (const v of Object.values(VARIANTS)) {
    const need = minSeconds(v);
    assert.equal(canReachVerdict(v, need + 0.5), true);
    assert.equal(canReachVerdict(v, need - 0.5), false);
    // The boundary must agree with the STFT frame count actually produced.
    const frames = frameCount(Math.round(need * v.sampleRate), v.nFft, v.hopLength);
    assert.ok(frames >= v.thresholds.minFrames,
      `${v.id}: ${need}s yields ${frames} frames, needs ${v.thresholds.minFrames}`);
  }
});

test('a file shorter than the minimum reports why, not a bare "too short"', () => {
  // Reproduces ./synthesize.wav: 7.79 s is enough for 44.1 kHz (6 s) but two frames
  // short of the 16 kHz minimum (8 s), so no window can rescue it.
  const v16 = VARIANTS['16k'];
  const durationSec = 7.79;
  assert.equal(canReachVerdict(v16, durationSec), false);
  assert.equal(canReachVerdict(VARIANTS['44.1k'], durationSec), true);

  const frames = frameCount(Math.round(durationSec * v16.sampleRate), v16.nFft, v16.hopLength);
  assert.ok(frames < v16.thresholds.minFrames, `${frames} frames`);

  const r = new Accumulator(v16).result();
  assert.equal(r.verdict, 'insufficient');
  r.availableFrames = frames;
  r.requiredFrames = v16.thresholds.minFrames;
  r.availableSeconds = durationSec;
  const text = notes(r, v16).join(' ');
  assert.match(text, /7\.79 s/, 'states the actual duration');
  assert.match(text, /8\.0 s/, 'states the requirement');
  assert.match(text, new RegExp(`${frames} of the ${v16.thresholds.minFrames}`), 'states frames');
  assert.match(text, /No window setting can change that/, 'explains the window is irrelevant');
  assert.match(text, /44\.1 kHz variant/, 'points at the variant that matters for Irodori');
});

test('binomSurvival behaves like a binomial tail', () => {
  assert.equal(binomSurvival(0, 10, 0.3), 1);
  assert.equal(binomSurvival(11, 10, 0.3), 0);
  // P(X>=1) for n=1,p=0.25
  assert.ok(Math.abs(binomSurvival(1, 1, 0.25) - 0.25) < 1e-9);
  // P(X>=5) for n=10,p=0.5 = 0.623046875
  assert.ok(Math.abs(binomSurvival(5, 10, 0.5) - 0.623046875) < 1e-9);
});
