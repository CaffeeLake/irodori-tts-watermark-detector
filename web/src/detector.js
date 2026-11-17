/**
 * Irodori-TTS / SilentCipher watermark detector.
 *
 * Irodori-TTS watermarks its output with SilentCipher (sony/silentcipher, 44.1 kHz)
 * carrying the fixed 5-byte payload "IRDTS" (see irodori_tts/watermark.py).
 *
 * Detection pipeline (entirely client-side; inference runs in WebAssembly):
 *
 *   PCM -> mono -> resample to the variant's rate -> scale to VCTK reference power
 *       -> |STFT| -> dec_m_0 (ONNX/WASM) -> message_dim logits per frame
 *       -> period/rotation analysis -> verdict
 *
 * The encoder tiles a fixed `messageLen`-symbol word across time, repeating it once
 * every `messageLen` STFT frames.  Two independent channels of evidence come out of
 * that, and which ones are available depends on the variant (see variants.js):
 *
 *   1. `margin` / `pValue`  - payload-MATCHED evidence.  We score the known payload
 *                             word at all `messageLen` rotations; the wrong rotations
 *                             form an empirical null, so the test self-calibrates.
 *                             Only available when the variant can carry a byte payload
 *                             (44.1 kHz).  Drives the verdict there.
 *   2. `structure`          - payload-INDEPENDENT evidence that *some* SilentCipher
 *                             watermark is present: how strongly the per-frame
 *                             predictions repeat with period `messageLen`, corrected
 *                             for chance using coprime neighbouring periods.
 *                             Drives the verdict for the 16 kHz variant.
 *
 * `blind` always reports the symbol word actually recovered (and its byte/ASCII form
 * where that is meaningful), so a watermark carrying a *different* payload is never
 * silently reported as Irodori's.
 */

import {
  VARIANTS, DEFAULT_VARIANT, getVariant, nullPeriods, minSeconds, IRODORI_PAYLOAD,
  windowFrames, canReachVerdict,
} from './variants.js';

export {
  VARIANTS, DEFAULT_VARIANT, getVariant, nullPeriods, minSeconds, IRODORI_PAYLOAD,
  windowFrames, canReachVerdict,
};

/** The variant Irodori-TTS actually uses; kept as the default config. */
export const DEFAULTS = VARIANTS[DEFAULT_VARIANT];
export const THRESHOLDS = DEFAULTS.thresholds;

/** payload bytes -> the symbol word the encoder tiles across time. */
export function payloadToSymbols(payload, messageLen = DEFAULTS.messageLen) {
  let bits = '';
  for (const b of payload) bits += b.toString(2).padStart(8, '0');
  const syms = [];
  for (let i = 0; i < bits.length / 2; i++) syms.push(parseInt(bits.slice(i * 2, i * 2 + 2), 2) + 1);
  syms.push(0); // terminator
  if (syms.length !== messageLen) {
    throw new Error(`payload of ${payload.length} bytes yields ${syms.length} symbols, `
      + `expected ${messageLen}`);
  }
  return syms;
}

/** Inverse of payloadToSymbols: pairs of bits -> bytes. */
function symbolsToBytes(vals) {
  let bits = '';
  for (const v of vals) bits += v.toString(2).padStart(2, '0');
  const out = [];
  for (let i = 0; i < Math.floor(bits.length / 8); i++) out.push(parseInt(bits.slice(i * 8, i * 8 + 8), 2));
  return out;
}

export function bytesToAscii(bytes) {
  return bytes.map((c) => (c >= 32 && c < 127 ? String.fromCharCode(c) : '.')).join('');
}

/** What a variant's message word can hold. */
export function payloadCapacity(cfg) {
  const slots = cfg.messageLen - 1;      // one slot is the terminator
  const bits = slots * 2;                // the encoder packs two bits per slot
  return {
    slots,
    bits,
    bytes: bits % 8 === 0 ? bits / 8 : null,   // null => no whole-byte payload fits
    /** One-hot index is symbol+1 and must stay below messageDim. */
    maxSymbol: cfg.messageDim - 2,
  };
}

/** Raw symbol word (0-based values) -> the tiled word including the terminator. */
export function symbolsToWord(symbols, cfg) {
  return [...symbols.map((v) => v + 1), 0];
}

/**
 * Parse a user-supplied payload for `cfg`.
 *
 * Accepts, for variants that can carry bytes:
 *   "IRDTS"                  5 ASCII characters
 *   "73,82,68,84,83"         decimal bytes (comma or whitespace separated)
 *   "0x49 0x52 0x44 0x54 0x53"  hex bytes
 * and for variants that cannot (16 kHz), the raw symbol word instead:
 *   "0,1,2,2,1,0,1,2,0,0,2,1,1,2,0"
 *
 * @returns {{ok:boolean, bytes?:number[], symbols?:number[], word?:number[], error?:string}}
 */
export function parsePayload(text, cfg) {
  const cap = payloadCapacity(cfg);
  const raw = String(text ?? '').trim();
  if (!raw) return { ok: false, error: 'Enter a payload.' };

  const looksNumeric = /[,\s]/.test(raw) || /^0x/i.test(raw);

  if (!cfg.bytePayload) {
    if (!looksNumeric) {
      return {
        ok: false,
        error: `The ${cfg.id} variant cannot carry text: ${cap.slots} symbol slots hold `
          + `${cap.bits} bits, which is not a whole number of bytes. Enter ${cap.slots} `
          + `symbol values 0-${cap.maxSymbol} instead.`,
      };
    }
    const vals = raw.split(/[,\s]+/).filter(Boolean).map((t) => Number(t));
    if (vals.length !== cap.slots || vals.some((v) => !Number.isInteger(v))) {
      return { ok: false, error: `Expected ${cap.slots} whole numbers, got ${vals.length}.` };
    }
    const bad = vals.find((v) => v < 0 || v > cap.maxSymbol);
    if (bad !== undefined) {
      return {
        ok: false,
        error: `Symbol ${bad} is out of range: the ${cfg.id} variant allows 0-${cap.maxSymbol} `
          + `(the one-hot index is symbol+1 and must stay below messageDim ${cfg.messageDim}).`,
      };
    }
    return { ok: true, symbols: vals, word: symbolsToWord(vals, cfg) };
  }

  let bytes;
  if (looksNumeric) {
    const parts = raw.split(/[,\s]+/).filter(Boolean);
    bytes = parts.map((t) => (/^0x/i.test(t) ? parseInt(t, 16) : Number(t)));
    if (bytes.some((b) => !Number.isInteger(b))) {
      return { ok: false, error: 'Could not read that as a list of byte values.' };
    }
  } else {
    bytes = [...raw].map((ch) => ch.codePointAt(0));
    if (bytes.some((b) => b > 255)) {
      return { ok: false, error: 'Only single-byte (Latin-1) characters can be encoded.' };
    }
  }

  if (bytes.length !== cap.bytes) {
    return {
      ok: false,
      error: `The ${cfg.id} payload is exactly ${cap.bytes} bytes `
        + `(${cap.slots} slots x 2 bits = ${cap.bits} bits); got ${bytes.length}.`,
    };
  }
  const bad = bytes.find((b) => b < 0 || b > 255);
  if (bad !== undefined) return { ok: false, error: `Byte ${bad} is outside 0-255.` };

  return { ok: true, bytes, symbols: null, word: payloadToSymbols(bytes, cfg.messageLen) };
}

/**
 * A variant config with a different expected word.  The detection statistic compares
 * one rotation against the others, so its calibration depends on messageLen and
 * messageDim, not on which word is being looked for -- the shipped thresholds apply
 * to any payload (verified in test/payload.test.mjs).
 */
export function withPayload(cfg, payload) {
  if (payload == null) return cfg;
  const parsed = Array.isArray(payload)
    ? (cfg.bytePayload
      ? { ok: true, bytes: payload, word: payloadToSymbols(payload, cfg.messageLen) }
      : { ok: true, symbols: payload, word: symbolsToWord(payload, cfg) })
    : parsePayload(payload, cfg);
  if (!parsed.ok) throw new Error(parsed.error);
  return {
    ...cfg,
    payload: parsed.bytes ?? null,
    payloadSymbols: parsed.symbols ?? null,
    expectedWord: parsed.word,
  };
}

/**
 * Streaming accumulator over per-frame decoder logits.
 * Fed chunk by chunk so the UI can update and stop early.
 */
export class Accumulator {
  constructor(cfg = DEFAULTS) {
    this.cfg = cfg;
    this.L = cfg.messageLen;
    this.D = cfg.messageDim;
    this.nulls = nullPeriods(this.L);
    // An explicit word (from a user-supplied payload) wins; otherwise derive it from
    // the variant's default payload, if that variant can carry one at all.
    this.expected = cfg.expectedWord
      ?? (cfg.bytePayload && cfg.payload ? payloadToSymbols(cfg.payload, this.L) : null);
    this.frames = 0;
    this.firstFrame = null;
    this.hardHits = new Float64Array(this.L);   // per-rotation argmax matches
    this.softLL = new Float64Array(this.L);     // per-rotation summed log-prob
    this.argmax = new Uint8Array(1024);         // per-frame argmax (grows on demand)
  }

  #pushArgmax(v) {
    if (this.frames >= this.argmax.length) {
      const bigger = new Uint8Array(this.argmax.length * 2);
      bigger.set(this.argmax);
      this.argmax = bigger;
    }
    this.argmax[this.frames] = v;
  }

  /**
   * @param {Float32Array} logits  (D, n) row-major: logits[d * n + i]
   * @param {number} n             frames in this chunk
   * @param {number} frameOffset   absolute index of the first frame
   */
  push(logits, n, frameOffset) {
    const { L, D, expected } = this;
    if (this.firstFrame === null) this.firstFrame = frameOffset;
    for (let i = 0; i < n; i++) {
      let max = -Infinity, arg = 0;
      for (let d = 0; d < D; d++) {
        const v = logits[d * n + i];
        if (v > max) { max = v; arg = d; }
      }
      if (expected) {
        let sum = 0;
        for (let d = 0; d < D; d++) sum += Math.exp(logits[d * n + i] - max);
        const logSum = max + Math.log(sum);
        const t = frameOffset + i;
        for (let r = 0; r < L; r++) {
          const e = expected[(t + r) % L];
          if (arg === e) this.hardHits[r] += 1;
          this.softLL[r] += logits[e * n + i] - logSum;
        }
      }
      this.#pushArgmax(arg);
      this.frames++;
    }
  }

  /**
   * Vote concentration when the frame predictions are folded at `period`:
   * the fraction of frames agreeing with their position's modal symbol.
   *
   * Every frame is folded, including those in the final partial row.  That matters
   * because `structure()` compares periods against one another and so they must all
   * see the *same* audio: folding whole rows only gives each period a different slice
   * (at 261 frames period 21 covers 252 of them, period 20 covers 260, period 22 only
   * 242), and a transient that lands inside one slice but outside another then shows
   * up as "structure" that is really just a difference of windows.
   */
  #concentration(period) {
    const { D } = this;
    const T = this.frames;
    if (T < period) return 0;
    const base = this.firstFrame ?? 0;
    const counts = new Int32Array(period * D);
    for (let i = 0; i < T; i++) counts[((base + i) % period) * D + this.argmax[i]] += 1;
    let agree = 0;
    for (let p = 0; p < period; p++) {
      let max = 0;
      for (let d = 0; d < D; d++) if (counts[p * D + d] > max) max = counts[p * D + d];
      agree += max;
    }
    return agree / T;
  }

  /**
   * Payload-independent evidence that *some* SilentCipher watermark is present.
   *
   * Raw period-L concentration is not usable on its own: a signal whose decoder output
   * is nearly constant (a pure tone, say) folds perfectly at *every* period and would
   * score ~1.0.  We therefore compare period L against coprime neighbouring periods,
   * measured over the same frames, which share the marginal symbol distribution but
   * carry no watermark structure.  Constant or unstructured predictions score 0.
   */
  structure() {
    const T = this.frames;
    if (!T) return 0;
    const c = this.#concentration(this.L);
    const nulls = this.nulls.map((p) => this.#concentration(p)).filter((v) => v > 0);
    if (!nulls.length) return 0;
    const base = nulls.reduce((a, b) => a + b, 0) / nulls.length;

    // `1 - base` is the share of frames the null folds already fail to explain, and it
    // is what the lift gets normalised by -- so in frames, k = (1 - base) * T is also
    // the resolution of the statistic: one frame changing its fold moves the result by
    // 1/k.  A near-constant decoder output (a pure tone) leaves only k ~ 2..7 frames
    // unexplained, where the ratio is pure quantisation noise and readily reaches 0.4;
    // a genuine watermark leaves k >= 70 even at the minimum verdict window.  Below one
    // message word of unexplained frames there is nothing left to measure, so report no
    // structure rather than noise.
    if ((1 - base) * T < this.L) return 0;
    return Math.min(1, Math.max(0, (c - base) / (1 - base)));
  }

  /**
   * Recover the symbol word, reproducing silentcipher's mode-over-repetitions logic
   * (including `scipy.stats.mode`'s smallest-value tie-break).  Bytes are only derived
   * for variants whose symbol layout can represent them.
   */
  blindDecode() {
    const { L, D, cfg } = this;
    const empty = { symbols: null, message: null, ascii: null, confidence: 0, modal: null };
    const rows = Math.floor(this.frames / L);
    if (!rows) return empty;

    const base = this.firstFrame ?? 0;
    const counts = Array.from({ length: L }, () => new Int32Array(D));
    for (let i = 0; i < rows * L; i++) counts[(base + i) % L][this.argmax[i]] += 1;

    const modal = new Int32Array(L);   // indexed by (absolute frame) % L
    let agree = 0;
    for (let p = 0; p < L; p++) {
      let best = 0, bestC = -1;
      for (let d = 0; d < D; d++) if (counts[p][d] > bestC) { bestC = counts[p][d]; best = d; }
      modal[p] = best;
      agree += bestC;
    }
    const confidence = agree / (rows * L);

    // The word is read starting at the analysis origin, then rotated so the
    // terminator (symbol 0) sits at the end.
    const word = Array.from({ length: L }, (_, i) => modal[(base + i) % L]);
    const end = word.indexOf(0);
    if (end < 0) return { ...empty, confidence, modal };
    const rotated = [...word.slice(end + 1), ...word.slice(0, end)];
    if (rotated.length !== L - 1) return { ...empty, confidence, modal };

    const symbols = rotated.map((v) => v - 1);   // back to 0-based symbol values
    const out = { symbols, message: null, ascii: null, confidence, modal };
    if (cfg.bytePayload && symbols.every((v) => v >= 0 && v <= 3)) {
      out.message = symbolsToBytes(symbols);
      out.ascii = bytesToAscii(out.message);
    }
    return out;
  }

  /** Current statistics + verdict. */
  result(th = this.cfg.thresholds) {
    const { L, cfg } = this;
    const T = this.frames;
    const blank = {
      variant: cfg.id,
      bytePayload: cfg.bytePayload,
      frames: 0,
      seconds: 0,
      verdict: 'insufficient',
      shift: 0,
      structure: 0,
      structurePresent: false,
      match: null, nullMatch: null, margin: null, z: null, pValue: null,
      logLik: null, logLikRatio: null, symbolAgreement: null, payloadExact: null,
      perRotation: null, perRotationLogLik: null,
      // What this run was actually looking for, so the UI never claims to have
      // searched for the default payload when the caller asked for another one.
      expectedPayload: cfg.payload ?? null,
      expectedAscii: cfg.payload ? bytesToAscii(cfg.payload) : null,
      expectedSymbols: cfg.payloadSymbols ?? null,
      customPayload: cfg.expectedWord != null,
      blind: { symbols: null, message: null, ascii: null, confidence: 0 },
    };
    if (!T) return blank;

    const blind = this.blindDecode();
    const structure = this.structure();
    const seconds = (T * cfg.hopLength) / cfg.sampleRate;
    const common = {
      ...blank,
      frames: T,
      seconds,
      structure,
      structurePresent: structure >= th.structurePresent,
      blind: {
        symbols: blind.symbols, message: blind.message,
        ascii: blind.ascii, confidence: blind.confidence,
      },
    };

    /** Verdict from period structure alone; the only calibrated path for 16 kHz. */
    const structureVerdict = () => {
      if (T < th.minFrames) return 'insufficient';
      if (structure >= th.detectStructure) return 'detected';
      if (structure < th.rejectStructure) return 'not_detected';
      return 'inconclusive';
    };

    if (!this.expected) return { ...common, verdict: structureVerdict() };

    const match = Array.from(this.hardHits, (h) => h / T);
    const ll = Array.from(this.softLL, (s) => s / T);

    // Pick the rotation by log-likelihood: continuous, so no arbitrary tie-breaks.
    let best = 0;
    for (let r = 1; r < L; r++) if (ll[r] > ll[best]) best = r;

    const omit = (arr) => arr.filter((_, r) => r !== best);
    const oMatch = omit(match), oLl = omit(ll);
    const nullMatch = oMatch.reduce((a, b) => a + b, 0) / oMatch.length;
    const varr = oMatch.reduce((a, b) => a + (b - nullMatch) ** 2, 0) / (oMatch.length - 1);
    const z = (match[best] - nullMatch) / (Math.sqrt(varr) + 1e-12);
    const pValue = binomSurvival(Math.round(match[best] * T), T, Math.max(nullMatch, 1e-6));
    const logLikRatio = ll[best] - oLl.reduce((a, b) => a + b, 0) / oLl.length;
    const margin = match[best] - nullMatch;

    let symbolAgreement = 0;
    if (blind.modal) {
      for (let p = 0; p < L; p++) {
        if (blind.modal[p] === this.expected[(p + best) % L]) symbolAgreement++;
      }
    }

    // The payload-matched verdict is only used where its thresholds were calibrated.
    // A 16 kHz variant given an expected word still reports match/margin/agreement, but
    // its verdict stays on the structure test rather than borrowing 44.1 kHz cut-offs
    // that were never measured for a 16-rotation, 4-symbol null distribution.
    const payloadCalibrated = th.detectMargin != null && th.detectP != null;
    let verdict;
    if (!payloadCalibrated) verdict = structureVerdict();
    else if (T < th.minFrames) verdict = 'insufficient';
    else if (margin >= th.detectMargin && pValue <= th.detectP) verdict = 'detected';
    else if (margin < th.rejectMargin || pValue > th.rejectP) verdict = 'not_detected';
    else verdict = 'inconclusive';

    return {
      ...common,
      verdict,
      shift: best,
      match: match[best],
      nullMatch,
      margin,
      z,
      pValue,
      logLik: ll[best],
      logLikRatio,
      symbolAgreement,
      // "Did we read back exactly the word we were looking for?"  Compared as bytes
      // where the variant has them, as raw symbols where it does not (16 kHz).
      payloadExact: cfg.bytePayload
        ? (cfg.payload != null && blind.ascii === bytesToAscii(cfg.payload))
        : (blind.symbols != null && cfg.payloadSymbols != null
          && blind.symbols.length === cfg.payloadSymbols.length
          && blind.symbols.every((v, i) => v === cfg.payloadSymbols[i])),
      perRotation: match,
      perRotationLogLik: ll,
    };
  }
}

/** P(X >= k) for X ~ Binomial(n, p), computed with log-space terms. */
export function binomSurvival(k, n, p) {
  if (k <= 0) return 1;
  if (k > n) return 0;
  let acc = 0;
  for (let i = k; i <= n; i++) {
    acc += Math.exp(logChoose(n, i) + i * Math.log(p) + (n - i) * Math.log1p(-p));
  }
  return Math.min(1, Math.max(0, acc));
}

function logChoose(n, k) {
  return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1);
}

/** Lanczos approximation of log|Gamma(x)| for x > 0. */
function lgamma(x) {
  const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313,
    -176.61502916214059, 12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = 0.99999999999980993;
  const t = x + 7.5;
  for (let i = 0; i < 8; i++) a += g[i] / (x + i + 1);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

export function verdictLabel(r) {
  const v = typeof r === 'string' ? r : r.verdict;
  const byte = typeof r === 'string' ? true : r.bytePayload;
  if (v === 'detected') {
    return byte ? 'Irodori-TTS watermark DETECTED' : 'SilentCipher watermark DETECTED';
  }
  return {
    not_detected: byte ? 'No Irodori-TTS watermark found' : 'No SilentCipher watermark found',
    inconclusive: 'Inconclusive',
    insufficient: 'Not enough audio analysed',
  }[v] || v;
}

/** Human-readable caveats worth showing next to the verdict. */
export function notes(r, cfg = DEFAULTS, th = cfg.thresholds) {
  const out = [];
  if (r.payloadError) return [r.payloadError];
  /** The word this run searched for, which may not be the variant's default. */
  const want = r.expectedAscii
    ?? (r.expectedSymbols ? `[${r.expectedSymbols.join(', ')}]` : null)
    ?? (cfg.payload ? bytesToAscii(cfg.payload) : null);

  if (r.verdict === 'insufficient') {
    const need = minSeconds(cfg);
    const have = r.availableSeconds ?? r.seconds;
    const frames = r.availableFrames != null
      ? ` (${r.availableFrames} of the ${r.requiredFrames} STFT frames it needs)` : '';
    out.push(`The ${cfg.id} variant needs at least ${need.toFixed(1)} s of audio, but this `
      + `file provides ${have.toFixed(2)} s${frames}. No window setting can change that — `
      + 'the audio itself is too short.');
    if (!cfg.bytePayload) {
      out.push('Note that Irodori-TTS only ever uses the 44.1 kHz variant, which needs '
        + `${minSeconds(VARIANTS[DEFAULT_VARIANT]).toFixed(1)} s — so for the question `
        + '"is this Irodori-TTS output?" that variant is the one that matters.');
    }
  }

  if (!cfg.bytePayload) {
    if (r.verdict === 'detected') {
      out.push(`This is a payload-independent detection: the ${cfg.id} SilentCipher model `
        + 'cannot carry a byte payload at all (15 symbol slots vs. 4 symbols per byte), so '
        + 'there is no Irodori payload to match. It confirms a SilentCipher watermark, not '
        + 'that Irodori-TTS produced the audio.');
      if (r.blind.symbols) {
        out.push(`Recovered symbol word: [${r.blind.symbols.join(', ')}] `
          + `(confidence ${(r.blind.confidence * 100).toFixed(1)}%).`);
      }
    }
    return out;
  }

  const resolved = r.blind.confidence >= th.payloadResolved;
  if (r.verdict === 'detected' && !r.payloadExact) {
    if (resolved) {
      out.push(`The payload read out of this file is "${r.blind.ascii}", not "${want}" `
        + `(${r.symbolAgreement}/${cfg.messageLen} symbol positions agree). This looks like a `
        + 'SilentCipher watermark carrying a different payload.');
    } else {
      out.push(`The correlation against "${want}" is significant, but there are not yet `
        + 'enough clean repetitions to read the payload back exactly. Analyse more audio '
        + 'for an exact read-out.');
    }
  }
  if (r.verdict !== 'detected' && r.structurePresent) {
    out.push(`A period-${cfg.messageLen} SilentCipher-like pattern is present, but it does not `
      + `match "${want}" — most likely a SilentCipher watermark carrying a different payload.`);
  }
  return out;
}
