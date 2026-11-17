/**
 * Glue between the DSP front-end, the ONNX/WASM message decoder and the detector
 * statistics.  Runtime-agnostic: callers pass in an already-created
 * onnxruntime-web InferenceSession plus the `Tensor` constructor, so the exact
 * same code path is exercised by the browser UI and by the Node test harness.
 */

import { resampleSinc, stftMagnitude, meanPower, toMono, frameCount } from './dsp.js';
import { Accumulator, DEFAULTS } from './detector.js';

/**
 * Mono-downmix, resample to the variant's rate, normalise to SilentCipher's reference
 * power and compute the magnitude spectrogram.
 *
 * `opts.maxFrames` caps how much of the file is actually transformed.  This matters a
 * great deal: a verdict needs ~129 frames (6 s), so on a 5-minute file the naive
 * version resampled and transformed 7345 frames to use 129.  Measured on 341 s of
 * audio, the front end cost 7.2 s (5.2 s of it resampling) versus 150 ms for just the
 * frames needed -- a 48x saving, and the resampler is pure JS so it hurts most on the
 * slow machines that are already struggling with inference.
 *
 * Note on the power normalisation: the reference scales by the mean power of the
 * *resampled* signal, which would force a full-length resample purely to obtain one
 * scalar.  We measure the whole-signal power at the input rate instead.  The two differ
 * only by the energy above the new Nyquist frequency -- 0.077 % on real audio, i.e. a
 * 0.04 % difference in gain, which leaves the decoder's argmax decisions unchanged
 * (asserted by test/parity.test.mjs) and is four orders of magnitude below the
 * calibration margins.
 *
 * @param {Float32Array[]|Float32Array} channels
 * @param {number} sampleRate
 * @param {object} cfg                      variant configuration
 * @param {{maxFrames?:number}} opts        frame budget; omit to transform everything
 */
export function prepare(channels, sampleRate, cfg = DEFAULTS, opts = {}) {
  const mono = Array.isArray(channels) ? toMono(channels) : channels;
  const { maxFrames = Infinity } = opts;
  const durationSec = mono.length / sampleRate;

  // Whole-signal power, measured at the input rate (O(n), ~25 ms for 16 M samples).
  const power = meanPower(mono);
  if (!(power > 0)) {
    return {
      mag: new Float32Array(0), frames: 0, bands: cfg.messageBandSize,
      silent: true, durationSec, variant: cfg.id, fileFrames: 0, truncated: false,
    };
  }
  const gain = Math.sqrt(cfg.averageEnergyVctk / power);

  // Frame f of the STFT reads target-rate samples [f*hop - nFft/2, f*hop + nFft/2).
  // Keep the halo plus a couple of frames of slack so every frame we will actually use
  // is identical to what a full-length transform would have produced.
  let src = mono;
  let truncated = false;
  if (Number.isFinite(maxFrames)) {
    const framesNeeded = maxFrames + cfg.halo + 2;
    const needTarget = framesNeeded * cfg.hopLength + cfg.nFft;
    const needInput = Math.ceil((needTarget * sampleRate) / cfg.sampleRate) + 1024;
    if (needInput < mono.length) {
      src = mono.subarray(0, needInput);
      truncated = true;
    }
  }

  const y = resampleSinc(src, sampleRate, cfg.sampleRate);
  const scaled = new Float32Array(y.length);
  for (let i = 0; i < y.length; i++) scaled[i] = y[i] * gain;

  const { mag, frames, bands } = stftMagnitude(scaled, {
    nFft: cfg.nFft, hop: cfg.hopLength, bands: cfg.messageBandSize,
  });
  return {
    mag,
    frames,
    bands,
    silent: false,
    durationSec,
    variant: cfg.id,
    truncated,
    /** Frames the whole file would yield, for display when `truncated`. */
    fileFrames: frameCount(
      Math.round((mono.length * cfg.sampleRate) / sampleRate), cfg.nFft, cfg.hopLength,
    ),
  };
}

/**
 * Plan the chunk boundaries for `[start, end)`.
 *
 * Every chunk carries `halo` frames of discarded context on each side, so small chunks
 * compute more frames than they use: 21-frame chunks cost +101%, 129-frame chunks +8%.
 *
 * Measured in Firefox, however, total time is almost flat across chunk sizes (8%
 * spread over 21..129 for the same 129-frame budget): WASM inference here is bound by
 * memory bandwidth rather than FLOPs, so the per-frame cost rises from ~104 ms to
 * ~181 ms as tensors get bigger and almost exactly cancels the saved frames.
 *
 * Because total time is flat, chunks are kept small by default: one 129-frame pass is
 * only ~1.6% faster than three 43-frame passes, but it shows the user nothing for 25 s,
 * whereas small chunks stream partial statistics out as they are computed.  Pass
 * `firstChunk` explicitly if you would rather pay nothing for the halo.
 */
export function chunkPlan(start, end, firstChunk, chunkFrames) {
  const plan = [];
  let s = start;
  if (s < end) {
    const e = Math.min(end, s + Math.max(1, firstChunk));
    plan.push([s, e]);
    s = e;
  }
  while (s < end) {
    const e = Math.min(end, s + Math.max(1, chunkFrames));
    plan.push([s, e]);
    s = e;
  }
  return plan;
}

/**
 * Run the decoder over `[startFrame, startFrame + maxFrames)` in chunks.
 *
 * Each chunk is padded with `cfg.halo` frames of real context on both sides and
 * those context frames are discarded, which makes the chunked result bit-identical
 * to a single full-length forward pass regardless of how it is chunked.
 *
 * @param {{mag:Float32Array, frames:number, bands:number}} spec
 * @param {{run:Function}} session           onnxruntime InferenceSession
 * @param {Function} TensorCtor              onnxruntime Tensor constructor
 * @param {object} opts
 * @returns {Promise<{accumulator:Accumulator, result:object, stopped:boolean}>}
 */
export async function runDetector(spec, session, TensorCtor, opts = {}) {
  const cfg = opts.cfg || DEFAULTS;
  const {
    startFrame = 0,
    maxFrames = spec.frames,
    chunkFrames = 43,
    firstChunkFrames = chunkFrames,
    inputName = 'magnitude',
    earlyStop = true,
    onProgress = null,
    shouldStop = null,
  } = opts;

  const acc = new Accumulator(cfg);
  const total = spec.frames;
  const end = Math.min(total, startFrame + maxFrames);
  const halo = cfg.halo;
  const bands = spec.bands;
  let stopped = false;

  const plan = chunkPlan(startFrame, end, firstChunkFrames, chunkFrames);
  for (const [s, e] of plan) {
    const a = Math.max(0, s - halo);
    const b = Math.min(total, e + halo);
    const len = b - a;

    const buf = new Float32Array(bands * len);
    for (let band = 0; band < bands; band++) {
      buf.set(spec.mag.subarray(band * total + a, band * total + b), band * len);
    }

    const out = await session.run({ [inputName]: new TensorCtor('float32', buf, [1, 1, bands, len]) });
    const logits = out[opts.outputName ?? Object.keys(out)[0]].data;   // (1, D, len)
    const D = cfg.messageDim;
    const useful = e - s;
    const off = s - a;
    const slice = new Float32Array(D * useful);
    for (let d = 0; d < D; d++) {
      slice.set(logits.subarray(d * len + off, d * len + off + useful), d * useful);
    }
    acc.push(slice, useful, s);

    const result = acc.result();
    if (onProgress) await onProgress({ ...result, processedTo: e, totalFrames: total, analysisEnd: end });
    if (shouldStop && shouldStop()) { stopped = true; break; }
    if (earlyStop && result.verdict === 'detected') { stopped = true; break; }
  }

  return { accumulator: acc, result: acc.result(), stopped };
}
