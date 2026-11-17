/**
 * Analysis worker.  Keeps the heavy WebAssembly inference off the UI thread.
 *
 * One ONNX session is created lazily per SilentCipher variant, so loading the page
 * costs nothing extra if you only ever test 44.1 kHz audio.
 *
 * Protocol
 *   main -> worker : {type:'init', modelBaseUrl, wasmPath, threads}
 *                    {type:'analyse', pcm, sampleRate, variants:[id], maxSeconds,
 *                                     chunkFrames, earlyStop}
 *                    {type:'cancel'}
 *   worker -> main : {type:'ready'|'prepared'|'progress'|'variantDone'|'done'|'error', ...}
 */

import * as ort from '../vendor/ort.wasm.min.mjs';
import { prepare, runDetector } from './pipeline.js';
import { frameCount } from './dsp.js';
import {
  Accumulator, getVariant, DEFAULT_VARIANT, windowFrames, minSeconds, withPayload,
} from './detector.js';

const sessions = new Map();
let modelBaseUrl = null;
let cancelled = false;

async function init({ modelBaseUrl: base, wasmPath, threads, preload = DEFAULT_VARIANT }) {
  modelBaseUrl = base;
  ort.env.wasm.wasmPaths = wasmPath;
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.proxy = false;
  ort.env.logLevel = 'error';

  // Build the default session now rather than on the first file.  Otherwise the page
  // claims to be "ready" in ~0.5 s and the user's first analysis silently pays for
  // compiling the 15 MB WASM runtime, spawning its thread pool and parsing a 9.5 MB
  // model on top of the inference itself.
  const t0 = performance.now();
  if (preload) await sessionFor(preload);
  return {
    threads: ort.env.wasm.numThreads,
    preloaded: preload || null,
    preloadMs: performance.now() - t0,
  };
}

async function sessionFor(id) {
  if (sessions.has(id)) return sessions.get(id);
  const cfg = getVariant(id);
  const promise = ort.InferenceSession.create(new URL(cfg.model, modelBaseUrl).href, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  });
  sessions.set(id, promise);
  return promise;
}

async function analyseVariant(audio, sampleRate, id, opts) {
  // An explicit payload replaces the variant's default expected word.  If it cannot be
  // represented by this variant, skip it and say so rather than silently falling back
  // to the default word and reporting a search the caller never asked for.
  const requested = opts.payload?.[id] ?? opts.payload ?? null;
  let cfg;
  try {
    cfg = withPayload(getVariant(id), requested);
  } catch (err) {
    const base = getVariant(id);
    const result = new Accumulator(base).result();
    result.payloadError = String(err.message || err);
    return {
      variant: id, silent: false, result, payloadError: result.payloadError,
      stopped: false, inferMs: 0, prepMs: 0,
      durationSec: audio.length / sampleRate, totalFrames: 0, analysisFrames: 0,
    };
  }

  // A file shorter than the variant's minimum window can never reach a verdict, no
  // matter what the window is set to.  Detect that from the length alone and report it
  // immediately: running the decoder anyway costs ~10 s to produce "too short".
  const fileFrames = frameCount(
    Math.round((audio.length * cfg.sampleRate) / sampleRate), cfg.nFft, cfg.hopLength,
  );
  const durationSec = audio.length / sampleRate;
  if (fileFrames < cfg.thresholds.minFrames) {
    const result = new Accumulator(cfg).result();
    result.availableFrames = fileFrames;
    result.requiredFrames = cfg.thresholds.minFrames;
    result.availableSeconds = durationSec;
    result.requiredSeconds = minSeconds(cfg);
    self.postMessage({
      type: 'prepared',
      variant: id,
      frames: fileFrames,
      analysisFrames: 0,
      durationSec,
      tooShort: true,
      prepMs: 0,
    });
    return {
      variant: id, silent: false, result, tooShort: true, stopped: false,
      inferMs: 0, prepMs: 0, durationSec, totalFrames: fileFrames, analysisFrames: 0,
    };
  }

  const session = await sessionFor(id);

  // Resolve the budget first so the front end only transforms what is needed.
  const budget = windowFrames(opts.window, cfg);
  const t0 = performance.now();
  const spec = prepare(audio, sampleRate, cfg, { maxFrames: budget });
  const prepMs = performance.now() - t0;
  if (spec.silent) return { variant: id, silent: true, result: null };

  const maxFrames = Math.min(spec.frames, budget);

  self.postMessage({
    type: 'prepared',
    variant: id,
    frames: spec.fileFrames,
    analysisFrames: maxFrames,
    durationSec: spec.durationSec,
    truncated: spec.truncated,
    prepMs,
  });

  const t1 = performance.now();
  const { result, stopped } = await runDetector(spec, session, ort.Tensor, {
    cfg,
    chunkFrames: opts.chunkFrames,
    firstChunkFrames: opts.firstChunkFrames,
    maxFrames,
    earlyStop: opts.earlyStop,
    inputName: session.inputNames[0],
    outputName: session.outputNames[0],
    shouldStop: () => cancelled,
    onProgress: async (p) => {
      self.postMessage({
        type: 'progress', variant: id, ...p, elapsedMs: performance.now() - t1,
      });
    },
  });

  return {
    variant: id,
    silent: false,
    result,
    stopped,
    inferMs: performance.now() - t1,
    totalFrames: spec.fileFrames,
    analysisFrames: maxFrames,
    durationSec: spec.durationSec,
  };
}

async function analyse(msg) {
  cancelled = false;
  const audio = new Float32Array(msg.pcm);
  const ids = msg.variants && msg.variants.length ? msg.variants : [DEFAULT_VARIANT];
  const opts = {
    window: msg.window ?? 'min',
    payload: msg.payload ?? null,
    chunkFrames: msg.chunkFrames,
    firstChunkFrames: msg.firstChunkFrames,
    earlyStop: msg.earlyStop ?? true,
  };

  const runs = [];
  for (const id of ids) {
    const run = await analyseVariant(audio, msg.sampleRate, id, opts);
    runs.push(run);
    self.postMessage({ type: 'variantDone', ...run });
    if (cancelled) break;
    // Stop probing further variants once one of them is conclusive.
    if (opts.earlyStop && run.result && run.result.verdict === 'detected') break;
  }

  // Prefer a positive; otherwise report the most informative (most frames) run.
  const best = runs.find((r) => r.result && r.result.verdict === 'detected')
    ?? runs.find((r) => r.result && r.result.verdict === 'inconclusive')
    ?? runs.find((r) => r.result && r.result.verdict === 'not_detected')
    ?? runs.find((r) => r.result)
    ?? runs[0];

  self.postMessage({
    type: 'done',
    cancelled,
    runs: runs.map(({ variant, result, silent, stopped, inferMs, durationSec }) => ({
      variant, result, silent, stopped, inferMs, durationSec,
    })),
    ...best,
  });
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      self.postMessage({ type: 'ready', ...(await init(msg)) });
    } else if (msg.type === 'analyse') {
      await analyse(msg);
    } else if (msg.type === 'cancel') {
      cancelled = true;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String((err && err.stack) || err) });
  }
};
