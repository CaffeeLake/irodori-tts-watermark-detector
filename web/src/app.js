/** UI controller for the Irodori-TTS / SilentCipher watermark detector. */

import { decodeToMono } from './audio.js';
import {
  VARIANTS, DEFAULT_VARIANT, getVariant, minSeconds, canReachVerdict,
  verdictLabel, notes, bytesToAscii, parsePayload, payloadCapacity,
} from './detector.js';

const $ = (id) => document.getElementById(id);

const ui = {
  drop: $('drop'),
  file: $('file'),
  status: $('status'),
  bar: $('bar'),
  cancel: $('cancel'),
  results: $('results'),
  variant: $('variant'),
  payload: $('payload'),
  payloadHelp: $('payload-help'),
  window: $('window'),
  threads: $('threads'),
  earlyStop: $('earlyStop'),
  env: $('env'),
  warn: $('warn'),
  historyBar: $('history-bar'),
  historyCount: $('history-count'),
  clearAll: $('clear-all'),
};

let worker = null;
let ready = null;
let busy = false;
const queue = [];

function hardwareThreads() {
  const n = navigator.hardwareConcurrency || 4;
  return Math.max(1, Math.min(n, 8));
}

/**
 * The thread count has to survive the reload that applies it, otherwise the control
 * silently does nothing: setup used to overwrite the box with hardwareThreads() on
 * every load, so a user's choice was discarded before the worker ever saw it.
 */
const THREADS_KEY = 'irodori-wm-threads';
function savedThreads() {
  const v = Number(localStorage.getItem(THREADS_KEY));
  return Number.isFinite(v) && v >= 1 && v <= 32 ? v : hardwareThreads();
}

/** Which variants to probe, in order. */
function selectedVariants() {
  const v = ui.variant.value;
  return v === 'auto' ? ['44.1k', '16k'] : [v];
}

/**
 * The variant the payload box applies to.  In `auto` the box targets 44.1 kHz: the
 * 16 kHz model has no byte payload at all, so it keeps its payload-independent search.
 */
function payloadVariant() {
  return ui.variant.value === 'auto' ? DEFAULT_VARIANT : ui.variant.value;
}

/** `{ variantId: payload }` for the worker, or null to use the defaults. */
function selectedPayload() {
  const text = ui.payload.value.trim();
  const id = payloadVariant();
  if (!text) return null;
  const parsed = parsePayload(text, getVariant(id));
  return parsed.ok ? { [id]: text } : null;
}

/** Live validation + a hint describing what this variant accepts. */
function refreshPayloadHelp() {
  const cfg = getVariant(payloadVariant());
  const cap = payloadCapacity(cfg);
  const text = ui.payload.value.trim();
  const dflt = cfg.payload ? `"${bytesToAscii(cfg.payload)}"` : 'none';

  if (!text) {
    ui.payload.classList.remove('invalid');
    ui.payloadHelp.classList.remove('bad');
    ui.payload.placeholder = cfg.bytePayload ? bytesToAscii(cfg.payload ?? []) : `${cap.slots} symbols`;
    ui.payloadHelp.textContent = cfg.bytePayload
      ? `Blank = the Irodori default ${dflt}. Accepts ${cap.bytes} bytes as text,`
        + ' decimal (73,82,68,84,83) or hex (0x49 0x52 …).'
      : `The ${cfg.id} variant carries no byte payload (${cap.slots} slots x 2 bits`
        + ` = ${cap.bits} bits). Blank = payload-independent search, or enter`
        + ` ${cap.slots} symbols 0-${cap.maxSymbol}.`;
    return true;
  }
  const parsed = parsePayload(text, cfg);
  ui.payload.classList.toggle('invalid', !parsed.ok);
  ui.payloadHelp.classList.toggle('bad', !parsed.ok);
  ui.payloadHelp.textContent = parsed.ok
    ? `Searching for ${parsed.bytes
      ? `[${parsed.bytes.join(', ')}] = "${bytesToAscii(parsed.bytes)}"`
      : `symbols [${parsed.symbols.join(', ')}]`} on the ${cfg.id} variant.`
    : parsed.error;
  return parsed.ok;
}

function bootWorker() {
  worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  ready = new Promise((resolve, reject) => {
    worker.addEventListener('message', (ev) => {
      if (ev.data.type === 'ready') resolve(ev.data);
      if (ev.data.type === 'error') reject(new Error(ev.data.message));
    });
  });
  worker.postMessage({
    type: 'init',
    modelBaseUrl: new URL('../models/', import.meta.url).href,
    wasmPath: new URL('../vendor/', import.meta.url).href,
    threads: savedThreads(),
  });
  return ready;
}

function setStatus(text) { ui.status.textContent = text; }
function setProgress(frac) {
  ui.bar.classList.remove('indeterminate');
  ui.bar.style.width = `${Math.min(100, Math.max(0, frac * 100))}%`;
}
/** One forward pass is atomic, so mid-chunk progress is unknowable; animate instead. */
function setProgressBusy(on) {
  ui.bar.classList.toggle('indeterminate', on);
  if (!on) ui.bar.style.width = '0%';
}

/* ----------------------------------------------------------------- history --- */

/**
 * Card lifecycle: `data-state` is 'queued' while waiting, 'running' while the worker
 * is writing into it, then cleared.  Only 'running' cards resist removal -- queued
 * ones can be dropped (which also takes them out of the work queue).
 */
const cards = () => [...ui.results.querySelectorAll('.card')];
const removable = () => cards().filter((el) => el.dataset.state !== 'running');

function refreshHistory() {
  const all = cards();
  const queued = all.filter((el) => el.dataset.state === 'queued').length;
  const n = removable().length;
  ui.historyBar.hidden = all.length === 0;
  const parts = [all.length === 1 ? '1 result' : `${all.length} results`];
  if (queued) parts.push(`${queued} queued`);
  ui.historyCount.textContent = parts.join(' · ');
  ui.clearAll.disabled = n === 0;
  ui.clearAll.textContent = n < all.length ? `Clear all (${n})` : 'Clear all';
  for (const el of all) {
    el.querySelector('.del').disabled = el.dataset.state === 'running';
  }
}

function removeCard(el) {
  if (el.dataset.state === 'running') return;
  const i = queue.findIndex((q) => q.el === el);
  if (i >= 0) queue.splice(i, 1);     // never started: drop it from the work queue too
  el.remove();
  refreshHistory();
}

/* --------------------------------------------------------------- rendering --- */

function card(name, position) {
  const el = document.createElement('article');
  el.className = 'card queued';
  el.dataset.state = 'queued';
  el.innerHTML = `
    <header>
      <span class="name"></span>
      <span class="badge queued">queued</span>
      <button class="del" type="button" title="Remove from the queue"
              aria-label="Remove from the queue"></button>
    </header>
    <div class="meta"></div>
    <div class="stats"></div>
    <div class="notes"></div>
    <div class="rot" hidden><canvas width="420" height="64"></canvas>
      <div class="rot-cap">per-rotation match rate (the tall bar is the payload alignment)</div></div>`;
  el.querySelector('.name').textContent = name;
  el.querySelector('.del').textContent = '\u00d7';
  el.querySelector('.del').addEventListener('click', () => removeCard(el));
  el.querySelector('.meta').innerHTML = position > 0
    ? `<span>waiting — ${position} ahead in the queue</span>`
    : '<span>waiting…</span>';
  // Active work (running, then queued in run order) stays at the top; finished
  // results live below it, newest first.
  ui.results.insertBefore(el, firstFinishedCard());
  refreshHistory();
  return el;
}

/** The first completed card, i.e. the boundary between active work and history. */
function firstFinishedCard(except = null) {
  return cards().find((el) => el !== except && !el.dataset.state) || null;
}

/**
 * Rolling estimate of inference cost, so the card can show an expected duration up
 * front.  WASM runs this model at roughly 0.2 s per STFT frame on four threads, i.e.
 * tens of seconds for a verdict, and an unannounced wait that long reads as a hang.
 */
const MS_PER_FRAME_KEY = 'irodori-wm-ms-per-frame';
function msPerFrame() {
  const v = Number(localStorage.getItem(MS_PER_FRAME_KEY));
  return Number.isFinite(v) && v > 10 ? v : 200;
}
function recordMsPerFrame(inferMs, frames) {
  if (!frames || !inferMs) return;
  const next = 0.6 * msPerFrame() + 0.4 * (inferMs / frames);
  try { localStorage.setItem(MS_PER_FRAME_KEY, String(next)); } catch { /* private mode */ }
}

/** Live "analysing 12.3 s" readout; one forward pass can take tens of seconds. */
function startTicker(el, label) {
  const badge = el.querySelector('.badge');
  const t0 = performance.now();
  const tick = () => {
    badge.textContent = `${label} ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  };
  tick();
  const id = setInterval(tick, 100);
  return {
    relabel: (next) => { label = next; tick(); },
    stop: () => clearInterval(id),
    elapsed: () => performance.now() - t0,
  };
}

const BADGE = {
  detected: ['detected', 'ok'],
  not_detected: ['not detected', 'no'],
  inconclusive: ['inconclusive', 'warn'],
  insufficient: ['audio too short', 'warn'],
};

function render(el, r, extra = {}, { updateBadge = true } = {}) {
  const cfg = getVariant(r.variant);
  const want = r.expectedAscii
    ?? (r.expectedSymbols ? `[${r.expectedSymbols.join(', ')}]` : null);
  const [text, cls] = BADGE[r.verdict] || [r.verdict, 'warn'];
  if (updateBadge) {
    el.className = `card ${cls}`;
    const badge = el.querySelector('.badge');
    badge.textContent = text;
    badge.className = `badge ${cls}`;
  }

  el.querySelector('.meta').innerHTML = [
    `variant ${cfg.id}`,
    extra.durationSec != null ? `${extra.durationSec.toFixed(2)} s audio` : null,
    `analysed ${r.seconds.toFixed(2)} s (${r.frames} frames)`,
    extra.timing || (extra.inferMs != null ? `${(extra.inferMs / 1000).toFixed(1)} s inference` : null),
    extra.stopped ? 'stopped early' : null,
  ].filter(Boolean).map((s) => `<span>${s}</span>`).join('');

  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const rows = [['verdict', verdictLabel(r)]];

  if (want) rows.push(['searching for', `"${want}"${r.customPayload ? ' (custom)' : ' (default)'}`]);

  if (r.match != null) {
    rows.push(
      ['payload match', `${pct(r.match)} vs ${pct(r.nullMatch)} chance (margin ${pct(r.margin)})`],
      ['p-value', r.pValue < 1e-300 ? '< 1e-300' : r.pValue.toExponential(2)],
      ['log-likelihood ratio', r.logLikRatio.toFixed(2)],
    );
  } else {
    rows.push(['payload match', cfg.bytePayload
      ? 'n/a'
      : 'n/a — this variant cannot carry a byte payload']);
  }

  rows.push(['watermark structure',
    `${pct(r.structure)} (period-${cfg.messageLen}, chance-corrected)`
    + `${r.structurePresent ? ' — present' : ''}`]);

  if (cfg.bytePayload) {
    rows.push(['recovered payload', r.blind.message
      ? `"${r.blind.ascii}" [${r.blind.message.join(', ')}] conf ${pct(r.blind.confidence)}`
      : 'not recoverable yet']);
    if (r.symbolAgreement != null) {
      rows.push(['symbol agreement', `${r.symbolAgreement}/${cfg.messageLen} vs "${want}"`]);
    }
    rows.push(['frame alignment', `rotation ${r.shift}`]);
  } else {
    rows.push(['recovered symbols', r.blind.symbols
      ? `[${r.blind.symbols.join(', ')}] conf ${pct(r.blind.confidence)}`
      : 'not recoverable yet']);
  }

  el.querySelector('.stats').innerHTML = rows
    .map(([k, v]) => `<div class="row"><span class="k">${k}</span><span class="v">${v}</span></div>`)
    .join('');

  el.querySelector('.notes').innerHTML = notes(r, cfg)
    .map((n) => `<p class="note">${n}</p>`).join('');

  const rot = el.querySelector('.rot');
  rot.hidden = !r.perRotation;
  if (r.perRotation) drawRotations(el.querySelector('canvas'), r);
}

function drawRotations(canvas, r) {
  if (!r.perRotation || !r.perRotation.length) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  const n = r.perRotation.length;
  const bw = W / n;
  const max = Math.max(0.5, ...r.perRotation);
  for (let i = 0; i < n; i++) {
    const h = (r.perRotation[i] / max) * (H - 6);
    ctx.fillStyle = i === r.shift ? '#2bb673' : '#3a4a5a';
    ctx.fillRect(i * bw + 1, H - h, bw - 2, h);
  }
  ctx.strokeStyle = '#8899aa';
  ctx.setLineDash([3, 3]);
  const y = H - (r.nullMatch / max) * (H - 6);
  ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
}

/* ----------------------------------------------------------------- driving --- */

async function analyseFile(file, el) {
  const variants = selectedVariants();
  el.dataset.state = 'running';
  el.className = 'card pending';
  el.querySelector('.meta').innerHTML = '<span>decoding…</span>';
  refreshHistory();
  const ticker = startTicker(el, 'decoding');
  setProgressBusy(true);
  try {
    const bytes = await file.arrayBuffer();
    const tDecode = performance.now();
    const { pcm, sampleRate, channels, durationSec } = await decodeToMono(bytes);
    const decodeMs = performance.now() - tDecode;
    const pending = queue.length;
    const usable = variants.filter((v) => canReachVerdict(getVariant(v), durationSec));
    const tooShort = variants.filter((v) => !usable.includes(v));
    setStatus(`${file.name}: ${durationSec.toFixed(2)} s, ${channels} ch @ ${sampleRate} Hz`
      + ` — analysing…${pending ? ` (${pending} queued)` : ''}`);
    if (tooShort.length) {
      // Say so before spending any time, and be explicit that the window cannot help.
      el.querySelector('.notes').innerHTML = tooShort.map((v) => {
        const c = getVariant(v);
        return `<p class="note">${durationSec.toFixed(2)} s of audio is too short for the `
          + `${c.id} variant, which needs ${minSeconds(c).toFixed(1)} s. It is skipped`
          + `${usable.length ? '' : ' — no verdict is possible for this file'}.</p>`;
      }).join('');
    }
    // Resampling + STFT run before the first forward pass; say so rather than
    // claiming to be analysing already.
    ticker.relabel('preparing');

    let prepMs = 0;
    const done = await new Promise((resolve, reject) => {
      const frameTarget = new Map();
      const onMessage = (ev) => {
        const m = ev.data;
        if (m.type === 'prepared') {
          prepMs += m.prepMs;
          frameTarget.set(m.variant, m.analysisFrames);
          ticker.relabel(variants.length > 1 ? `analysing ${m.variant}` : 'analysing');
          const eta = (m.analysisFrames * msPerFrame()) / 1000;
          el.querySelector('.meta').innerHTML =
            `<span>variant ${m.variant}</span><span>${m.durationSec.toFixed(2)} s audio</span>`
            + `<span>${m.analysisFrames} of ${m.frames} frames</span>`
            + `<span>~${eta < 90 ? `${eta.toFixed(0)} s` : `${(eta / 60).toFixed(1)} min`} expected</span>`;
        } else if (m.type === 'progress') {
          const denom = frameTarget.get(m.variant) || m.analysisEnd || 1;
          // Partial statistics are worth showing, but the badge belongs to the ticker
          // until the run finishes -- otherwise it flickers through 'too short'.
          render(el, m, { durationSec, inferMs: m.elapsedMs }, { updateBadge: false });
          el.querySelector('.meta').insertAdjacentHTML('beforeend',
            `<span>${m.processedTo}/${denom} frames</span>`);
          // Only meaningful once more than one pass has landed.
          if (m.processedTo < denom) setProgress(m.processedTo / denom);
          else setProgressBusy(true);
          ticker.relabel(variants.length > 1 ? `analysing ${m.variant}` : 'analysing');
        } else if (m.type === 'done') {
          worker.removeEventListener('message', onMessage);
          resolve(m);
        } else if (m.type === 'error') {
          worker.removeEventListener('message', onMessage);
          reject(new Error(m.message));
        }
      };
      worker.addEventListener('message', onMessage);
      worker.postMessage({
        type: 'analyse',
        pcm: pcm.buffer,
        sampleRate,
        variants,
        payload: selectedPayload(),
        window: ui.window.value,
        earlyStop: ui.earlyStop.checked,
      }, [pcm.buffer]);
    });
    ticker.stop();

    if (!done.result) {
      el.className = 'card warn';
      el.querySelector('.badge').textContent = done.silent ? 'silent' : 'no result';
      el.querySelector('.stats').textContent = done.silent
        ? 'The file contains no signal (zero power).'
        : 'The analysis produced no result.';
      return;
    }
    const totalInfer = done.runs.reduce((a, b) => a + (b.inferMs || 0), 0);
    const totalFrames = done.runs.reduce((a, b) => a + (b.result ? b.result.frames : 0), 0);
    recordMsPerFrame(totalInfer, totalFrames);
    render(el, done.result, {
      durationSec: done.durationSec,
      inferMs: totalInfer,
      // Breaking the wall clock down makes a slow run diagnosable: decoding and the
      // DSP front end are usually negligible, inference is not.
      timing: `${(decodeMs / 1000).toFixed(1)} s decode`
        + ` · ${(prepMs / 1000).toFixed(1)} s resample+STFT`
        + ` · ${(totalInfer / 1000).toFixed(1)} s inference`,
      stopped: done.stopped && !done.cancelled,
    });
    if (done.runs.length > 1) {
      const tried = done.runs
        .map((x) => `${x.variant}: ${x.result ? x.result.verdict : 'n/a'}`).join(', ');
      el.querySelector('.meta').insertAdjacentHTML('beforeend', `<span>tried ${tried}</span>`);
    }
    setProgress(1);
  } catch (err) {
    el.className = 'card no';
    el.querySelector('.badge').textContent = 'error';
    el.querySelector('.stats').textContent = String(err.message || err);
  } finally {
    ticker.stop();
    // The worker is done with this card: it may now be removed, and it drops out of
    // the active block to the top of the finished list.
    ui.results.insertBefore(el, firstFinishedCard(el));
    delete el.dataset.state;
    refreshHistory();
  }
}

/** Keep every queued card's "N ahead in the queue" line accurate. */
function refreshQueuePositions() {
  queue.forEach((q, i) => {
    if (q.el.dataset.state !== 'queued') return;
    q.el.querySelector('.meta').innerHTML = i === 0
      ? '<span>next up…</span>'
      : `<span>waiting — ${i} ahead in the queue</span>`;
  });
}

async function pump() {
  if (busy) return;
  busy = true;
  ui.cancel.disabled = false;
  while (queue.length) {
    const { file, el } = queue.shift();
    refreshQueuePositions();
    await analyseFile(file, el);
  }
  busy = false;
  ui.cancel.disabled = true;
  setStatus('Idle.');
  setProgressBusy(false);
}

function enqueue(files) {
  const list = [...files];
  if (!list.length) return;
  // Create the cards up front so files dropped during an analysis are visible
  // immediately instead of appearing only once their turn comes round.
  for (const f of list) queue.push({ file: f, el: card(f.name, queue.length + (busy ? 1 : 0)) });
  refreshQueuePositions();
  if (busy) {
    const n = queue.length;
    setStatus(`${ui.status.textContent.replace(/ \(\d+ queued\)$/, '')} (${n} queued)`);
  }
  pump();
}

/* ------------------------------------------------------------------- setup --- */

ui.threads.value = String(savedThreads());

ui.file.addEventListener('change', () => enqueue(ui.file.files));
ui.drop.addEventListener('click', () => ui.file.click());
ui.drop.addEventListener('dragover', (e) => { e.preventDefault(); ui.drop.classList.add('over'); });
ui.drop.addEventListener('dragleave', () => ui.drop.classList.remove('over'));
ui.drop.addEventListener('drop', (e) => {
  e.preventDefault();
  ui.drop.classList.remove('over');
  enqueue(e.dataTransfer.files);
});
ui.payload.addEventListener('input', refreshPayloadHelp);
ui.variant.addEventListener('change', refreshPayloadHelp);
refreshPayloadHelp();

ui.cancel.addEventListener('click', () => worker && worker.postMessage({ type: 'cancel' }));
ui.threads.addEventListener('change', () => {
  const v = Math.max(1, Math.min(32, Number(ui.threads.value) || hardwareThreads()));
  try { localStorage.setItem(THREADS_KEY, String(v)); } catch { /* private mode */ }
  location.reload();   // the WASM thread pool is fixed when the runtime initialises
});

ui.clearAll.addEventListener('click', () => {
  // Leaves a still-running analysis in place; its own button re-enables when it finishes.
  // Queued cards are dropped from the work queue as well, so they never start.
  for (const el of removable()) removeCard(el);
  refreshQueuePositions();
  refreshHistory();
});

refreshHistory();

setStatus('Loading WebAssembly runtime and model…');
bootWorker().then((info) => {
  const need = Object.values(VARIANTS)
    .map((v) => `${v.id} ≥ ${minSeconds(v).toFixed(1)} s`).join(', ');
  // onnxruntime-web's thread pool needs SharedArrayBuffer, which needs the page to be
  // cross-origin isolated.  Without it ORT keeps reporting the requested thread count
  // but runs on one thread, which measured 2.4x slower -- by far the biggest
  // performance cliff there is, and invisible unless we say so.
  const isolated = self.crossOriginIsolated === true;
  const threaded = isolated && typeof SharedArrayBuffer !== 'undefined';
  ui.env.textContent = 'onnxruntime-web WASM · '
    + (threaded ? `${info.threads} thread(s)` : 'single-threaded (no SharedArrayBuffer)')
    + ` · Irodori payload "${bytesToAscii(VARIANTS[DEFAULT_VARIANT].payload)}"`
    + ` · verdict needs ${need}`;
  // src/coi.js may be about to reload the page with a service worker that supplies the
  // headers; do not flash the banner in that window.
  if (!threaded && !window.__coiPending) {
    ui.warn.hidden = false;
    ui.warn.innerHTML = '<strong>Running single-threaded — about 2.1x slower than it '
      + 'needs to be.</strong><br>onnxruntime-web can only use multiple WASM threads on '
      + 'a cross-origin isolated page, which requires the '
      + '<code>Cross-Origin-Opener-Policy: same-origin</code> and '
      + '<code>Cross-Origin-Embedder-Policy: require-corp</code> response headers. '
      + 'Serve this page with <code>npm run serve</code> (which sets them) rather than '
      + 'from <code>file://</code> or a plain static server.';
  }
  setStatus('Ready — drop audio or video files to test.');
}).catch((err) => {
  setStatus(`Failed to initialise: ${err.message}`);
});

// Hook used by the browser checks to drive the page without touching the DOM.
window.__detector = {
  /** @param {ArrayBuffer} bytes */
  analyseBuffer: async (bytes, opts = {}) => {
    await ready;
    const { pcm, sampleRate } = await decodeToMono(bytes);
    return new Promise((resolve) => {
      const onMessage = (ev) => {
        const { type } = ev.data;
        if (type !== 'done' && type !== 'error') return;
        worker.removeEventListener('message', onMessage);
        resolve(ev.data);
      };
      worker.addEventListener('message', onMessage);
      worker.postMessage({
        type: 'analyse',
        pcm: pcm.buffer,
        sampleRate,
        variants: opts.variants ?? [DEFAULT_VARIANT],
        window: opts.window ?? 'min',
        chunkFrames: opts.chunkFrames ?? 43,
        earlyStop: opts.earlyStop ?? true,
      }, [pcm.buffer]);
    });
  },
  ready: () => ready,
  /** Raw worker handle, used by the performance probe. */
  get worker() { return worker; },
  /** Drive the normal UI path (history cards included) with File objects. */
  enqueue: (files) => enqueue(files),
};
