/**
 * Verifies the built bundle the way the two target hosts actually serve it.
 *
 *   plain server, no response headers  -> GitHub Pages. The service worker must make
 *                                         the page cross-origin isolated after one
 *                                         reload, so inference keeps its thread pool.
 *   server honouring _headers          -> Cloudflare Pages. Isolated on the first load.
 *   served under /some/repo/           -> GitHub Pages project sites. Nothing may
 *                                         depend on being at the domain root.
 *
 * Run ./build.sh first.
 *
 *   node test/dist.check.mjs
 */
import { createServer } from 'node:http';
import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { stat } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const ROOT = path.resolve(import.meta.dirname, '..');
const DIST = process.env.DIST ? path.resolve(process.env.DIST) : path.join(ROOT, 'dist');
if (!existsSync(path.join(DIST, 'index.html'))) {
  console.error(`no bundle at ${DIST} — run ./build.sh first`);
  process.exit(1);
}

const { firefox, chromium, webkit } = await import('playwright');
const browserType = { firefox, chromium, webkit }[process.env.BROWSER || 'firefox'];

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8', '.onnx': 'application/octet-stream',
};

/** Parse the Cloudflare-style _headers file into [globPrefix, headers] rules. */
function parseHeaders(file) {
  if (!existsSync(file)) return [];
  const rules = [];
  let current = null;
  for (const line of readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    if (!/^\s/.test(line)) {
      current = { pattern: line.trim(), headers: {} };
      rules.push(current);
    } else if (current) {
      const i = line.indexOf(':');
      if (i > 0) current.headers[line.slice(0, i).trim()] = line.slice(i + 1).trim();
    }
  }
  return rules;
}

/**
 * @param {boolean} applyHeaders emulate Cloudflare Pages (_headers) or a bare host
 * @param {string} base          mount point, e.g. '/' or '/Irodori-TTS-Watermark-Detector/'
 */
function serve(applyHeaders, base = '/') {
  const rules = applyHeaders ? parseHeaders(path.join(DIST, '_headers')) : [];
  const server = createServer(async (req, res) => {
    let rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (!rel.startsWith(base)) { res.writeHead(404).end('outside base'); return; }
    rel = '/' + rel.slice(base.length);
    // Side route for test audio, so the bundle under test is not polluted with it.
    if (rel.startsWith('/__sample/')) {
      const f = path.join(ROOT, 'test', 'samples', path.basename(rel));
      try {
        const st = await stat(f);
        res.writeHead(200, {
          'Content-Type': rel.endsWith('.wav') ? 'audio/wav' : 'audio/webm',
          'Content-Length': st.size,
          'Cross-Origin-Opener-Policy': 'same-origin',
          'Cross-Origin-Embedder-Policy': 'require-corp' });
        createReadStream(f).pipe(res);
      } catch { res.writeHead(404).end(); }
      return;
    }
    if (rel.endsWith('/')) rel += 'index.html';
    const file = path.join(DIST, path.normalize(rel));
    if (!file.startsWith(DIST)) { res.writeHead(403).end(); return; }
    try {
      const s = await stat(file);
      const headers = {
        'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
        'Content-Length': s.size,
      };
      for (const rule of rules) {
        const p = rule.pattern.replace(/\*$/, '');
        if (rule.pattern === '/*' || rel === rule.pattern || rel.startsWith(p)) {
          Object.assign(headers, rule.headers);
        }
      }
      res.writeHead(200, headers);
      createReadStream(file).pipe(res);
    } catch { res.writeHead(404).end('not found'); }
  });
  return new Promise((r) => server.listen(0, '127.0.0.1', () => r({
    url: `http://127.0.0.1:${server.address().port}${base}`,
    close: () => new Promise((x) => server.close(x)),
  })));
}

const failures = [];
const check = (name, cond, detail) => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
  if (!cond) failures.push(name);
};

/* --------------------------------------------------------- bundle contents --- */
console.log(`\nbundle at ${path.relative(ROOT, DIST)}:`);
for (const f of ['index.html', 'coi-sw.js', '_headers', '.nojekyll', 'src/app.js',
  'models/meta.json', 'models/dec_m_0.44k1.onnx', 'vendor/ort-wasm-simd-threaded.wasm']) {
  check(`contains ${f}`, existsSync(path.join(DIST, f)));
}
check('no source maps shipped', !existsSync(path.join(DIST, 'vendor', 'ort.wasm.min.mjs.map')));
const hdr = readFileSync(path.join(DIST, '_headers'), 'utf8');
check('_headers sets COOP', /Cross-Origin-Opener-Policy:\s*same-origin/.test(hdr));
check('_headers sets COEP', /Cross-Origin-Embedder-Policy:\s*require-corp/.test(hdr));
check('index.html loads the COI registration before the module graph',
  /<script src="\.\/src\/coi\.js"><\/script>/.test(readFileSync(path.join(DIST, 'index.html'), 'utf8')));

const browser = await browserType.launch();

async function boot(url, { reloadOnce = false } = {}) {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('   [page exception]', e.message));
  await page.goto(`${url}index.html`);
  if (reloadOnce) {
    // The service worker can only add headers to responses it serves, so isolation
    // starts at the next navigation. Wait for it to take control, then reload.
    await page.waitForFunction(
      () => navigator.serviceWorker && navigator.serviceWorker.controller !== null,
      null, { timeout: 60000 },
    ).catch(() => {});
    await page.goto(`${url}index.html`);
  }
  await page.waitForFunction(() => window.__detector && window.__detector.ready(), null, { timeout: 180000 });
  const info = await page.evaluate(() => ({
    isolated: self.crossOriginIsolated === true,
    sab: typeof SharedArrayBuffer !== 'undefined',
    env: document.getElementById('env').textContent,
    warned: !document.getElementById('warn').hidden,
    controller: !!(navigator.serviceWorker && navigator.serviceWorker.controller),
  }));
  return { page, info };
}

/* ------------------------------------- Cloudflare Pages (headers honoured) --- */
console.log('\nemulating Cloudflare Pages (_headers applied):');
let s = await serve(true);
let { page, info } = await boot(s.url);
console.log(`    ${info.env}`);
check('isolated on first load', info.isolated === true);
check('SharedArrayBuffer available', info.sab === true);
check('no slowdown banner', info.warned === false);
check('threads reported', /\d+ thread\(s\)/.test(info.env), info.env);
await page.close();
await s.close();

/* ------------------------------------------- GitHub Pages (no headers) ------ */
console.log('\nemulating GitHub Pages (no response headers, service worker must fix it):');
s = await serve(false);
({ page, info } = await boot(s.url, { reloadOnce: true }));
console.log(`    ${info.env}`);
check('service worker took control', info.controller === true);
check('isolated after the reload', info.isolated === true,
  `isolated=${info.isolated} sab=${info.sab}`);
check('SharedArrayBuffer available', info.sab === true);
check('no slowdown banner', info.warned === false, info.env);

// Loading is not enough: the built bundle must actually detect a watermark.
const run = await page.evaluate(async () => {
  // The uncompressed sample, so the payload resolves byte-exactly at the minimum
  // window; Opus coding costs enough repetitions that it needs a wider one.
  const bytes = await (await fetch('./__sample/44k1_watermarked.wav')).arrayBuffer();
  const res = await window.__detector.analyseBuffer(bytes, {
    variants: ['44.1k'], window: 'min', earlyStop: true,
  });
  return {
    verdict: res.result ? res.result.verdict : 'n/a',
    ascii: res.result ? res.result.blind.ascii : null,
    margin: res.result ? res.result.margin : null,
    inferMs: res.inferMs,
    frames: res.result ? res.result.frames : 0,
  };
});
console.log(`    detection: verdict=${run.verdict} blind=${run.ascii} `
  + `margin=${run.margin.toFixed(3)} ${run.frames} frames in ${(run.inferMs / 1000).toFixed(1)} s`);
check('the built bundle detects the watermark', run.verdict === 'detected', run.verdict);
check('and reads the payload back', run.ascii === 'IRDTS', String(run.ascii));
check('at multi-threaded speed (<450 ms/frame)', run.inferMs / run.frames < 450,
  `${(run.inferMs / run.frames).toFixed(0)} ms/frame`);
await page.close();
await s.close();

/* -------------------------------- GitHub Pages project site subpath --------- */
console.log('\nemulating a GitHub Pages project site at /Irodori-TTS-Watermark-Detector/:');
s = await serve(true, '/Irodori-TTS-Watermark-Detector/');
({ page, info } = await boot(s.url));
console.log(`    ${info.env}`);
check('works under a subpath', info.isolated === true);
check('model and runtime resolved relatively', /\d+ thread\(s\)/.test(info.env), info.env);
await page.close();
await s.close();

await browser.close();
console.log(`\n${failures.length ? `FAILED (${failures.length}): ${failures.join(', ')}` : 'all dist checks passed'}`);
process.exit(failures.length ? 1 : 0);
