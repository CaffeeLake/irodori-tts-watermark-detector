/**
 * Guards the biggest performance cliff there is.
 *
 * onnxruntime-web can only use multiple WASM threads on a cross-origin isolated page.
 * Without the COOP/COEP headers it keeps reporting the requested thread count but runs
 * single-threaded, measured at 555 vs 263 ms per frame (2.1x).
 *
 * Three paths have to hold:
 *   headers set            -> isolated immediately, no warning
 *   no headers, SW allowed -> web/coi-sw.js supplies them; isolated after one reload
 *   no headers, SW blocked -> cannot be fixed, so the UI must say so
 *
 *   node test/isolation.check.mjs
 */
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import { startServer, WEB_ROOT } from '../tools/serve.mjs';

const { firefox, chromium, webkit } = await import('playwright');
const browserType = { firefox, chromium, webkit }[process.env.BROWSER || 'firefox'];

const TYPES = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8', '.onnx': 'application/octet-stream',
};

/** The same files, served without the isolation headers. */
function startPlainServer() {
  const server = createServer(async (req, res) => {
    let rel = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (rel.endsWith('/')) rel += 'index.html';
    const file = path.join(WEB_ROOT, path.normalize(rel));
    if (!file.startsWith(WEB_ROOT)) { res.writeHead(403).end(); return; }
    try {
      const s = await stat(file);
      res.writeHead(200, {
        'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
        'Content-Length': s.size,
      });
      createReadStream(file).pipe(res);
    } catch { res.writeHead(404).end('not found'); }
  });
  return new Promise((r) => server.listen(0, '127.0.0.1', () => r({
    url: `http://127.0.0.1:${server.address().port}`,
    close: () => new Promise((x) => server.close(x)),
  })));
}

const failures = [];
const check = (name, cond, detail) => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
  if (!cond) failures.push(name);
};

const browser = await browserType.launch();

async function inspect(url, { serviceWorkers = 'allow', reloadOnce = false } = {}) {
  const context = await browser.newContext({ serviceWorkers });
  const page = await context.newPage();
  page.on('pageerror', (e) => console.error('   [page exception]', e.message));
  await page.goto(`${url}/index.html`);
  if (reloadOnce) {
    await page.waitForFunction(
      () => navigator.serviceWorker && navigator.serviceWorker.controller !== null,
      null, { timeout: 60000 },
    ).catch(() => {});
    await page.goto(`${url}/index.html`);
  }
  await page.waitForFunction(() => window.__detector && window.__detector.ready(), null, { timeout: 180000 });
  const out = await page.evaluate(() => ({
    isolated: self.crossOriginIsolated === true,
    sab: typeof SharedArrayBuffer !== 'undefined',
    env: document.getElementById('env').textContent,
    warnShown: !document.getElementById('warn').hidden,
    warnText: document.getElementById('warn').textContent,
    controller: !!(navigator.serviceWorker && navigator.serviceWorker.controller),
  }));
  await context.close();
  return out;
}

console.log('\nserved by tools/serve.mjs (COOP/COEP set):');
const good = await startServer({});
let r = await inspect(good.url);
console.log(`    env: ${r.env}`);
check('page is cross-origin isolated', r.isolated === true);
check('SharedArrayBuffer is available', r.sab === true);
check('no warning shown', r.warnShown === false);
check('thread count reported', /\d+ thread\(s\)/.test(r.env), r.env);
await good.close();

console.log('\nserved without the isolation headers, service worker allowed:');
const plain = await startPlainServer();
r = await inspect(plain.url, { reloadOnce: true });
console.log(`    env: ${r.env}`);
check('service worker took control', r.controller === true);
check('the service worker restores isolation', r.isolated === true);
check('SharedArrayBuffer restored', r.sab === true);
check('no warning once rescued', r.warnShown === false, r.env);
check('threads reported after rescue', /\d+ thread\(s\)/.test(r.env), r.env);

console.log('\nserved without the isolation headers, service worker blocked:');
r = await inspect(plain.url, { serviceWorkers: 'block' });
console.log(`    env: ${r.env}`);
check('page is NOT isolated', r.isolated === false);
check('warning is shown', r.warnShown === true);
check('warning explains the slowdown', /slower/i.test(r.warnText), r.warnText.slice(0, 60));
check('warning names the headers',
  /Cross-Origin-Opener-Policy/.test(r.warnText) && /Cross-Origin-Embedder-Policy/.test(r.warnText));
check('warning says how to fix it', /npm run serve/.test(r.warnText));
check('env no longer claims multiple threads',
  /single-threaded/.test(r.env), r.env);
await plain.close();

await browser.close();
console.log(`\n${failures.length ? `FAILED (${failures.length}): ${failures.join(', ')}` : 'all isolation checks passed'}`);
process.exit(failures.length ? 1 : 0);
