#!/usr/bin/env node
/**
 * Minimal static server for web/.
 *
 * Sends the Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy pair that
 * SharedArrayBuffer — and therefore onnxruntime-web's multi-threaded WASM backend —
 * requires.  Without these headers ORT silently falls back to a single thread.
 *
 * Usable from the CLI (`node tools/serve.mjs`) or programmatically (`startServer()`),
 * which is how the browser test spins up its own instance on a free port.
 */
import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import path from 'node:path';

export const WEB_ROOT = path.resolve(import.meta.dirname, '..', 'web');

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.wasm': 'application/wasm',
  '.onnx': 'application/octet-stream',
  '.map': 'application/json; charset=utf-8',
};

export function createStaticServer(root = WEB_ROOT) {
  return createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    let rel = decodeURIComponent(url.pathname);
    if (rel.endsWith('/')) rel += 'index.html';
    const file = path.join(root, path.normalize(rel));

    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
    res.setHeader('Cache-Control', 'no-store');

    if (!file.startsWith(root)) { res.writeHead(403).end('forbidden'); return; }
    try {
      const s = await stat(file);
      if (!s.isFile()) throw new Error('not a file');
      res.writeHead(200, {
        'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream',
        'Content-Length': s.size,
      });
      createReadStream(file).pipe(res);
    } catch {
      res.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found');
    }
  });
}

/**
 * Listen on `port` (0 = any free port). Resolves with { url, port, host, close }.
 *
 * `url` is always something a client can actually connect to: wildcard bind
 * addresses are reported back as loopback, since http://0.0.0.0/ is not routable.
 */
export function startServer({ port = 0, host = '127.0.0.1', root = WEB_ROOT } = {}) {
  const server = createStaticServer(root);
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => {
      const actual = server.address().port;
      const WILDCARD = { '0.0.0.0': '127.0.0.1', '::': '[::1]', '': '127.0.0.1' };
      const reachable = WILDCARD[host] ?? (host.includes(':') ? `[${host}]` : host);
      resolve({
        url: `http://${reachable}:${actual}`,
        port: actual,
        host,
        close: () => new Promise((r) => server.close(r)),
      });
    });
  });
}

const invokedDirectly = process.argv[1] && import.meta.url.endsWith(path.basename(process.argv[1]));
if (invokedDirectly) {
  // `--root dist` previews the built bundle with the same headers Cloudflare Pages
  // applies, which is the quickest way to sanity-check a build locally.
  const rootArg = process.argv.indexOf('--root');
  const root = rootArg > 0 && process.argv[rootArg + 1]
    ? path.resolve(process.argv[rootArg + 1])
    : WEB_ROOT;
  // Loopback by default: this serves local files with COEP relaxed, so it should not
  // be exposed on every interface unless explicitly asked for via HOST.
  const { url, host, port } = await startServer({
    port: Number(process.env.PORT || 8080),
    host: process.env.SERVE_HOST || '127.0.0.1',
    root,
  });
  console.log(`serving ${root} at ${url}/ (bound to ${host}:${port}, COOP/COEP enabled)`);
}
