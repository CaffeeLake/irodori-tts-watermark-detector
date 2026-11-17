/**
 * Cross-origin isolation service worker.
 *
 * onnxruntime-web's WASM thread pool needs SharedArrayBuffer, which the browser only
 * grants to cross-origin isolated pages — i.e. pages served with
 *   Cross-Origin-Opener-Policy: same-origin
 *   Cross-Origin-Embedder-Policy: require-corp
 * Without them ORT silently runs on one thread, measured at 555 vs 263 ms per frame
 * (2.1x slower).
 *
 * Cloudflare Pages can set those headers (see _headers), but **GitHub Pages cannot set
 * response headers at all**.  A service worker can: it re-issues each request and adds
 * the headers to the response it hands back, which makes the page isolated from the
 * second load onwards.  src/coi.js registers this and reloads once when needed.
 *
 * This worker caches nothing — every request is passed straight through — so it cannot
 * serve stale content.
 */

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  const req = event.request;

  // Range/cache-only requests must not be rewritten: reconstructing the response would
  // drop the 206 semantics the browser relies on.
  if (req.cache === 'only-if-cached' && req.mode !== 'same-origin') return;

  event.respondWith((async () => {
    try {
      const res = await fetch(req);
      // Opaque responses (status 0) carry no readable headers; pass them through.
      if (res.status === 0) return res;
      const headers = new Headers(res.headers);
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
      return new Response(res.body, {
        status: res.status,
        statusText: res.statusText,
        headers,
      });
    } catch (err) {
      return new Response(String(err), { status: 502, statusText: 'coi-sw fetch failed' });
    }
  })());
});
