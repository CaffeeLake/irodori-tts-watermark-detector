/**
 * Registers coi-sw.js when the page is not already cross-origin isolated, then reloads
 * once so the worker can serve the isolating headers.
 *
 * Loaded as a classic (non-module) script from <head> so it runs before the module
 * graph — there is no point fetching 25 MB of model and runtime only to reload.
 *
 * A no-op when the server already sets COOP/COEP (`npm run serve`, Cloudflare Pages
 * via _headers), which is why it is safe to ship unconditionally.  Needed on GitHub
 * Pages, which cannot set response headers at all.
 */
(() => {
  'use strict';

  if (window.crossOriginIsolated) return;              // nothing to fix
  if (!('serviceWorker' in navigator)) return;         // cannot be fixed this way
  // Service workers require a secure context; file:// and plain http:// (non-localhost)
  // are out of reach, so do not bother trying.
  if (!window.isSecureContext) return;

  // Reload at most once per tab: if isolation still fails (some browsers, or a
  // disabled worker) we must not loop.
  const FLAG = 'irodori-wm-coi-reloaded';
  const alreadyTried = sessionStorage.getItem(FLAG) === '1';

  // Tells app.js a reload may be imminent, so it does not flash the
  // "running single-threaded" banner on a page that is about to be replaced.
  window.__coiPending = !alreadyTried;
  const giveUp = () => { window.__coiPending = false; };

  navigator.serviceWorker.register(
    new URL('../coi-sw.js', document.currentScript.src).href,
    { scope: './' },
  ).then((reg) => {
    if (alreadyTried) { giveUp(); return; }
    const reload = () => {
      sessionStorage.setItem(FLAG, '1');
      window.location.reload();
    };
    // Already controlling us? Then the headers should have been applied; if they were
    // not, the page simply stays single-threaded and warns.
    if (navigator.serviceWorker.controller) { giveUp(); return; }
    if (reg.active) { reload(); return; }
    reg.addEventListener('updatefound', () => {
      const sw = reg.installing || reg.waiting;
      if (!sw) { giveUp(); return; }
      sw.addEventListener('statechange', () => { if (sw.state === 'activated') reload(); });
    });
  }).catch(() => { giveUp(); /* isolation stays off; the UI explains the cost */ });
})();
