/**
 * Browser check for the result-history controls: per-card delete and clear-all,
 * including the guard that a still-running analysis cannot be removed.
 *
 *   node test/history.check.mjs
 */
import path from 'node:path';
import process from 'node:process';

import { startServer } from '../tools/serve.mjs';

const ROOT = path.resolve(import.meta.dirname, '..');
const { firefox, chromium, webkit } = await import('playwright');
const browserType = { firefox, chromium, webkit }[process.env.BROWSER || 'firefox'];

const server = await startServer({ port: Number(process.env.PORT || 0) });
const failures = [];
const check = (name, cond, detail) => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'}  ${name}${detail ? ` — ${detail}` : ''}`);
  if (!cond) failures.push(name);
};

const browser = await browserType.launch();
const page = await browser.newPage();
page.on('pageerror', (e) => { console.error('   [page exception]', e.message); failures.push('pageerror'); });

await page.goto(`${server.url}/index.html`);
await page.waitForFunction(() => window.__detector && window.__detector.ready(), null, { timeout: 120000 });

const state = () => page.evaluate(() => ({
  cards: document.querySelectorAll('#results .card').length,
  busy: document.querySelectorAll('#results .card[data-state="running"]').length,
  active: document.querySelectorAll('#results .card[data-state]').length,
  barHidden: document.getElementById('history-bar').hidden,
  count: document.getElementById('history-count').textContent,
  clearDisabled: document.getElementById('clear-all').disabled,
  clearText: document.getElementById('clear-all').textContent,
  delDisabled: [...document.querySelectorAll('#results .card .del')].map((b) => b.disabled),
}));

/* ------------------------------------------------- empty history at startup --- */
let s = await state();
check('history bar hidden when empty', s.barHidden === true, `hidden=${s.barHidden}`);
check('no cards at startup', s.cards === 0, `cards=${s.cards}`);

/* ------------------------------- three finished cards (short silent WAVs) ----- */
await page.evaluate(async () => {
  // A tiny silent WAV finishes instantly ("silent" verdict) without running inference.
  const wav = (secs) => {
    const n = 44100 * secs, b = new ArrayBuffer(44 + n * 4), v = new DataView(b);
    const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
    str(0, 'RIFF'); v.setUint32(4, 36 + n * 4, true); str(8, 'WAVEfmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 3, true); v.setUint16(22, 1, true);
    v.setUint32(24, 44100, true); v.setUint32(28, 44100 * 4, true);
    v.setUint16(32, 4, true); v.setUint16(34, 32, true);
    str(36, 'data'); v.setUint32(40, n * 4, true);
    return b;
  };
  const files = [1, 2, 3].map((i) => new File([wav(1)], `silent-${i}.wav`));
  window.__detector.enqueue(files);
});
// Cards now appear as soon as the files are queued, so wait for the queue to drain
// (no card left carrying a data-state) rather than merely for the cards to exist.
await page.waitForFunction(() => document.querySelectorAll('#results .card').length === 3
  && document.querySelectorAll('#results .card[data-state]').length === 0,
null, { timeout: 120000 });

s = await state();
check('three results listed', s.cards === 3, `cards=${s.cards}`);
check('history bar visible', s.barHidden === false);
check('count label', s.count === '3 results', `"${s.count}"`);
check('clear-all enabled', s.clearDisabled === false);
check('clear-all label has no partial count', s.clearText === 'Clear all', `"${s.clearText}"`);
check('all delete buttons enabled', s.delDisabled.every((d) => d === false), JSON.stringify(s.delDisabled));

/* ----------------------------------------------------- delete a single card --- */
const firstName = await page.evaluate(() => document.querySelector('#results .card .name').textContent);
await page.click('#results .card .del');
s = await state();
check('one card removed', s.cards === 2, `cards=${s.cards}`);
check('count label updated', s.count === '2 results', `"${s.count}"`);
const names = await page.evaluate(() => [...document.querySelectorAll('#results .card .name')].map((e) => e.textContent));
check('the clicked card is the one removed', !names.includes(firstName), `${firstName} vs ${names}`);

/* ---------------------------------------------------------------- clear all --- */
await page.click('#clear-all');
s = await state();
check('all cards cleared', s.cards === 0, `cards=${s.cards}`);
check('history bar hidden again', s.barHidden === true);

/* ------------------------- a running analysis must not be removable ---------- */
await page.evaluate(() => {
  const n = 48000 * 20, b = new ArrayBuffer(44 + n * 4), v = new DataView(b);
  const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + n * 4, true); str(8, 'WAVEfmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 3, true); v.setUint16(22, 1, true);
  v.setUint32(24, 48000, true); v.setUint32(28, 48000 * 4, true);
  v.setUint16(32, 4, true); v.setUint16(34, 32, true);
  str(36, 'data'); v.setUint32(40, n * 4, true);
  const a = new Float32Array(b, 44);
  for (let i = 0; i < n; i++) a[i] = 0.2 * Math.sin((2 * Math.PI * 300 * i) / 48000);
  window.__detector.enqueue([new File([b], 'long-tone.wav')]);
});
await page.waitForFunction(
  () => document.querySelectorAll('#results .card[data-state="running"]').length === 1,
  null, { timeout: 60000 },
);
s = await state();
check('running card is marked busy', s.busy === 1, `busy=${s.busy}`);
check('its delete button is disabled', s.delDisabled[0] === true, JSON.stringify(s.delDisabled));
check('clear-all is disabled while it is the only card', s.clearDisabled === true, `disabled=${s.clearDisabled}`);

await page.click('#results .card .del', { force: true });
s = await state();
check('clicking a disabled delete does nothing', s.cards === 1, `cards=${s.cards}`);
await page.click('#clear-all', { force: true });
s = await state();
check('clear-all keeps the running card', s.cards === 1, `cards=${s.cards}`);

// Cancel and wait for it to settle, then confirm it becomes removable.
await page.click('#cancel');
await page.waitForFunction(
  () => document.querySelectorAll('#results .card[data-state]').length === 0,
  null, { timeout: 180000 },
);
s = await state();
check('card becomes removable once finished', s.delDisabled[0] === false, JSON.stringify(s.delDisabled));
check('clear-all re-enabled', s.clearDisabled === false);
await page.click('#clear-all');
s = await state();
check('history empty after final clear', s.cards === 0 && s.barHidden === true, `cards=${s.cards}`);

await browser.close();
await server.close();
console.log(`\n${failures.length ? `FAILED (${failures.length}): ${failures.join(', ')}` : 'all history checks passed'}`);
process.exit(failures.length ? 1 : 0);
