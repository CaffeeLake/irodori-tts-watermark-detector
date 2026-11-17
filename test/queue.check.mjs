/**
 * Browser check of the work queue: files submitted while an analysis is running must
 * appear immediately as queued cards (not only once their turn arrives), show their
 * position, be removable before they start, and run in submission order.
 *
 *   node test/queue.check.mjs
 */
import path from 'node:path';
import process from 'node:process';

import { startServer } from '../tools/serve.mjs';

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
await page.waitForFunction(() => window.__detector && window.__detector.ready(), null, { timeout: 180000 });

/** Build a silent WAV; silent files finish instantly without running inference. */
await page.evaluate(() => {
  window.__mkwav = (secs, rate = 44100) => {
    const n = Math.round(rate * secs), b = new ArrayBuffer(44 + n * 4), v = new DataView(b);
    const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
    str(0, 'RIFF'); v.setUint32(4, 36 + n * 4, true); str(8, 'WAVEfmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 3, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 4, true);
    v.setUint16(32, 4, true); v.setUint16(34, 32, true);
    str(36, 'data'); v.setUint32(40, n * 4, true);
    return b;
  };
  // A tone keeps the worker busy long enough to submit more files mid-flight.
  window.__mktone = (secs, rate = 44100) => {
    const b = window.__mkwav(secs, rate);
    const a = new Float32Array(b, 44);
    for (let i = 0; i < a.length; i++) a[i] = 0.25 * Math.sin((2 * Math.PI * 300 * i) / rate);
    return b;
  };
});

const state = () => page.evaluate(() => ({
  names: [...document.querySelectorAll('#results .card')].map((e) => ({
    name: e.querySelector('.name').textContent,
    state: e.dataset.state || 'done',
    badge: e.querySelector('.badge').textContent,
    meta: e.querySelector('.meta').textContent,
    delDisabled: e.querySelector('.del').disabled,
  })),
  count: document.getElementById('history-count').textContent,
  status: document.getElementById('status').textContent,
}));

/* ---------------------- submit a long job, then three more mid-flight ------- */
await page.evaluate(() => {
  window.__detector.enqueue([new File([window.__mktone(14)], 'running.wav')]);
});
await page.waitForFunction(
  () => document.querySelector('#results .card[data-state="running"]'), null, { timeout: 60000 },
);

await page.evaluate(() => {
  window.__detector.enqueue([
    new File([window.__mkwav(1)], 'queued-1.wav'),
    new File([window.__mkwav(1)], 'queued-2.wav'),
    new File([window.__mkwav(1)], 'queued-3.wav'),
  ]);
});

// Decode and STFT both precede the first forward pass; wait until the worker has
// reported its frame budget so the elapsed/ETA assertions are not racing those stages.
await page.waitForFunction(
  () => /expected/.test(
    document.querySelector('#results .card[data-state="running"] .meta').textContent),
  null, { timeout: 120000 },
);

let s = await state();
console.log(`  status: ${s.status}`);
for (const c of s.names) console.log(`    ${c.state.padEnd(8)} ${c.name.padEnd(12)} badge="${c.badge}" meta="${c.meta}"`);

check('all four files are visible immediately', s.names.length === 4, `${s.names.length} cards`);
check('the running file is listed first', s.names[0].state === 'running', s.names[0].state);
check('queued files follow in submission order',
  s.names.slice(1).map((c) => c.name).join(',') === 'queued-1.wav,queued-2.wav,queued-3.wav',
  s.names.slice(1).map((c) => c.name).join(','));
check('exactly one is running',
  s.names.filter((c) => c.state === 'running').length === 1);
check('the other three are queued',
  s.names.filter((c) => c.state === 'queued').length === 3);
check('queued cards say "queued"',
  s.names.filter((c) => c.state === 'queued').every((c) => c.badge === 'queued'));
check('the running card shows elapsed time',
  /analysing.*\d+\.\d s$/.test(s.names.find((c) => c.state === 'running').badge),
  s.names.find((c) => c.state === 'running').badge);
check('the running card shows an expected duration and frame budget',
  /expected/.test(s.names.find((c) => c.state === 'running').meta)
  && /\d+ of \d+ frames/.test(s.names.find((c) => c.state === 'running').meta),
  s.names.find((c) => c.state === 'running').meta);
check('queue depth is in the history count', /3 queued/.test(s.count), s.count);
check('queue positions are shown',
  s.names.some((c) => /ahead in the queue/.test(c.meta)) && s.names.some((c) => /next up/.test(c.meta)),
  s.names.filter((c) => c.state === 'queued').map((c) => c.meta).join(' | '));
check('running card cannot be deleted',
  s.names.find((c) => c.state === 'running').delDisabled === true);
check('queued cards can be deleted',
  s.names.filter((c) => c.state === 'queued').every((c) => c.delDisabled === false));

/* ------------------------- removing a queued item cancels it ---------------- */
await page.evaluate(() => {
  const el = [...document.querySelectorAll('#results .card')]
    .find((e) => e.querySelector('.name').textContent === 'queued-2.wav');
  el.querySelector('.del').click();
});
s = await state();
check('deleting a queued card removes it',
  !s.names.some((c) => c.name === 'queued-2.wav'), s.names.map((c) => c.name).join(','));
check('queue depth updated', /2 queued/.test(s.count), s.count);

/* --------------------------- the rest must still run ----------------------- */
await page.waitForFunction(
  () => document.querySelectorAll('#results .card[data-state]').length === 0,
  null, { timeout: 300000 },
);
s = await state();
console.log('  after draining:');
for (const c of s.names) console.log(`    ${c.state.padEnd(8)} ${c.name.padEnd(12)} badge="${c.badge}"`);
check('three cards remain (the deleted one never ran)', s.names.length === 3, `${s.names.length}`);
check('queued-2 never ran', !s.names.some((c) => c.name === 'queued-2.wav'));
check('remaining queued files were processed',
  ['queued-1.wav', 'queued-3.wav'].every((n) => s.names.some((c) => c.name === n)),
  s.names.map((c) => c.name).join(','));
check('finished cards are ordered newest first',
  s.names.map((c) => c.name).join(',') === 'queued-3.wav,queued-1.wav,running.wav',
  s.names.map((c) => c.name).join(','));
check('no queue depth left in the count', !/queued/.test(s.count), s.count);
check('status returns to idle', /Idle/.test(s.status), s.status);

await browser.close();
await server.close();
console.log(`\n${failures.length ? `FAILED (${failures.length}): ${failures.join(', ')}` : 'all queue checks passed'}`);
process.exit(failures.length ? 1 : 0);
