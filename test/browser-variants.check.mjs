/**
 * Real-browser check of both SilentCipher variants, including the "auto" probe order.
 *
 * Uses the generated samples in test/samples/ (tools/make_samples.py), so it does not
 * depend on any audio being present in the repo root.
 *
 *   node test/browser-variants.check.mjs
 */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

import { startServer } from '../tools/serve.mjs';
import { VARIANTS } from '../web/src/detector.js';

const ROOT = path.resolve(import.meta.dirname, '..');
const SAMPLES = path.join(ROOT, 'test', 'samples');
const slug = { '44.1k': '44k1', '16k': '16k' };

const { firefox, chromium, webkit } = await import('playwright');
const browserType = { firefox, chromium, webkit }[process.env.BROWSER || 'firefox'];

const metaFor = (v) => {
  const p = path.join(SAMPLES, `${slug[v]}_samples.json`);
  return existsSync(p) ? JSON.parse(readFileSync(p, 'utf8')) : null;
};
if (!metaFor('44.1k') || !metaFor('16k')) {
  console.error('missing samples — run: npm run samples');
  process.exit(1);
}

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
console.log(`\nbrowser=${process.env.BROWSER || 'firefox'}  `
  + `${await page.evaluate(() => document.getElementById('env').textContent)}\n`);

async function analyse(file, variants, win = 'all') {
  const payload = Array.from(readFileSync(file));
  return page.evaluate(async ({ payload, variants, win }) => {
    const buf = new Uint8Array(payload).buffer;
    const res = await window.__detector.analyseBuffer(buf, {
      variants, window: win, earlyStop: false,
    });
    if (res.type === 'error') return { error: res.message };
    const pick = (r) => (r ? {
      variant: r.variant, verdict: r.verdict, structure: r.structure,
      margin: r.margin, pValue: r.pValue, frames: r.frames,
      ascii: r.blind.ascii, symbols: r.blind.symbols, conf: r.blind.confidence,
    } : null);
    return {
      best: pick(res.result),
      runs: (res.runs || []).map((x) => ({ variant: x.variant, ...pick(x.result) })),
    };
  }, { payload, variants, win });
}

const fmt = (r) => `${r.variant} verdict=${r.verdict} struct=${r.structure.toFixed(3)}`
  + (r.margin != null ? ` margin=${r.margin.toFixed(3)} p=${r.pValue.toExponential(1)}` : '')
  + ` frames=${r.frames} out=${r.ascii ?? (r.symbols ? `[${r.symbols.join(',')}]` : 'null')}`;

/* ------------------------------------------------------ explicit variants --- */
console.log('44.1 kHz sample via the 44.1 kHz detector:');
let r = await analyse(path.join(SAMPLES, '44k1_watermarked.wav'), ['44.1k']);
if (r.error) check('44.1k', false, r.error);
else {
  console.log(`    ${fmt(r.best)}`);
  check('44.1k -> detected', r.best.verdict === 'detected', r.best.verdict);
  check('44.1k -> payload IRDTS', r.best.ascii === 'IRDTS', String(r.best.ascii));
}

console.log('\n16 kHz sample via the 16 kHz detector:');
r = await analyse(path.join(SAMPLES, '16k_watermarked.wav'), ['16k']);
if (r.error) check('16k', false, r.error);
else {
  console.log(`    ${fmt(r.best)}`);
  check('16k -> detected', r.best.verdict === 'detected', r.best.verdict);
  check('16k -> no bytes derived', r.best.ascii == null, String(r.best.ascii));
  check('16k -> symbol word recovered',
    JSON.stringify(r.best.symbols) === JSON.stringify(metaFor('16k').message_symbols),
    JSON.stringify(r.best.symbols));
  check('16k -> structure above threshold',
    r.best.structure >= VARIANTS['16k'].thresholds.detectStructure,
    r.best.structure.toFixed(3));
}

/* ------------------------------------------------------------ auto probing -- */
console.log('\nauto mode on the 16 kHz sample (must fall through 44.1k to 16k):');
r = await analyse(path.join(SAMPLES, '16k_watermarked.wav'), ['44.1k', '16k']);
if (r.error) check('auto/16k', false, r.error);
else {
  for (const x of r.runs) console.log(`    ${fmt(x)}`);
  check('auto tried both variants', r.runs.length === 2, `${r.runs.length} run(s)`);
  check('auto picked 16k', r.best.variant === '16k', r.best.variant);
  check('auto verdict detected', r.best.verdict === 'detected', r.best.verdict);
  const v441 = r.runs.find((x) => x.variant === '44.1k');
  check('44.1k leg did not fire', v441 && v441.verdict !== 'detected', v441 && v441.verdict);
}

console.log('\nauto mode on the 44.1 kHz sample (should settle on 44.1k):');
r = await analyse(path.join(SAMPLES, '44k1_watermarked.wav'), ['44.1k', '16k']);
if (r.error) check('auto/44.1k', false, r.error);
else {
  for (const x of r.runs) console.log(`    ${fmt(x)}`);
  check('auto picked 44.1k', r.best.variant === '44.1k', r.best.variant);
  check('auto payload IRDTS', r.best.ascii === 'IRDTS', String(r.best.ascii));
}

/* ---------------------------------------------------------------- negative -- */
console.log('\nunwatermarked carrier through both variants:');
r = await analyse(path.join(SAMPLES, '16k_clean.wav'), ['44.1k', '16k']);
if (r.error) check('clean', false, r.error);
else {
  for (const x of r.runs) console.log(`    ${fmt(x)}`);
  check('no variant fires on clean audio',
    r.runs.every((x) => x.verdict !== 'detected'),
    r.runs.map((x) => `${x.variant}:${x.verdict}`).join(' '));
}

await browser.close();
await server.close();
console.log(`\n${failures.length ? `FAILED (${failures.length}): ${failures.join(', ')}` : 'all variant browser checks passed'}`);
process.exit(failures.length ? 1 : 0);
