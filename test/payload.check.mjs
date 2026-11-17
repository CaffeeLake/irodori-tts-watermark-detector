/**
 * Browser check for the custom-payload control: live validation, that the entered
 * payload is what actually gets searched, and that the card says which one it used.
 *
 *   node test/payload.check.mjs
 */
import { existsSync, readFileSync, copyFileSync, rmSync } from 'node:fs';
import path from 'node:path';
import process from 'node:process';

import { startServer, WEB_ROOT } from '../tools/serve.mjs';

const ROOT = path.resolve(import.meta.dirname, '..');
const SAMPLES = path.join(ROOT, 'test', 'samples');
const HELLO = path.join(SAMPLES, '44k1_hello_watermarked.wav');
if (!existsSync(HELLO)) {
  console.error('missing test/samples/44k1_hello_watermarked.wav — see README (npm run samples)');
  process.exit(1);
}

const { firefox, chromium, webkit } = await import('playwright');
const browserType = { firefox, chromium, webkit }[process.env.BROWSER || 'firefox'];

const TMP = path.join(WEB_ROOT, '__tmp_payload.wav');
copyFileSync(HELLO, TMP);

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

/* ------------------------------------------------------- live validation --- */
console.log('\nvalidation feedback:');
const typed = async (text) => page.evaluate((t) => {
  const el = document.getElementById('payload');
  el.value = t;
  el.dispatchEvent(new Event('input'));
  return {
    help: document.getElementById('payload-help').textContent,
    bad: document.getElementById('payload-help').classList.contains('bad'),
    invalid: el.classList.contains('invalid'),
  };
}, text);

let v = await typed('');
check('blank explains the default', /Blank = the Irodori default "IRDTS"/.test(v.help), v.help);
check('blank is not an error', v.bad === false && v.invalid === false);

v = await typed('HELLO');
check('valid text is accepted', v.bad === false && v.invalid === false, v.help);
check('valid text echoes the bytes', /\[72, 69, 76, 76, 79\] = "HELLO"/.test(v.help), v.help);

v = await typed('72,69,76,76,79');
check('decimal form accepted', v.bad === false && /"HELLO"/.test(v.help), v.help);

v = await typed('ABC');
check('wrong length rejected', v.bad === true && v.invalid === true, v.help);
check('error states the requirement', /exactly 5 bytes/.test(v.help), v.help);

v = await typed('300,1,2,3,4');
check('out-of-range byte rejected', v.bad === true, v.help);

/* ------------------------------------------------------------- end to end --- */
async function run(payloadText) {
  return page.evaluate(async ({ payloadText }) => {
    const el = document.getElementById('payload');
    el.value = payloadText;
    el.dispatchEvent(new Event('input'));
    const bytes = await (await fetch('./__tmp_payload.wav')).arrayBuffer();
    const f = new File([bytes], 'hello.wav');
    document.querySelectorAll('#results .card').forEach((c) => c.remove());
    window.__detector.enqueue([f]);
    await new Promise((res) => {
      const t = setInterval(() => {
        if (document.querySelectorAll('#results .card[data-state]').length === 0
          && document.querySelector('#results .card')) { clearInterval(t); res(); }
      }, 200);
    });
    const card = document.querySelector('#results .card');
    const row = (k) => [...card.querySelectorAll('.row')]
      .find((r) => r.querySelector('.k').textContent === k)?.querySelector('.v').textContent ?? '';
    return {
      badge: card.querySelector('.badge').textContent,
      searching: row('searching for'),
      recovered: row('recovered payload'),
      agreement: row('symbol agreement'),
      notes: [...card.querySelectorAll('.note')].map((n) => n.textContent).join(' '),
    };
  }, { payloadText });
}

console.log('\nsample watermarked with "HELLO", searching for HELLO:');
let r = await run('HELLO');
console.log(`    badge=${r.badge} | ${r.searching} | ${r.recovered}`);
check('detected', r.badge === 'detected', r.badge);
check('card says it searched for HELLO (custom)', /"HELLO" \(custom\)/.test(r.searching), r.searching);
check('recovered payload is HELLO', /"HELLO"/.test(r.recovered), r.recovered);
check('agreement is against HELLO', /vs "HELLO"/.test(r.agreement), r.agreement);

console.log('\nsame sample, default payload (IRDTS):');
r = await run('');
console.log(`    badge=${r.badge} | ${r.searching} | ${r.recovered}`);
check('not reported as an IRDTS match', r.badge !== 'detected', r.badge);
check('card says it searched for IRDTS (default)', /"IRDTS" \(default\)/.test(r.searching), r.searching);
check('still reports the real payload', /"HELLO"/.test(r.recovered), r.recovered);
check('note explains a different payload is present',
  /different payload/.test(r.notes), r.notes.slice(0, 90));

rmSync(TMP, { force: true });
await browser.close();
await server.close();
console.log(`\n${failures.length ? `FAILED (${failures.length}): ${failures.join(', ')}` : 'all payload checks passed'}`);
process.exit(failures.length ? 1 : 0);
