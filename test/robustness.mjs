/**
 * Robustness survey: how does the detector hold up when a watermarked file has been
 * transcoded to another sample rate / codec, or shifted in time?
 *
 * Runs the shipped JS pipeline (same modules the browser loads) over a matrix of
 * ffmpeg-generated variants of the repo's watermarked clips and prints a table.
 *
 *   node test/robustness.mjs                  # default matrix
 *   CLIPS=part0,part2 node test/robustness.mjs
 *   SECONDS=6 node test/robustness.mjs
 */
import { readdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { prepare, runDetector } from '../web/src/pipeline.js';
import { DEFAULTS, THRESHOLDS } from '../web/src/detector.js';
import { ROOT, decodePcm, getSession, Tensor } from './helpers.mjs';

const SECONDS = Number(process.env.SECONDS || 6);
const FRAMES = Math.round((SECONDS * DEFAULTS.sampleRate) / DEFAULTS.hopLength);
const work = mkdtempSync(path.join(tmpdir(), 'wm-robust-'));

const want = (process.env.CLIPS || 'part0,part2').split(',');
const clips = readdirSync(ROOT)
  .filter((f) => f.endsWith('.webm') && want.some((w) => f.includes(w)))
  .sort();

/** Transcode `src` with the given ffmpeg output args; returns the new path. */
function transcode(src, name, args) {
  const out = path.join(work, name);
  execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', src, ...args, out], { maxBuffer: 1 << 30 });
  return out;
}

/**
 * The matrix. Each entry produces one variant of a watermarked clip.
 * `-ar` forces a sample-rate conversion away from (and later back to) 44.1 kHz.
 */
const VARIANTS = [
  ['baseline webm/opus 48k', (s) => s],
  ['wav 44100 (native)', (s) => transcode(s, 'a44100.wav', ['-ar', '44100', '-c:a', 'pcm_s16le'])],
  ['wav 48000', (s) => transcode(s, 'a48000.wav', ['-ar', '48000', '-c:a', 'pcm_s16le'])],
  ['wav 32000', (s) => transcode(s, 'a32000.wav', ['-ar', '32000', '-c:a', 'pcm_s16le'])],
  ['wav 22050', (s) => transcode(s, 'a22050.wav', ['-ar', '22050', '-c:a', 'pcm_s16le'])],
  ['wav 16000', (s) => transcode(s, 'a16000.wav', ['-ar', '16000', '-c:a', 'pcm_s16le'])],
  ['wav 88200 (upsampled)', (s) => transcode(s, 'a88200.wav', ['-ar', '88200', '-c:a', 'pcm_s16le'])],
  ['wav 96000 (upsampled)', (s) => transcode(s, 'a96000.wav', ['-ar', '96000', '-c:a', 'pcm_s16le'])],
  ['mp3 320k @44.1k', (s) => transcode(s, 'm320.mp3', ['-ar', '44100', '-b:a', '320k'])],
  ['mp3 128k @44.1k', (s) => transcode(s, 'm128.mp3', ['-ar', '44100', '-b:a', '128k'])],
  ['mp3 64k @44.1k', (s) => transcode(s, 'm64.mp3', ['-ar', '44100', '-b:a', '64k'])],
  ['aac 128k @44.1k', (s) => transcode(s, 'a128.m4a', ['-ar', '44100', '-b:a', '128k', '-c:a', 'aac'])],
  ['ogg vorbis 128k', (s) => transcode(s, 'v128.ogg', ['-b:a', '128k', '-c:a', 'libvorbis'])],
  ['opus 64k @48k', (s) => transcode(s, 'o64.opus', ['-b:a', '64k', '-c:a', 'libopus'])],
  ['44.1k->16k->44.1k', (s) => {
    const mid = transcode(s, 'rt16.wav', ['-ar', '16000', '-c:a', 'pcm_s16le']);
    return transcode(mid, 'rt16back.wav', ['-ar', '44100', '-c:a', 'pcm_s16le']);
  }],
  ['mp3 128k -> wav 48k', (s) => {
    const mid = transcode(s, 'chain.mp3', ['-ar', '44100', '-b:a', '128k']);
    return transcode(mid, 'chain.wav', ['-ar', '48000', '-c:a', 'pcm_s16le']);
  }],
];

/** Sample offsets applied to the decoded 48 kHz PCM (hop is 2048 @ 44.1 kHz). */
const OFFSETS_48K = [0, 139, 557, 1114, 2229];

async function score(pcm, rate) {
  const spec = prepare(pcm, rate);
  if (spec.silent || spec.frames < FRAMES) return null;
  const session = await getSession();
  const { result } = await runDetector(spec, session, Tensor, {
    chunkFrames: 43, maxFrames: FRAMES, earlyStop: false,
  });
  return result;
}

const row = (label, r) => {
  if (!r) { console.log(`  ${label.padEnd(26)} (too short)`); return; }
  const ok = r.verdict === 'detected' ? 'DETECTED' : r.verdict;
  console.log(`  ${label.padEnd(26)} ${ok.padEnd(13)} margin=${r.margin.toFixed(3)} `
    + `match=${r.match.toFixed(3)} p=${r.pValue.toExponential(1).padStart(8)} `
    + `struct=${r.structure.toFixed(3)} shift=${String(r.shift).padStart(2)} blind=${r.blind.ascii ?? '-'}`);
};

console.log(`\nAnalysis window: ${SECONDS}s (${FRAMES} frames), threshold margin>=${THRESHOLDS.detectMargin}, p<=${THRESHOLDS.detectP}\n`);

try {
  for (const clip of clips) {
    const src = path.join(ROOT, clip);
    console.log(`=== ${clip} : format / sample-rate transcodes ===`);
    for (const [label, make] of VARIANTS) {
      let r = null;
      try {
        r = await score(decodePcm(make(src), 48000), 48000);
      } catch (e) {
        console.log(`  ${label.padEnd(26)} ERROR ${String(e.message).slice(0, 60)}`);
        continue;
      }
      row(label, r);
    }

    console.log(`\n=== ${clip} : time offsets (STFT grid misalignment) ===`);
    const pcm = decodePcm(src, 48000);
    for (const off of OFFSETS_48K) {
      const frames44 = (off * 44100) / 48000 / DEFAULTS.hopLength;
      const r = await score(pcm.subarray(off), 48000);
      row(`drop ${String(off).padStart(4)} smp (${frames44.toFixed(2)} fr)`, r);
    }
    console.log('');
  }
} finally {
  rmSync(work, { recursive: true, force: true });
}
