/**
 * The two published SilentCipher variants, and what each one can actually tell us.
 *
 * Irodori-TTS only ever uses the 44.1 kHz model (`irodori_tts/watermark.py` constructs
 * `SilentCipherWatermarker(device=...)` and never overrides `model_type="44.1k"`), so
 * that variant gets the full payload-matched test against b"IRDTS".
 *
 * The 16 kHz model is supported for completeness, but note an upstream limitation:
 * `silentcipher`'s byte encoder emits four 2-bit symbols per byte while this model has
 * `message_len - 1 = 15` symbol slots, and 4n = 15 has no solution.  Its one-hot index
 * is `symbol + 1`, which must stay below `message_dim = 4`, capping symbols at 2.
 * Consequently **no byte payload can be encoded or decoded with the 16 kHz model** —
 * `encode_wav()` raises AssertionError for every payload.  Detection for that variant
 * is therefore payload-independent: we test for period-16 structure and report the raw
 * symbol word, but there is no "IRDTS" to match.
 */

/** SilentCipher normalises every input to this reference power before analysis. */
export const AVERAGE_ENERGY_VCTK = 0.002837200844477648;

/** Conv receptive-field radius in frames (10 layers of 3x3) — identical for both. */
const HALO = 10;

/**
 * Periods used as the empirical null when measuring period-`period` structure.
 *
 * Only periods coprime to `period` are usable: if gcd(p, period) = g > 1 then folding
 * at p still groups frames into period/g distinct values and leaks real structure into
 * the null.  Multiples of `period` leak completely.  We take the nearest coprime
 * neighbours, which keeps the votes per fold position (and hence the chance
 * concentration) comparable; every fold is measured over the same frames.
 *
 * For 21 this yields [19, 20, 22, 23]; for 16 it yields [13, 15, 17, 19]
 * (14 and 18 are excluded, both sharing a factor of 2 with 16).
 */
export function nullPeriods(period, count = 4) {
  const gcd = (a, b) => (b ? gcd(b, a % b) : a);
  const out = [];
  for (let d = 1; out.length < count && d < period; d++) {
    for (const p of [period - d, period + d]) {
      if (p >= 4 && gcd(p, period) === 1 && out.length < count) out.push(p);
    }
  }
  return out.sort((a, b) => a - b);
}

export const IRODORI_PAYLOAD = [73, 82, 68, 84, 83]; // "IRDTS"

export const VARIANTS = Object.freeze({
  '44.1k': Object.freeze({
    id: '44.1k',
    label: '44.1 kHz (used by Irodori-TTS)',
    model: 'dec_m_0.44k1.onnx',
    sampleRate: 44100,
    nFft: 4096,
    hopLength: 2048,
    messageBandSize: 1024,
    messageDim: 5,
    messageLen: 21,
    halo: HALO,
    averageEnergyVctk: AVERAGE_ENERGY_VCTK,
    /** 20 slots x 2 bits = 40 bits = exactly 5 bytes, fully consumed by "IRDTS". */
    bytePayload: true,
    payload: IRODORI_PAYLOAD,
    /**
     * Calibrated on the exported eval-mode model: the four Irodori clips versus nine
     * negative controls (gaussian / pink noise, near-silence, sweep, harmonic tone,
     * time-reversed watermarked speech, and SilentCipher watermarks carrying different
     * payloads), worst case over every window position.
     *
     *   window | positives (min margin / max p) | negatives (max margin / min p)
     *    4 s   | 0.233 / 5.2e-06                | 0.276 / 1.4e-07   <- OVERLAPS
     *    6 s   | 0.291 / 4.7e-12                | 0.169 / 5.9e-05   <- clean
     *    8 s   | 0.322 / 1.8e-18                | 0.157 / 1.7e-05
     */
    thresholds: Object.freeze({
      minFrames: 129,        // ~6.0 s
      detectMargin: 0.23,
      detectP: 1e-8,
      rejectMargin: 0.20,
      rejectP: 1e-6,
      structurePresent: 0.10,
      payloadResolved: 0.65,
    }),
  }),

  '16k': Object.freeze({
    id: '16k',
    label: '16 kHz (payload-independent only)',
    model: 'dec_m_0.16k.onnx',
    sampleRate: 16000,
    nFft: 2048,
    hopLength: 1024,
    messageBandSize: 512,
    messageDim: 4,
    messageLen: 16,
    halo: HALO,
    averageEnergyVctk: AVERAGE_ENERGY_VCTK,
    /** See the module comment: byte payloads are impossible for this variant. */
    bytePayload: false,
    payload: null,
    /**
     * Calibrated on genuine 16 kHz watermarks versus audio without one.
     *
     * Positives: speech watermarked with the real 16 kHz model (several carriers and
     * messages), raw and after an Opus round trip.  Negatives: the same carriers
     * unwatermarked, 44.1 kHz-watermarked speech (the key confusable), noise, a tone
     * and a sweep.  Worst case over every window position:
     *
     *   window | positives (min structure) | negatives (max structure)
     *    4 s   | 0.326                     | 0.440   <- OVERLAPS
     *    6 s   | 0.217                     | 0.239   <- still overlaps
     *    8 s   | 0.349                     | 0.102   <- clean
     *   12 s   | 0.263                     | 0.067
     *
     * The weakest positives are Opus-coded synthetic carriers, which lose a lot of
     * watermark energy; detection stays well clear of the negatives from 8 s on, so
     * this variant needs two seconds more audio than the 44.1 kHz one.
     */
    thresholds: Object.freeze({
      minFrames: 125,        // ~8.0 s
      detectStructure: 0.20,
      rejectStructure: 0.15,
      structurePresent: 0.15,
    }),
  }),
});

export const DEFAULT_VARIANT = '44.1k';

/** Seconds of audio a variant needs before it will commit to a verdict. */
export function minSeconds(v) {
  return (v.thresholds.minFrames * v.hopLength) / v.sampleRate;
}

/**
 * Frame budget requested by a window setting, deliberately independent of the file
 * length so it can bound the DSP front end before any spectrogram exists.
 *
 *   'min'  -> exactly the variant's calibrated minimum verdict window.  The default:
 *             a fully valid operating point (the thresholds are calibrated there) that
 *             bounds the wait, since going further only strengthens a negative result.
 *   'all'  -> Infinity (the whole file).
 *   number -> that many seconds, but never less than the minimum, so a larger window
 *             can never leave the analysis unable to reach a verdict.
 *
 * Note that this cannot conjure audio: a file shorter than `minFrames` can never reach
 * a verdict for that variant, whatever the window is set to.  Use `canReachVerdict()`
 * to detect that case up front instead of spending inference on it.
 */
export function windowFrames(windowSpec, cfg) {
  const min = cfg.thresholds.minFrames;
  if (windowSpec === 'all') return Infinity;
  if (windowSpec === 'min' || windowSpec == null) return min;
  const seconds = Number(windowSpec);
  if (!Number.isFinite(seconds) || seconds <= 0) return min;
  return Math.max(min, Math.round((seconds * cfg.sampleRate) / cfg.hopLength));
}

/** Can `durationSec` of audio ever yield a verdict for this variant? */
export function canReachVerdict(cfg, durationSec) {
  return durationSec >= minSeconds(cfg);
}

export function getVariant(id) {
  const v = VARIANTS[id];
  if (!v) throw new Error(`unknown SilentCipher variant: ${id}`);
  return v;
}
