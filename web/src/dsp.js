/**
 * Signal-processing primitives needed to reproduce SilentCipher's analysis front-end
 * (as used by Irodori-TTS) bit-for-bit in the browser.
 *
 * Everything here is a deliberate port of the reference implementation:
 *   - resampleSinc    <- torchaudio.functional.resample (sinc_interp_hann)
 *   - stftMagnitude   <- silentcipher.stft.STFT.transform (torch.stft, center=True, reflect)
 */

/* ------------------------------------------------------------------ FFT ---- */

/**
 * In-place iterative radix-2 complex FFT. `n` must be a power of two.
 * Twiddle factors and the bit-reversal permutation are precomputed once.
 */
export class FFT {
  constructor(n) {
    if ((n & (n - 1)) !== 0) throw new Error(`FFT size must be a power of two, got ${n}`);
    this.n = n;
    this.rev = new Uint32Array(n);
    let bits = 0;
    while (1 << bits < n) bits++;
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / n);
    }
  }

  /** @param {Float64Array} re @param {Float64Array} im */
  transform(re, im) {
    const { n, rev, cos, sin } = this;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let k = 0, tw = 0; k < half; k++, tw += step) {
          const wr = cos[tw], wi = sin[tw];
          const a = i + k, b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr;        im[a] += xi;
        }
      }
    }
  }
}

/* ------------------------------------------------------------- resample ---- */

const gcd = (a, b) => (b ? gcd(b, a % b) : a);

/**
 * Build the polyphase sinc kernel bank exactly as torchaudio does.
 * Returns { kernels: Float64Array[newFreq][width*2+origFreq], width, origFreq, newFreq }.
 */
export function sincResampleKernel(origRate, newRate, lowpassFilterWidth = 6, rolloff = 0.99) {
  const g = gcd(origRate, newRate);
  const origFreq = origRate / g;
  const newFreq = newRate / g;
  const baseFreq = Math.min(origFreq, newFreq) * rolloff;
  const width = Math.ceil((lowpassFilterWidth * origFreq) / baseFreq);
  const klen = 2 * width + origFreq;
  const scale = baseFreq / origFreq;

  const kernels = [];
  for (let j = 0; j < newFreq; j++) {
    const k = new Float64Array(klen);
    for (let i = 0; i < klen; i++) {
      // idx = (-width + i) / origFreq ;  t = (-j / newFreq + idx) * baseFreq
      let t = (-j / newFreq + (-width + i) / origFreq) * baseFreq;
      t = Math.min(lowpassFilterWidth, Math.max(-lowpassFilterWidth, t));
      const window = Math.cos((t * Math.PI) / lowpassFilterWidth / 2) ** 2;
      t *= Math.PI;
      const sinc = t === 0 ? 1.0 : Math.sin(t) / t;
      k[i] = sinc * window * scale;
    }
    kernels.push(k);
  }
  return { kernels, width, origFreq, newFreq };
}

/**
 * Resample `x` from `origRate` to `newRate`.
 * Mirrors torchaudio.functional.resample (default sinc_interp_hann) including the
 * zero padding and the ceil() output-length rule.
 * @param {Float32Array|Float64Array} x
 * @returns {Float32Array}
 */
export function resampleSinc(x, origRate, newRate, lowpassFilterWidth = 6, rolloff = 0.99) {
  if (origRate === newRate) return x instanceof Float32Array ? x : Float32Array.from(x);
  const { kernels, width, origFreq, newFreq } = sincResampleKernel(
    origRate, newRate, lowpassFilterWidth, rolloff,
  );
  const length = x.length;
  const klen = 2 * width + origFreq;
  const padded = new Float64Array(width + length + width + origFreq);
  for (let i = 0; i < length; i++) padded[width + i] = x[i];

  const nStrides = Math.floor((padded.length - klen) / origFreq) + 1;
  const target = Math.ceil((newFreq * length) / origFreq);
  const out = new Float32Array(target);

  for (let s = 0, base = 0; s < nStrides; s++, base += origFreq) {
    for (let j = 0; j < newFreq; j++) {
      const o = s * newFreq + j;
      if (o >= target) break;
      const k = kernels[j];
      let acc = 0;
      for (let i = 0; i < klen; i++) acc += padded[base + i] * k[i];
      out[o] = acc;
    }
    if (s * newFreq >= target) break;
  }
  return out;
}

/* ------------------------------------------------------------------ STFT --- */

/** Periodic Hann window, identical to torch.hann_window(n). */
export function hannPeriodic(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

/**
 * Number of STFT frames silentcipher produces for `nSamples` input samples.
 * silentcipher first zero-pads the signal up to a multiple of `winLen`
 * (always adding a full window when it is already a multiple), then calls
 * torch.stft with center=True.
 */
export function frameCount(nSamples, nFft = 4096, hop = 2048) {
  const padded = nSamples + (nFft - (nSamples % nFft));
  return 1 + Math.floor(padded / hop);
}

/**
 * Magnitude spectrogram, laid out as `nBands` rows of `frames` columns
 * (row-major, i.e. mag[band * frames + frame]) which is exactly the memory
 * order the ONNX decoder expects for its (1,1,band,frames) input.
 *
 * @param {Float32Array} x  signal at 44.1 kHz
 * @param {{nFft?:number, hop?:number, bands?:number}} opts
 */
export function stftMagnitude(x, { nFft = 4096, hop = 2048, bands = 1024 } = {}) {
  const n = x.length;
  // 1) silentcipher's trailing zero pad to a whole number of windows
  const padLen = nFft - (n % nFft);
  const L = n + padLen;
  // 2) torch.stft(center=True) reflect-pads nFft/2 on both sides
  const half = nFft >> 1;
  const buf = new Float64Array(L + 2 * half);
  for (let i = 0; i < L; i++) buf[half + i] = i < n ? x[i] : 0;
  for (let i = 0; i < half; i++) {
    buf[half - 1 - i] = buf[half + 1 + i];           // reflect left
    buf[half + L + i] = buf[half + L - 2 - i];       // reflect right
  }

  const frames = 1 + Math.floor(L / hop);
  const win = hannPeriodic(nFft);
  const fft = new FFT(nFft);
  const re = new Float64Array(nFft);
  const im = new Float64Array(nFft);
  const mag = new Float32Array(bands * frames);

  for (let f = 0; f < frames; f++) {
    const off = f * hop;
    for (let i = 0; i < nFft; i++) { re[i] = buf[off + i] * win[i]; im[i] = 0; }
    fft.transform(re, im);
    for (let b = 0; b < bands; b++) {
      mag[b * frames + f] = Math.hypot(re[b], im[b]);
    }
  }
  return { mag, frames, bands };
}

/** Mean of x^2. */
export function meanPower(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return s / x.length;
}

/** Downmix an array of channel Float32Arrays to mono (plain average). */
export function toMono(channels) {
  if (channels.length === 1) return channels[0];
  const n = channels[0].length;
  const out = new Float32Array(n);
  for (const ch of channels) for (let i = 0; i < n; i++) out[i] += ch[i];
  for (let i = 0; i < n; i++) out[i] /= channels.length;
  return out;
}
