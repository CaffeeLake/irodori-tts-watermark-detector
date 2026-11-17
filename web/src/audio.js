/**
 * Browser audio decoding.
 *
 * `decodeAudioData` always resamples to the AudioContext's rate, so we decode at
 * 48 kHz: that is Opus' native rate (the WebM clips produced by Irodori-TTS pipelines
 * need no resampling at all) and a harmless upsample for anything else.  The
 * 48 kHz -> 44.1 kHz step is then done by our own torchaudio-compatible sinc
 * resampler so the front-end stays bit-comparable with the Python reference.
 */

export const DECODE_RATE = 48000;

/**
 * @param {ArrayBuffer} arrayBuffer raw file bytes
 * @returns {Promise<{pcm: Float32Array, sampleRate: number, channels: number, durationSec: number}>}
 */
export async function decodeToMono(arrayBuffer) {
  const Ctx = self.OfflineAudioContext || self.webkitOfflineAudioContext;
  if (!Ctx) throw new Error('OfflineAudioContext is unavailable in this browser');

  // length must be >= 1; the context is only used as a decoder.
  const ctx = new Ctx(1, 1, DECODE_RATE);
  const buf = await ctx.decodeAudioData(arrayBuffer);

  const n = buf.length;
  const channels = buf.numberOfChannels;
  const pcm = new Float32Array(n);
  for (let c = 0; c < channels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < n; i++) pcm[i] += ch[i];
  }
  if (channels > 1) for (let i = 0; i < n; i++) pcm[i] /= channels;

  return { pcm, sampleRate: buf.sampleRate, channels, durationSec: n / buf.sampleRate };
}
