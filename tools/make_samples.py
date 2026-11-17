#!/usr/bin/env python
"""Synthesise genuine SilentCipher-watermarked test audio for both variants.

Run once per variant -- silentcipher's STFT class uses a Singleton metaclass, so
building both models in one process would silently give the second one the first
one's FFT size:

    python tools/make_samples.py --variant 44.1k
    python tools/make_samples.py --variant 16k

The carrier is synthesised (see synth_speech) rather than taken from real audio, so the
samples are self-contained and provably carry only the watermark we put there.  Noise
cannot be used: SilentCipher was trained on speech and neither model recovers its own
watermark from pink or gaussian noise, so the carrier needs formant structure and
syllabic dynamics.

For 44.1 kHz the real `encode_wav()` is used with Irodori's payload b"IRDTS".

`silentcipher.Model.encode_wav()` cannot be used for the 16 kHz model: its byte encoder
emits four 2-bit symbols per byte, while that model has `message_len - 1 = 15` symbol
slots (4n = 15 has no solution), so it raises AssertionError for every payload.  We
therefore replicate the body of `encode_wav()` and feed a valid raw symbol word
(15 slots, values 0..message_dim-2 so the one-hot index `symbol+1` stays in range).

This helper is verified against the real `encode_wav()` on the 44.1 kHz model, where
both paths are available: the two agree bit-for-bit (see --self-test).

Outputs per variant (mono, at the variant's own rate):
    test/samples/<slug>_watermarked.wav     raw
    test/samples/<slug>_watermarked.webm    after an Opus round trip
    test/samples/<slug>_clean.wav           the same carrier, unwatermarked
    test/samples/<slug>_samples.json        what was embedded
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torchaudio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "test", "samples")

# 15 slots, values 0..2 -- the only layout the 16 kHz model can represent.
MESSAGE = [0, 1, 2, 2, 1, 0, 1, 2, 0, 0, 2, 1, 1, 2, 0]


def encode_symbols(model, y, orig_sr, symbols, message_sdr=None):
    """Mirror silentcipher.Model.encode_wav() but with a raw symbol message."""
    cfg = model.config
    assert len(symbols) == cfg.message_len - 1, f"need {cfg.message_len - 1} symbols"
    assert all(0 <= s <= cfg.message_dim - 2 for s in symbols), "symbol out of range"
    if message_sdr is None:
        message_sdr = cfg.message_sdr

    y = torch.as_tensor(np.ascontiguousarray(y), dtype=torch.float32)
    with torch.no_grad():
        orig = y.clone()
        if orig_sr != model.sr:
            y = torchaudio.functional.resample(y.view(1, -1), orig_sr, model.sr).squeeze()
        power = torch.mean(y ** 2)
        y = (y * torch.sqrt(torch.tensor(model.average_energy_VCTK) / power)).unsqueeze(0).unsqueeze(0)

        carrier, phase = model.stft.transform(y.squeeze(1))
        carrier, phase = carrier[:, None], phase[:, None]

        msgs, _ = model.letters_encoding(carrier.shape[3], [list(symbols)])
        msg_enc = model.enc_c.transform_message(
            torch.tensor(msgs, dtype=torch.float32).unsqueeze(0))
        merged = torch.cat((model.enc_c(carrier),
                            carrier.repeat(1, 32, 1, 1),
                            msg_enc.repeat(1, 32, 1, 1)), dim=1)
        info = model.dec_c(merged, message_sdr)
        if cfg.utterance_level_normalization:
            info = info * (torch.mean(carrier ** 2, dim=(2, 3), keepdim=True) ** 0.5)
        elif cfg.frame_level_normalization:
            info = info * (torch.mean(carrier ** 2, dim=2, keepdim=True) ** 0.5)
        if cfg.ensure_negative_message:
            info = -info
            recon = torch.nn.functional.relu(info + carrier)
        else:
            recon = torch.abs(info + carrier)

        model.stft.num_samples = y.shape[2]
        out = model.stft.inverse(recon.squeeze(1), phase.squeeze(1))[0, 0]
        out = out * torch.sqrt(power / torch.tensor(model.average_energy_VCTK))
        if orig_sr != model.sr:
            out = torchaudio.functional.resample(out.view(1, -1), model.sr, orig_sr).squeeze()
            out = out[:orig.shape[0]]
        k = min(len(orig), len(out))
        rms1 = float((orig[:k] ** 2).mean() ** 0.5)
        rms2 = float(((orig[:k] - out[:k]) ** 2).mean() ** 0.5)
    return out.numpy(), 20 * np.log10(rms1 / max(rms2, 1e-20))


def self_test() -> None:
    """encode_symbols() must reproduce encode_wav() exactly on the 44.1 kHz model."""
    import silentcipher

    m = silentcipher.get_model(model_type="44.1k", device="cpu")
    rng = np.random.default_rng(0)
    y = (0.1 * rng.standard_normal(44100 * 4)).astype(np.float32)
    payload = [73, 82, 68, 84, 83]
    bits = "".join(f"{b:08b}" for b in payload)
    symbols = [int(bits[i * 2:i * 2 + 2], 2) for i in range(len(bits) // 2)]

    ref, _ = m.encode_wav(torch.tensor(y), 44100, list(payload), calc_sdr=False)
    ref = ref.numpy()
    mine, sdr = encode_symbols(m, y, 44100, symbols)
    rel = float(np.sqrt(((ref - mine) ** 2).mean()) / np.sqrt((ref ** 2).mean()))
    print(f"self-test: encode_symbols vs encode_wav relative difference {rel:.3e} "
          f"(SDR {sdr:.2f} dB)")
    assert rel < 1e-6, "encode_symbols does not reproduce encode_wav"
    print("self-test OK")


def synth_speech(seconds, sr, seed=0):
    """A watermark-free, speech-like carrier: pulsed excitation + drifting formants
    + fricative bursts + a syllabic envelope.  SilentCipher embeds into this reliably,
    unlike noise."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr

    f0 = 115 + 25 * np.sin(2 * np.pi * 0.6 * t) + 6 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    phase = np.cumsum(f0) / sr
    pulses = np.zeros(n)
    pulses[(np.floor(phase) != np.floor(np.roll(phase, 1))).nonzero()[0]] = 1.0
    exc = pulses - 0.35 * np.roll(pulses, 2)
    frica = (np.sin(2 * np.pi * 2.7 * t + 1.1) > 0.75).astype(float)
    exc = exc + 0.25 * rng.standard_normal(n) * frica

    def resonator(x, fc, bw):
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * fc / sr
        return lfilter([1 - r], [1.0, -2 * r * np.cos(theta), r * r], x)

    out = np.zeros(n)
    seg = sr // 50
    for base, span, bw, gain in ((520, 180, 70, 1.0), (1500, 450, 110, 0.6), (2600, 500, 160, 0.35)):
        fc = base + span * np.sin(2 * np.pi * 1.3 * t + base / 700.0)
        y = np.zeros(n)
        for s in range(0, n, seg):
            e = min(n, s + seg)
            y[s:e] = resonator(exc[s:e], float(fc[s:e].mean()), bw)
        out += gain * y

    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.4 * t)
    env *= (np.sin(2 * np.pi * 0.35 * t + 0.4) > -0.55)
    out *= env
    return (out / (np.abs(out).max() + 1e-9) * 0.25).astype(np.float32)


def opus_roundtrip(y, rate, bitrate="32k"):
    with tempfile.TemporaryDirectory() as d:
        raw, webm, out = f"{d}/a.f32", f"{d}/a.webm", f"{d}/b.f32"
        np.asarray(y, dtype=np.float32).tofile(raw)
        subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(rate), "-ac", "1",
                        "-i", raw, "-c:a", "libopus", "-b:a", bitrate, webm], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-i", webm, "-f", "f32le", "-ac", "1",
                        "-ar", str(rate), out], check=True)
        return np.fromfile(out, dtype=np.float32), open(webm, "rb").read()


def write_wav(path, y, rate):
    """32-bit float WAV, written by hand: torchaudio.save() silently ignores the
    `encoding`/`bits_per_sample` arguments under the TorchCodec backend and would
    quantise these reference samples to 16 bits."""
    import struct

    x = np.asarray(y, dtype="<f4")
    data = x.tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, rate, rate * 4, 4, 32))
        f.write(b"data" + struct.pack("<I", len(data)) + data)


VARIANT_SLUG = {"44.1k": "44k1", "16k": "16k"}
IRODORI_PAYLOAD = [73, 82, 68, 84, 83]  # "IRDTS"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["44.1k", "16k"], default="16k")
    ap.add_argument("--self-test", action="store_true",
                    help="only verify the encoder against upstream encode_wav()")
    ap.add_argument("--carrier", default=None,
                    help="source audio; default is a synthesised speech-like carrier")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--payload", default=None,
                    help="44.1k only: 5 bytes as text (HELLO) or decimal (72,69,76,76,79)")
    ap.add_argument("--slug-suffix", default="", help="suffix for the output file names")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return 0

    import silentcipher

    # NOTE: silentcipher's STFT uses a Singleton metaclass, so only ever build ONE
    # model per process or the second one silently inherits the first one's FFT size.
    m = silentcipher.get_model(model_type=args.variant, device="cpu")
    sr = m.config.SR
    slug = VARIANT_SLUG[args.variant] + args.slug_suffix

    if args.carrier:
        y, ysr = m.load_audio(args.carrier)
        if y.ndim > 1:
            y = y.mean(1)
        clean = torchaudio.functional.resample(
            torch.tensor(y).view(1, -1), ysr, sr).squeeze().numpy()
        n = int(args.seconds * sr)
        clean = clean[:n] if len(clean) >= n else np.pad(clean, (0, n - len(clean)))
        carrier_name = os.path.basename(args.carrier)
    else:
        clean = synth_speech(args.seconds, sr)
        carrier_name = "synth_speech (never watermarked)"
    print(f"variant {args.variant} @ {sr} Hz, carrier: {carrier_name}")

    meta = {
        "variant": args.variant,
        "sample_rate": sr,
        "seconds": len(clean) / sr,
        "carrier": carrier_name,
        "message_len": m.config.message_len,
        "message_dim": m.config.message_dim,
    }

    if args.variant == "44.1k":
        payload = IRODORI_PAYLOAD
        if args.payload:
            t = args.payload
            payload = ([int(x, 0) for x in t.replace(",", " ").split()]
                       if ("," in t or " " in t) else [ord(c) for c in t])
            if len(payload) != 5 or any(not 0 <= b <= 255 for b in payload):
                print(f"payload must be 5 bytes, got {payload}", file=sys.stderr)
                return 1
        # The real encoder with the requested payload.
        wm, _ = m.encode_wav(torch.tensor(clean), sr, list(payload), calc_sdr=False)
        wm = wm.numpy()
        k = min(len(clean), len(wm))
        sdr = 20 * np.log10(np.sqrt((clean[:k] ** 2).mean())
                            / np.sqrt(((clean[:k] - wm[:k]) ** 2).mean()))
        meta["payload"] = payload
        meta["payload_ascii"] = "".join(chr(c) for c in payload)
        meta["note"] = ("Genuine Irodori payload embedded with the upstream encode_wav(); "
                        f"{slug}_clean.wav is the same carrier with no watermark.")
    else:
        wm, sdr = encode_symbols(m, clean, sr, MESSAGE)
        meta["message_symbols"] = MESSAGE
        meta["note"] = ("The 16 kHz model cannot carry a byte payload (15 symbol slots vs. "
                        "4 symbols per byte), so this is a raw symbol word with values 0..2. "
                        f"{slug}_clean.wav is the same carrier with no watermark.")
    meta["sdr_db"] = round(float(sdr), 2)
    print(f"  watermark embedded: SDR {sdr:.1f} dB, {len(wm) / sr:.1f} s")

    os.makedirs(OUT, exist_ok=True)
    write_wav(os.path.join(OUT, f"{slug}_watermarked.wav"), wm, sr)
    write_wav(os.path.join(OUT, f"{slug}_clean.wav"), clean, sr)
    _, webm = opus_roundtrip(wm, sr)
    with open(os.path.join(OUT, f"{slug}_watermarked.webm"), "wb") as f:
        f.write(webm)
    json.dump(meta, open(os.path.join(OUT, f"{slug}_samples.json"), "w"), indent=2)

    for f in sorted(x for x in os.listdir(OUT) if x.startswith(slug)):
        print(f"  {f}: {os.path.getsize(os.path.join(OUT, f))} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
