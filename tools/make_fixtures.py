#!/usr/bin/env python
"""Generate reference fixtures from the PyTorch/torchaudio implementation so the
JavaScript front-end (resampler / STFT / detector) can be checked for parity.

Run inside an environment that has torch + torchaudio (e.g. the Irodori-TTS venv):

    python tools/make_fixtures.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torchaudio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "test", "fixtures")
SR, N_FFT, HOP, BAND = 44100, 4096, 2048, 1024
AVG_E = 0.002837200844477648


def load_decoder():
    """The message decoder in **eval** mode (running BatchNorm statistics).

    NOTE: silentcipher never calls .eval() on its modules, so the upstream decoder
    actually runs BatchNorm with *batch* statistics.  We deliberately export and test
    the eval-mode model because it is frame-local and therefore streamable/chunkable;
    see README.md ("BatchNorm fidelity").  Beware that running a module in train mode
    mutates its running statistics, so never do that before taking eval references.
    """
    import yaml
    from huggingface_hub import snapshot_download
    from silentcipher.model import MsgDecoder

    folder = snapshot_download(repo_id="sony/silentcipher", allow_patterns=["44_1_khz/*"])
    ckpt = os.path.join(folder, "44_1_khz", "73999_iteration")
    cfg = yaml.safe_load(open(os.path.join(ckpt, "hparams.yaml")))
    dec = MsgDecoder(message_dim=cfg["message_dim"], message_band_size=cfg["message_band_size"])
    state = torch.load(os.path.join(ckpt, "dec_m_0.ckpt"), map_location="cpu")
    dec.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    dec.eval()
    return dec


def stft_magnitude(y: torch.Tensor) -> torch.Tensor:
    """silentcipher.stft.STFT.transform, magnitude branch only."""
    x = y.unsqueeze(0)
    x = torch.nn.functional.pad(x, (0, N_FFT - x.shape[1] % N_FFT))
    fft = torch.stft(x, N_FFT, HOP, N_FFT, window=torch.hann_window(N_FFT), return_complex=True)
    sq = fft.real ** 2 + fft.imag ** 2
    eps = torch.ones_like(sq) * (sq == 0).float() * 1e-24
    return (torch.sqrt(sq + eps) - torch.sqrt(eps))[0]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    manifest = {"sample_rate": SR, "n_fft": N_FFT, "hop_length": HOP, "bands": BAND, "cases": []}

    rng = np.random.default_rng(1234)
    n = 48000 * 3
    t = np.arange(n) / 48000.0
    sig = (0.3 * np.sin(2 * np.pi * 220 * t)
           + 0.2 * np.sin(2 * np.pi * 3000 * t + 0.7)
           + 0.05 * rng.standard_normal(n)).astype(np.float32)

    dec = load_decoder()

    for name, x48 in [("synth48k", sig)]:
        y = torch.tensor(x48)
        res = torchaudio.functional.resample(y.view(1, -1), 48000, SR).squeeze()
        gain = torch.sqrt(torch.tensor(AVG_E) / torch.mean(res ** 2))
        mag = stft_magnitude(res * gain)[:BAND]  # (BAND, frames)

        with torch.no_grad():
            logits = dec(mag[None, None])[0, 0]  # (message_dim, frames)

        (res.numpy().astype(np.float32)).tofile(os.path.join(OUT, f"{name}.resampled.f32"))
        mag.numpy().astype(np.float32).tofile(os.path.join(OUT, f"{name}.mag.f32"))
        np.asarray(x48, dtype=np.float32).tofile(os.path.join(OUT, f"{name}.input48k.f32"))
        logits.numpy().astype(np.float32).tofile(os.path.join(OUT, f"{name}.logits.f32"))
        manifest["cases"].append({
            "name": name,
            "input_rate": 48000,
            "input_samples": int(n),
            "resampled_samples": int(res.numel()),
            "frames": int(mag.shape[1]),
            "message_dim": int(logits.shape[0]),
        })
        print(f"{name}: {n} @48k -> {res.numel()} @44.1k, {mag.shape[1]} frames, "
              f"logits {tuple(logits.shape)}")

    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print("wrote", os.path.join(OUT, "manifest.json"))


if __name__ == "__main__":
    main()
