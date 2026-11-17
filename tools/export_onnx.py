#!/usr/bin/env python
"""Export SilentCipher message decoders to ONNX for the browser detector.

Irodori-TTS watermarks with SilentCipher (`sony/silentcipher`) using the 44.1 kHz
model and the fixed payload b"IRDTS" (see irodori_tts/watermark.py).  Detection only
needs the *message decoder* branch:

    audio -> resample(SR) -> power-normalise -> |STFT| -> dec_m_0 -> message_dim logits/frame

so we export just `MsgDecoder` (10 gated conv layers + a linear read-out over the first
`message_band_size` frequency bands).  Encoder / carrier-decoder are not needed.

Both published variants are exported:

    44.1k : SR 44100, N_FFT 4096, hop 2048, 1024 bands, message_dim 5, message_len 21
    16k   : SR 16000, N_FFT 2048, hop 1024,  512 bands, message_dim 4, message_len 16

IMPORTANT - BatchNorm mode:
    silentcipher never calls .eval() on its modules, so upstream the decoder's ten
    BatchNorm2d layers run with *batch* statistics.  We export the eval-mode
    (running-statistics) model on purpose: batch statistics depend on how much audio
    sits in the tensor, which would make results window-dependent and break the
    chunked / streaming / early-stopping design.  See README.md ("BatchNorm fidelity").

Output (per variant):
    web/models/dec_m_0.<variant>.onnx   float32, what the browser loads
    web/models/meta.json                hyper-parameters for the JS side
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IRODORI_WATERMARK_PAYLOAD = [73, 82, 68, 84, 83]  # "IRDTS"

# variant -> (subdirectory in the Hub repo, output file slug)
VARIANTS = {
    "44.1k": ("44_1_khz/73999_iteration", "44k1"),
    "16k": ("16_khz/97561_iteration", "16k"),
}


def find_checkpoint(variant: str) -> str:
    from huggingface_hub import snapshot_download

    sub, _ = VARIANTS[variant]
    folder = snapshot_download(repo_id="sony/silentcipher",
                               allow_patterns=[sub.split("/")[0] + "/*"])
    return os.path.join(folder, *sub.split("/"))


class Wrapped(torch.nn.Module):
    """(1,1,band,T) magnitude -> (1,message_dim,T) logits (drop the dummy axis)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, mag):
        return self.inner(mag)[:, 0]


def export_variant(variant: str, ckpt_dir: str, out_dir: str, opset: int, quantize: bool) -> dict:
    from silentcipher.model import MsgDecoder

    cfg = yaml.safe_load(open(os.path.join(ckpt_dir, "hparams.yaml")))
    msg_dim, band = cfg["message_dim"], cfg["message_band_size"]

    dec = MsgDecoder(message_dim=msg_dim, message_band_size=band)
    state = torch.load(os.path.join(ckpt_dir, "dec_m_0.ckpt"), map_location="cpu")
    dec.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
    dec.eval()

    model = Wrapped(dec).eval()
    slug = VARIANTS[variant][1]
    path = os.path.join(out_dir, f"dec_m_0.{slug}.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            torch.randn(1, 1, band, 8).abs(),
            path,
            input_names=["magnitude"],
            output_names=["logits"],
            dynamic_axes={"magnitude": {3: "frames"}, "logits": {2: "frames"}},
            opset_version=opset,
            do_constant_folding=True,
        )
    print(f"  wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    # ---- parity against PyTorch ------------------------------------------------
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for frames in (cfg["message_len"], cfg["message_len"] * 2 + 1):
        x = torch.randn(1, 1, band, frames).abs()
        with torch.no_grad():
            ref = model(x).numpy()
        got = sess.run(None, {"magnitude": x.numpy()})[0]
        err = float(np.abs(ref - got).max())
        agree = float((ref.argmax(1) == got.argmax(1)).mean())
        print(f"  parity frames={frames}: max|diff|={err:.3e} argmax agreement={agree:.4f}")
        assert err < 1e-3, "ONNX export diverges from PyTorch"
        assert agree == 1.0, "argmax disagreement with PyTorch"

    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        int8 = os.path.join(out_dir, f"dec_m_0.{slug}.int8.onnx")
        quantize_dynamic(path, int8, weight_type=QuantType.QInt8)
        print(f"  wrote {int8} ({os.path.getsize(int8) / 1e6:.1f} MB)")

    entry = {
        "variant": variant,
        "model": os.path.basename(path),
        "sample_rate": cfg["SR"],
        "n_fft": cfg["N_FFT"],
        "hop_length": cfg["HOP_LENGTH"],
        "message_band_size": band,
        "message_dim": msg_dim,
        "message_len": cfg["message_len"],
        "n_messages": cfg["n_messages"],
        "message_sdr": cfg["message_sdr"],
    }

    # Can a byte payload actually be represented by this variant?
    #
    # silentcipher's binary_encode() emits 4 two-bit symbols per byte, and
    # letters_encoding() asserts len(symbols) == message_len - 1.  It also builds the
    # one-hot with index = symbol + 1, so the index must stay below message_dim.
    slots = cfg["message_len"] - 1
    byte_payload_ok = slots % 4 == 0 and msg_dim >= 5
    entry["symbol_slots"] = slots
    entry["byte_payload_supported"] = byte_payload_ok
    if byte_payload_ok:
        entry["payload"] = IRODORI_WATERMARK_PAYLOAD
        entry["payload_ascii"] = "".join(chr(c) for c in IRODORI_WATERMARK_PAYLOAD)
        entry["payload_bits"] = slots * 2
    else:
        entry["payload"] = None
        entry["payload_ascii"] = None
        entry["note"] = (
            f"message_len-1 = {slots} symbol slots and message_dim = {msg_dim}; "
            "silentcipher's byte encoder emits 4 symbols per byte with one-hot index "
            "symbol+1, so no byte payload can be encoded or decoded for this variant. "
            "Detection is payload-independent (period-%d structure) only." % cfg["message_len"]
        )
        print(f"  note: {entry['note']}")
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint dir, only valid together with a single --variant")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "web", "models"))
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--quantize",
        action="store_true",
        help="also emit int8 models. NOT recommended: dynamic quantisation of these "
             "Conv layers measured ~4x SLOWER than fp32 under onnxruntime-web's WASM "
             "kernels (2.8x -> 11.7x realtime cost), so the shipped build uses fp32.",
    )
    args = ap.parse_args()

    wanted = list(VARIANTS) if args.variant == "all" else [args.variant]
    if args.ckpt and len(wanted) != 1:
        ap.error("--ckpt requires a single --variant")

    os.makedirs(args.out_dir, exist_ok=True)
    meta = {
        "comment": "SilentCipher message decoders for Irodori-TTS watermark detection",
        "average_energy_vctk": 0.002837200844477648,
        "irodori_variant": "44.1k",
        "variants": {},
    }
    for variant in wanted:
        ckpt = args.ckpt or find_checkpoint(variant)
        print(f"\n[{variant}] checkpoint: {ckpt}")
        meta["variants"][variant] = export_variant(
            variant, ckpt, args.out_dir, args.opset, args.quantize,
        )

    meta_path = os.path.join(args.out_dir, "meta.json")
    if args.variant != "all" and os.path.exists(meta_path):
        # keep whatever was exported previously for the other variant
        old = json.load(open(meta_path))
        old.setdefault("variants", {}).update(meta["variants"])
        meta = {**old, **{k: v for k, v in meta.items() if k != "variants"},
                "variants": old["variants"]}
    json.dump(meta, open(meta_path, "w"), indent=2)
    print(f"\nwrote {meta_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
