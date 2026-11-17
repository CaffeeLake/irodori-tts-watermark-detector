# Irodori-TTS Watermark Detector

A browser-only detector for the audio watermark that [Irodori-TTS](../Irodori-TTS)
embeds into generated speech. Model inference runs in **WebAssembly**
(onnxruntime-web, SIMD + threads); nothing is uploaded, no server-side component exists.

Irodori-TTS watermarks with [SilentCipher](https://github.com/sony/silentcipher)
(`sony/silentcipher`, 44.1 kHz model) carrying the fixed 5-byte payload `IRDTS`:

```python
# Irodori-TTS/irodori_tts/watermark.py
IRODORI_WATERMARK_PAYLOAD = (73, 82, 68, 84, 83)  # "IRDTS"
```

Both published SilentCipher variants are supported:

| variant | used by Irodori-TTS | payload test | verdict needs | what a detection means |
|---|---|---|---|---|
| **44.1 kHz** | yes | payload-matched against `IRDTS` | ≥ 6 s | watermark consistent with Irodori's payload |
| **16 kHz** | no | **impossible** (see below) | ≥ 8 s | *a* SilentCipher watermark is present |

### The 16 kHz variant cannot carry a byte payload

This is an upstream limitation, verified empirically rather than assumed.
`silentcipher`'s byte encoder emits four 2-bit symbols per byte, but the 16 kHz model
has `message_len - 1 = 15` symbol slots, and `4n = 15` has no integer solution. Its
one-hot index is `symbol + 1`, which must stay below `message_dim = 4`, capping symbols
at 2 (the byte path needs 0..3). `encode_wav()` therefore raises `AssertionError` for
*every* payload, including 1-, 2-, 3-, 4- and 5-byte ones.

So for 16 kHz audio there is no `IRDTS` to match. Detection is **payload-independent**:
we test for period-16 structure and report the raw 15-symbol word that was recovered.
A 16 kHz detection confirms a SilentCipher watermark, **not** Irodori-TTS provenance.

## Results

### The four Irodori-TTS clips originally in this repository

All four were watermarked by Irodori-TTS. Verdicts from the real browser (headless
Firefox, WASM EP, 4 threads), stopping as soon as the evidence became conclusive:

Analysed in full, every clip returned the payload byte-exactly —
`[73, 82, 68, 84, 83]` = `"IRDTS"` — with p-values from 1.9e-39 to 1.5e-115, agreeing
with the upstream `silentcipher` reference decoder.

> **Note:** those four `.webm` files were deleted from the working tree by something
> outside this project (no commit ever contained them, so they are not recoverable from
> git). The table above is retained as the record of that run. The test suite now runs
> entirely off the generated samples below, so it no longer depends on them.

### Generated samples (reproducible)

`pnpm run samples` embeds real watermarks into a synthesised, never-watermarked carrier,
so each sample provably carries only the watermark we put there:

| sample | detector | verdict | statistic | read back |
|---|---|---|---|---|
| `44k1_watermarked.wav` | 44.1 kHz | **detected** | margin 0.464, p 2.2e-62 | `IRDTS` |
| `44k1_watermarked.webm` (Opus) | 44.1 kHz | **detected** | margin 0.351, p 6.2e-33 | `IRDTS` |
| `16k_watermarked.wav` | 16 kHz | **detected** | structure 0.696 | `[0,1,2,…]` symbol word |
| `16k_watermarked.webm` (Opus) | 16 kHz | **detected** | structure 0.263 | symbol word |
| `44k1_clean.wav` / `16k_clean.wav` | both | not detected | — | — |
| 16 kHz watermark | **44.1 kHz** | not detected | — | — |
| 44.1 kHz watermark | **16 kHz** | not detected | — | — |

The last two rows matter: the variants do not cross-trigger.

Negative controls (white noise, pure tone, chirp, time-reversed watermarked speech,
and SilentCipher watermarks carrying *different* payloads) are all correctly rejected.

## Usage

```bash
pnpm install          # or npm install
pnpm run vendor        # copy the onnxruntime-web WASM runtime into web/vendor/
pnpm run serve         # http://127.0.0.1:8080
```

Then drop audio or video files onto the page. Any format WebAudio can decode works
(webm/opus, ogg, mp3, m4a, wav, flac, mp4 …).

Files are processed one at a time. Dropping more while an analysis is running adds
them to a queue, and every file gets a card **immediately** showing its position
(`next up…`, `waiting — 2 ahead in the queue`). The list is ordered *running first,
then queued in run order, then finished results newest-first*.

Each entry has a `×` button to delete it. Queued entries can be removed before they
start (which also drops them from the queue); the running one cannot, because the
worker is still writing into it, so its button stays disabled and **Clear all** skips
it and shows the count it would actually remove (e.g. `Clear all (2)`).

While a file is running the card shows the stage (`decoding` → `preparing` →
`analysing`), elapsed time, the frame budget and an expected duration learned from
previous runs. Inference takes tens of seconds (see Limits), so the progress bar is
deliberately indeterminate during a forward pass rather than inventing a percentage —
a single pass is atomic and its internal progress is genuinely unknown.

`tools/serve.mjs` sets the `Cross-Origin-Opener-Policy` / `Cross-Origin-Embedder-Policy`
headers that `SharedArrayBuffer` requires. **Without them onnxruntime-web silently
falls back to a single thread** and is ~4x slower, so keep them if you deploy this.

The **variant** selector picks which model to probe. It defaults to 44.1 kHz — the only
one Irodori-TTS uses — because `auto` also runs the 16 kHz pass whenever the first finds
nothing, roughly doubling the wait for a negative result.

The **payload** box selects which word to look for. Leave it blank for Irodori's
`IRDTS`; otherwise enter any 5 bytes as text (`HELLO`), decimal (`72,69,76,76,79`) or
hex (`0x48 0x45 …`). It is validated as you type against the selected variant's capacity
and the result card states which payload it actually searched for. In `auto` the box
applies to the 44.1 kHz pass; the 16 kHz model has no byte payload at all, so it takes a
15-value symbol word (`0,1,2,…`, each 0-2) or stays payload-independent when blank.

The thresholds carry over to any payload because the statistic scores one rotation of
the expected word against the other rotations — its null distribution depends on
`messageLen` and `messageDim`, not on which word is sought. Verified end to end against
a sample watermarked with `HELLO` (`pnpm run test:payload`).

The **window** selector bounds the work. It defaults to *minimum needed*, i.e. exactly
the variant's calibrated verdict window (6 s for 44.1 kHz, 8 s for 16 kHz). That is a
fully valid operating point — the thresholds are calibrated there — and it bounds the
wait. Longer windows only strengthen a *negative* conclusion, and cost ~0.2 s of WASM
inference per extra frame (~4.3 s per extra second of audio at 44.1 kHz), so the
default no longer runs a 12 s window on files that never trigger an early stop.
Any explicit value is clamped up to the minimum so the analysis can always reach a
verdict.

## Deploying

```bash
./build.sh              # -> dist/   (or: npm run build)
pnpm run preview         # serve dist/ locally with the production headers
pnpm run test:dist       # verify the bundle the way both hosts serve it
```

`dist/` is ~33 MB, almost all of it the 14 MB onnxruntime-web WASM runtime and two
9 MB models. `./build.sh --no-16k` drops the 16 kHz model if you only care about
Irodori output.

| | Cloudflare Pages | GitHub Pages |
|---|---|---|
| deploy | `wrangler pages deploy dist` | publish `dist/` (see `.github/workflows/pages.yml`) |
| COOP/COEP | `_headers`, applied on the first load | **impossible** — supplied by a service worker |
| first visit | isolated immediately | reloads once, then isolated |

**The headers are the whole ballgame.** onnxruntime-web's thread pool needs
`SharedArrayBuffer`, which needs a cross-origin isolated page; without it inference runs
single-threaded at 555 vs 263 ms per frame (2.1x slower). `build.sh` therefore emits:

- `_headers` — Cloudflare Pages reads this and sets COOP/COEP directly.
- `.nojekyll` — GitHub Pages runs Jekyll by default, which silently discards files
  whose names begin with an underscore, `_headers` included.
- `coi-sw.js` — GitHub Pages cannot set response headers at all, so this service worker
  re-issues each request and adds them itself. `src/coi.js` registers it and reloads
  once; it is a no-op wherever the server already sets the headers. It caches nothing,
  so it cannot serve stale content.

All asset URLs are relative, so a GitHub Pages *project* site (`/<repo>/`) works without
configuration. `pnpm run test:dist` asserts all three cases — headers honoured, no
headers at all, and a subpath mount — and that the built bundle still detects `IRDTS` at
multi-threaded speed.

After deploying, confirm the page does **not** show the "Running single-threaded"
banner.

### Regenerating the models

```bash
python tools/export_onnx.py                  # both variants, ~9.5 MB each
python tools/export_onnx.py --variant 16k    # just one
```
Needs `torch`, `onnx`, `onnxruntime`, `silentcipher`, `huggingface_hub`. Checkpoints are
pulled from the `sony/silentcipher` Hub repo. The script verifies each export against
PyTorch (100 % argmax agreement) before writing.

### Regenerating the test samples

```bash
pnpm run samples   # == python tools/make_samples.py --variant 44.1k && ... --variant 16k
python tools/make_samples.py --self-test     # verify the encoder against upstream
```

For 44.1 kHz this calls the real `encode_wav()` with Irodori's payload. For 16 kHz,
whose byte path is unusable, it replicates `encode_wav()`'s body with a raw symbol word;
`--self-test` proves that replication is bit-identical to upstream on the 44.1 kHz model
(relative difference 0.0e+00).

Run the script once per variant: `silentcipher`'s `STFT` class uses a `Singleton`
metaclass, so constructing both models in one process silently gives the second one the
first one's FFT size.

## How it works

```
file bytes
  -> WebAudio decodeAudioData @48 kHz   (native rate for Opus; no resampling)
  -> mono downmix
  -> sinc resample to 44.1 kHz          (torchaudio-compatible kernel, web/src/dsp.js)
  -> scale to SilentCipher's reference power (average_energy_VCTK)
  -> |STFT| 4096 / hop 2048, first 1024 bands
  -> dec_m_0  (ONNX, onnxruntime-web WASM EP)   -> 5 symbol logits per frame
  -> rotation correlation against the known payload
```

Only SilentCipher's **message decoder** (`dec_m_0`) is needed; the encoder and carrier
decoder are irrelevant for detection, which keeps the model at 9.5 MB.

### The detection statistic

The encoder tiles a fixed 21-symbol word across time — 20 payload symbols of two bits
each, plus a terminator — repeating once per 21 STFT frames (~0.98 s). Rather than only
blind-decoding the message, we score the *known* `IRDTS` word at all 21 rotations. The
20 wrong rotations form an empirical null distribution, so the test calibrates itself to
the material at hand and is far more sensitive on short excerpts than blind decoding.

Three independent quantities are reported:

- **`margin` / `pValue`** — payload-matched evidence. Drives the verdict.
- **`structure`** — payload-*independent* evidence that *some* SilentCipher watermark is
  present. Raw period-21 consistency is not usable alone (a pure tone makes the decoder
  emit a constant symbol, which folds perfectly at *every* period and would score ~1.0),
  so it is chance-corrected against neighbouring periods 19, 20, 22, 23. Every period
  folds exactly the same frames — otherwise an edge transient landing in one period's
  frame range but not another's reads as structure — and a near-constant output, which
  leaves the correction nothing to normalise against, scores 0 by construction.
- **`blind` / `symbolAgreement`** — the payload actually recovered, so a watermark
  carrying a *different* payload is never silently reported as Irodori's.

### Chunking is exact, not approximate

The decoder is 10 layers of 3x3 convolutions, so output frame *t* depends only on input
frames *t*±10. Each chunk is therefore computed with a 10-frame halo of real context
which is then discarded, making chunked inference bit-identical to one full-length
forward pass (asserted in `test/e2e.test.mjs`). This is what makes streaming progress
and early stopping safe.

## Calibration

Thresholds live in `web/src/variants.js`, one set per variant, calibrated on the
exported models against negative controls, worst case over every window position.

### 44.1 kHz — payload-matched

| window | positives (min margin / max p) | negatives (max margin / min p) |
|---|---|---|
| 4 s | 0.233 / 5.2e-06 | 0.276 / 1.4e-07 — **overlaps** |
| 5 s | 0.281 / 1.0e-09 | 0.218 / 3.3e-06 |
| 6 s | 0.291 / 4.7e-12 | 0.169 / 5.9e-05 — clean |
| 8 s | 0.322 / 1.8e-18 | 0.157 / 1.7e-05 |
| 10 s | 0.335 / 2.7e-24 | 0.146 / 7.1e-06 |

The hardest negative is a SilentCipher watermark with a *different* payload on
speech-like audio, which still correlates weakly with the Irodori word. **Four seconds
is not enough to separate the classes, so no verdict is issued below six seconds.**

### 16 kHz — payload-independent

| window | positives (min structure) | negatives (max structure) |
|---|---|---|
| 4 s | 0.326 | 0.440 — **overlaps** |
| 6 s | 0.217 | 0.239 — **still overlaps** |
| 8 s | 0.349 | 0.102 — clean |
| 12 s | 0.263 | 0.067 |

Positives are speech watermarked with the real 16 kHz model, raw and Opus-coded;
negatives include the same carriers unwatermarked plus 44.1 kHz-watermarked speech (the
key confusable), noise, a tone and a sweep. The weakest positives are Opus-coded
synthetic carriers. **This variant needs 8 s**, two more than 44.1 kHz.

## Limitations — read before trusting a verdict

- **Similar payloads correlate.** The encoder packs two bits per symbol, so payloads
  sharing high bits share symbols. ASCII words are the worst case — every printable
  letter starts `01`, so unrelated 5-letter payloads already agree on many positions.
  Measured on real samples at the 6 s window:

  | sample | searching for | margin | verdict |
  |---|---|---|---|
  | watermarked `HELLO` | `HELLO` | 0.531 | detected |
  | watermarked `IRDTS` | `IRDTS` | 0.477 | detected |
  | watermarked `HELLO` | `IRDTS` | 0.207 | inconclusive |
  | watermarked `IRDTS` | `HELLO` | 0.220 | inconclusive |

  The wrong-payload cases land just under the 0.23 detection threshold, so they are
  reported as *inconclusive* rather than matched — correct, but with less headroom than
  the unrelated-payload negatives used for calibration (max 0.169). A payload differing
  in a single byte (`IRDTX` vs `IRDTS`) correlates more strongly still and can be
  reported as detected. The UI therefore always shows the recovered payload and the
  symbol agreement, and warns when they disagree. Treat "detected" as "carries a
  SilentCipher watermark consistent with this payload", and confirm with the byte-exact
  read-out on a longer excerpt before claiming provenance.
- **Absence of evidence.** "No watermark found" only covers the audio actually analysed.
  Widen the *window* and disable early stopping before concluding anything. Note also
  that Irodori-TTS skips watermarking entirely when the `silentcipher` package or its
  checkpoint is unavailable — `watermark.py` logs *"SilentCipher package is unavailable;
  generated audio will not be watermarked"* and returns the audio untouched — so
  genuine Irodori-TTS output can legitimately carry no watermark at all.
- **Minimum length is a hard floor.** A file shorter than a variant's minimum window can
  never reach a verdict, and no window setting changes that; the audio simply is not
  there. Such variants are now detected from the file length and skipped instantly
  (they used to spend ~10 s of inference to report "too short"), with a note stating the
  actual duration, the requirement and the frame counts.
- **Speed.** A 44.1 kHz verdict needs 6 s of audio = 129 STFT frames, costing ~30 s in
  Firefox on 4 threads (~200-230 ms per frame). This is inherent: the decoder is
  ~2.4 GMAC per frame at 1024xT resolution, so one verdict is ~630 GFLOP.

  Wall-clock, measured end to end in the browser (44.1 kHz only, 4 threads):

  | audio | window | frames | wall clock |
  |---|---|---|---|
  | 14 s, watermarked | minimum | 129 | 29.8 s (stops early) |
  | 14 s, not watermarked | minimum (default) | 129 | 31.6 s |
  | 14 s, not watermarked | 12 s | 258 | 56.1 s |
  | **341 s**, watermarked | minimum | 129 | 26.5 s |

  Cost is set by the *window*, not the file length: a five-minute file costs the same as
  a fifteen-second one. That took a fix — the DSP front end used to resample and
  transform the whole file to use 129 frames of it:

  | 341 s file | before | after |
  |---|---|---|
  | resample 48 k -> 44.1 k | 5.2 s | 0.1 s |
  | STFT (7345 vs 145 frames) | 2.0 s | 0.04 s |
  | **front end total** | **7.2 s** | **0.2 s (40x)** |

  Scale by your CPU: these figures are from a 4-core machine, and the cost is almost
  entirely the WASM convolutions, so a machine half as fast takes twice as long. The
  result card breaks the wall clock down into decode / resample+STFT / inference.

  Three optimisations were implemented, measured and **rejected**, so nobody repeats them:

  | attempt | result |
  |---|---|
  | fuse each layer's `conv`+`gate` into one convolution (same maths, half the input reads) | bit-identical output, **3% slower** on both the CPU and WASM EPs — ORT already handles the pattern |
  | larger chunks to cut the 10-frame halo overhead (+101% at 21 frames vs +8% at 129) | only **1.6% faster**, because WASM here is memory-bandwidth bound: ms/computed-frame rises 104 -> 181 as tensors grow, cancelling the saved frames. Small chunks are kept so partial statistics stream out from ~8 s instead of nothing for 30 s |
  | int8 dynamic quantisation | **4x slower** than fp32 under ORT's WASM kernels |

- **Serve the page with the isolation headers, or pay 2.1x.** This is the biggest cliff
  there is. onnxruntime-web's thread pool needs `SharedArrayBuffer`, which needs a
  cross-origin isolated page. Without `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`, ORT keeps reporting the requested thread
  count but runs on **one** thread:

  | configuration | isolated | ms/frame | 6 s verdict |
  |---|---|---|---|
  | `pnpm run serve`, 4 threads | yes | 263 | ~34 s |
  | `pnpm run serve`, 2 threads | yes | 281 | ~36 s |
  | `pnpm run serve`, 1 thread | yes | 555 | ~72 s |
  | **plain static server / `file://`** | **no** | **555** | **~72 s** |

  The page now detects this and says so in a banner instead of just being slow; the
  thread box also persists across the reload that applies it (it previously reset
  itself, so changing it did nothing). Scaling saturates past two threads, which is
  consistent with the memory-bandwidth-bound behaviour above.

  The only large levers left are the thread count, the isolation headers, and a non-WASM
  backend (WebGPU).

  One deliberate deviation makes the front end cheap: the reference scales the signal by
  the mean power of the *resampled* waveform, which would force a full-length resample
  just to get one scalar. We measure whole-signal power at the input rate instead. The
  two differ by the energy above the new Nyquist frequency — 0.077 % on real audio, a
  0.039 % difference in gain — which was measured to leave verdict, match, margin,
  structure, shift, p-value and the recovered payload **bit-identical** on watermarked,
  Opus-coded and clean audio, and leaves the decoder's per-frame argmax 100 % unchanged
  against PyTorch (`test/parity.test.mjs`).

- **Why a verdict is slower than it used to be.** An early build issued verdicts from
  4 s of audio (~15 s of compute). Calibration later showed that 4 s cannot separate the
  classes — a SilentCipher watermark carrying a *different* payload reaches margin 0.276
  there, above the weakest true positive at 0.233 — so the minimum was raised to 6 s.
  That is a deliberate correctness trade, not a regression; partial statistics are shown
  while it runs so there is something to look at before the verdict lands.
- **16 kHz detections prove nothing about Irodori.** That variant carries no payload, so
  a detection only says "a SilentCipher watermark is here". Irodori-TTS never uses it.
- **Carrier matters.** SilentCipher was trained on speech and neither model recovers its
  own watermark from pink or gaussian noise, even at 49 dB SDR. Noise-like audio is
  effectively untestable rather than "clean".
- **The 44.1 kHz watermark is hard to remove by band-limiting.** It survived resampling
  to 16 kHz and even a 4 kHz low-pass in our checks, which is why the generated samples
  use a synthetic carrier rather than a filtered real one.
- **Robustness.** Not evaluated against adversarial watermark removal, heavy
  time-stretching, pitch shifting, or aggressive low-bitrate re-encoding.
  `pnpm run robustness` surveys format/sample-rate transcodes and time offsets.

### BatchNorm fidelity (important)

`silentcipher` never calls `.eval()` on its modules, so the upstream decoder runs its
10 `BatchNorm2d` layers with **batch** statistics. We deliberately export the
**eval-mode** (running-statistics) model instead, because batch statistics depend on how
much audio is in the tensor — which would make results window-dependent and break
chunking, streaming and early stopping.

The cost is measurable but small; both agree on the payload:

| clip | reference (batch stats) | shipped (running stats) |
|---|---|---|
| `part0` | `IRDTS`, conf 0.756 | `IRDTS`, conf 0.654 |
| `part3` | `IRDTS`, conf 0.824 | `IRDTS`, conf 0.756 |

The eval-mode model is thus slightly *less* sensitive — the conservative direction. The
thresholds above are calibrated on the shipped model, not on the reference.

Note for anyone reproducing these numbers: running a torch module in train mode
*mutates* its BatchNorm running statistics, so a train-mode pass before an eval-mode
pass will silently corrupt the eval result.

## Tests

```bash
pnpm run samples        # generate watermarked test audio (needs torch + silentcipher)
pnpm test               # DSP parity, detector logic, both variants end to end
pnpm run test:unit      # fast: DSP parity + detector logic only
pnpm run test:variants  # both variants through the WASM decoder
pnpm run test:payload   # arbitrary payloads: parsing, and detection of a custom word
pnpm run test:browser   # headless Firefox against the real page — authoritative
pnpm run test:browser:variants  # headless Firefox: both variants + auto probing
pnpm run test:history   # headless Firefox: result-history delete / clear-all controls
pnpm run test:queue     # headless Firefox: queue display, ordering and cancellation
pnpm run test:isolation # headless Firefox: COOP/COEP detection and the slowdown warning
pnpm run test:browser:payload  # headless Firefox: payload box validation and read-back
pnpm run test:dist      # headless Firefox: the built bundle on both hosting models
pnpm run robustness     # transcode / sample-rate / time-offset survey
```

| file | what it covers |
|---|---|
| `test/dsp.test.mjs` | JS resampler vs `torchaudio`, JS STFT vs `silentcipher`, detector statistics on synthetic symbol streams |
| `test/parity.test.mjs` | whole chain (PCM → resample → STFT → ONNX/WASM) vs PyTorch logits: 100 % argmax agreement, 3.8e-5 relative error |
| `test/variants.test.mjs` | both variants on generated samples, plus cross-variant independence |
| `test/payload.test.mjs` | payload parsing, and that a custom payload detects with the shipped thresholds |
| `test/e2e.test.mjs` | negative controls (noise, tone, clean carrier, time-reversal); chunked ≡ full-length inference |
| `test/browser.check.mjs` | real browser: WebAudio decode, worker, WASM EP, verdicts |
| `test/history.check.mjs` | real browser: per-result delete, clear-all, and the running guard |
| `test/queue.check.mjs` | real browser: queued cards, ordering, positions, pre-start removal |
| `test/isolation.check.mjs` | real browser: cross-origin isolation detection and its warning |
| `test/payload.check.mjs` | real browser: payload validation and which payload was searched |
| `test/dist.check.mjs` | the `dist/` bundle under Cloudflare-style headers, bare GitHub Pages and a subpath |
| `test/browser-variants.check.mjs` | real browser: variant selection and `auto` fall-through |

Every test drives the generated sample audio in `test/samples/`, so the suite is
self-contained and does not depend on any file being dropped into the repo. Regenerate
fixtures with `python tools/make_fixtures.py` and samples with `pnpm run samples` (both
need `torch`, `torchaudio`, `silentcipher`).

## Layout

```
build.sh              static bundle -> dist/ (Cloudflare Pages / GitHub Pages)
web/index.html        UI
web/coi-sw.js         service worker supplying COOP/COEP where the host cannot
web/src/coi.js        registers it and reloads once, when not already isolated
web/src/dsp.js        FFT, torchaudio-compatible sinc resampler, STFT
web/src/variants.js   per-variant model/DSP parameters and calibrated thresholds
web/src/detector.js   symbol coding, streaming accumulator, statistics, verdict
web/src/pipeline.js   front-end + chunked ONNX inference (runtime agnostic)
web/src/worker.js     keeps WASM inference off the UI thread
web/src/audio.js      WebAudio decoding
web/models/           exported ONNX decoders (both variants) + metadata
web/vendor/           onnxruntime-web WASM runtime (via pnpm run vendor)
tools/                ONNX export, fixture/sample generation, static server, vendoring
```

## Licence

The detector code here is provided as-is. The model weights come from
[`sony/silentcipher`](https://huggingface.co/sony/silentcipher) and remain under their
original licence.
