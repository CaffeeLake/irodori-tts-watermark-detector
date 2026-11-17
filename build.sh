#!/usr/bin/env bash
#
# Build a static bundle in dist/ for Cloudflare Pages or GitHub Pages.
#
#   ./build.sh                 # dist/
#   ./build.sh --out public    # somewhere else
#   ./build.sh --no-16k        # drop the 16 kHz model (~9.5 MB smaller)
#
# There is nothing to compile: the app is plain ES modules plus the ONNX models and the
# onnxruntime-web runtime. The build copies those, then adds the bits each host needs:
#
#   _headers    Cloudflare Pages reads this and sets COOP/COEP, which onnxruntime-web
#               needs for SharedArrayBuffer (and hence its WASM thread pool).
#   .nojekyll   GitHub Pages runs Jekyll by default, which silently drops files and
#               directories whose names start with an underscore - including _headers.
#   coi-sw.js   GitHub Pages cannot set response headers at all, so a service worker
#               supplies them instead (see web/coi-sw.js). Already copied with web/.
#
# Without cross-origin isolation inference runs single-threaded: measured 555 vs
# 263 ms per STFT frame, i.e. 2.1x slower.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$ROOT/dist"
INCLUDE_16K=1

while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
    --no-16k) INCLUDE_16K=0; shift ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

say() { printf '%s\n' "$*"; }
die() { printf 'build.sh: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- prerequisites ----
[ -f "$ROOT/web/index.html" ] || die "web/index.html not found; run from the repo"

if [ ! -f "$ROOT/web/vendor/ort.wasm.min.mjs" ]; then
  say "vendor/ is empty - running tools/vendor.mjs"
  [ -d "$ROOT/node_modules/onnxruntime-web" ] \
    || die "node_modules/onnxruntime-web missing; run 'pnpm install' (or npm install) first"
  node "$ROOT/tools/vendor.mjs" >/dev/null
fi

[ -f "$ROOT/web/models/dec_m_0.44k1.onnx" ] \
  || die "web/models/dec_m_0.44k1.onnx missing; run 'python tools/export_onnx.py'"
[ -f "$ROOT/web/models/meta.json" ] || die "web/models/meta.json missing; re-run the export"

# ------------------------------------------------------------------------ copy ----
rm -rf "$OUT"
mkdir -p "$OUT"

# Everything the page actually loads. Excludes nothing from web/ except the 16 kHz
# model when --no-16k is given.
cp "$ROOT/web/index.html" "$OUT/"
cp "$ROOT/web/coi-sw.js" "$OUT/"
cp -R "$ROOT/web/src" "$OUT/src"
cp -R "$ROOT/web/vendor" "$OUT/vendor"
mkdir -p "$OUT/models"
cp "$ROOT/web/models/meta.json" "$OUT/models/"
cp "$ROOT/web/models/dec_m_0.44k1.onnx" "$OUT/models/"
if [ "$INCLUDE_16K" -eq 1 ] && [ -f "$ROOT/web/models/dec_m_0.16k.onnx" ]; then
  cp "$ROOT/web/models/dec_m_0.16k.onnx" "$OUT/models/"
else
  say "note: 16 kHz model excluded - selecting that variant in the UI will fail to load"
fi

# Source maps are not shipped; they would double the bundle for no benefit here.
find "$OUT" -name '*.map' -delete

# -------------------------------------------------------------- host-specific ----
# Cloudflare Pages: _headers. The models and runtime are large and versioned by
# content, so they get a long cache; HTML and JS stay revalidated so a redeploy is
# picked up immediately.
cat > "$OUT/_headers" <<'EOF'
/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
  Cross-Origin-Resource-Policy: same-origin
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer

/models/*
  Cache-Control: public, max-age=86400

/vendor/*
  Cache-Control: public, max-age=86400

/index.html
  Cache-Control: no-cache

/src/*
  Cache-Control: no-cache

/coi-sw.js
  Cache-Control: no-cache
EOF

# GitHub Pages: stop Jekyll from eating _headers and any underscore-prefixed asset.
touch "$OUT/.nojekyll"

# ---------------------------------------------------------------------- verify ----
missing=0
for f in index.html coi-sw.js _headers .nojekyll \
         src/app.js src/worker.js src/detector.js src/variants.js src/pipeline.js \
         src/dsp.js src/audio.js src/coi.js \
         models/meta.json models/dec_m_0.44k1.onnx \
         vendor/ort.wasm.min.mjs vendor/ort-wasm-simd-threaded.mjs \
         vendor/ort-wasm-simd-threaded.wasm; do
  [ -e "$OUT/$f" ] || { printf 'MISSING: %s\n' "$f" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || die "incomplete bundle"

# The page must not reference anything absolute, or it breaks under the /<repo>/ prefix
# GitHub Pages uses for project sites.
if grep -RIn --include='*.html' --include='*.js' -E '(src|href)="/[^/]' "$OUT" ; then
  die "absolute paths found; they break GitHub Pages project sites"
fi

say ""
say "built $(cd "$OUT" && find . -type f | wc -l | tr -d ' ') files into ${OUT#"$ROOT"/}"
du -sh "$OUT" | awk '{print "total: " $1}'
say ""
say "largest files:"
find "$OUT" -type f -printf '%s\t%P\n' 2>/dev/null | sort -rn | head -5 \
  | awk '{printf "  %6.1f MB  %s\n", $1/1048576, $2}'
say ""
say "deploy:"
say "  Cloudflare Pages  wrangler pages deploy ${OUT#"$ROOT"/}      (_headers sets COOP/COEP)"
say "  GitHub Pages      publish ${OUT#"$ROOT"/} to your Pages branch"
say "                    headers are impossible there, so coi-sw.js supplies them;"
say "                    the page reloads once on first visit to take effect"
say ""
say "check after deploying: the page must NOT show the 'Running single-threaded' banner."
