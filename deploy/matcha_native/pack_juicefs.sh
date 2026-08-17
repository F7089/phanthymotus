#!/usr/bin/env bash
# Put official Matcha assets on 74 JuiceFS data disk. Run on 74, not Jetson.
# Git must not contain these files (all >1MB except tiny fst/tokens).
#
#   bash deploy/matcha_native/pack_juicefs.sh
set -euo pipefail

JUICE_ROOT="${MELO_JUICE_ROOT:-/mnt/data/fanyi/phanthymotus_tts}"
WORKDIR="${MATCHA_PACK_WORKDIR:-$JUICE_ROOT/_matcha_dl}"
HTTP="http://172.28.4.81:34567/fanyi/phanthymotus_tts"
COS="https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public"
GH_TTS="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
GH_VOC="https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models"

mkdir -p "$JUICE_ROOT" "$WORKDIR"

fetch() {
  local dest="$1"
  shift
  if [[ -f "$dest" && $(stat -c%s "$dest" 2>/dev/null || stat -f%z "$dest") -gt 1024 ]]; then
    echo "skip existing $dest"
    return 0
  fi
  local url
  for url in "$@"; do
    echo "GET $url"
    if curl -fL --retry 3 -o "${dest}.part" "$url"; then
      mv "${dest}.part" "$dest"
      return 0
    fi
    rm -f "${dest}.part"
  done
  echo "failed: $dest" >&2
  return 1
}

fetch "$WORKDIR/matcha-icefall-zh-en.tar.bz2" \
  "$COS/matcha-icefall-zh-en.tar.bz2" \
  "$GH_TTS/matcha-icefall-zh-en.tar.bz2"

fetch "$WORKDIR/vocos-16khz-univ.onnx" \
  "$COS/vocos-16khz-univ.onnx" \
  "$GH_VOC/vocos-16khz-univ.onnx"

cp -a "$WORKDIR/matcha-icefall-zh-en.tar.bz2" "$JUICE_ROOT/matcha-icefall-zh-en.tar.bz2"
cp -a "$WORKDIR/vocos-16khz-univ.onnx" "$JUICE_ROOT/vocos-16khz-univ.onnx"

ls -lh "$JUICE_ROOT/matcha-icefall-zh-en.tar.bz2" "$JUICE_ROOT/vocos-16khz-univ.onnx"

echo "== HTTP =="
for name in matcha-icefall-zh-en.tar.bz2 vocos-16khz-univ.onnx; do
  echo "--- $HTTP/$name"
  curl -sI --noproxy '*' "$HTTP/$name" | head -5 || true
done

echo "packed. tar = acoustic + lexicon + fst + espeak; vocos = vocoder."
