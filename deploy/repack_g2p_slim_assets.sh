#!/usr/bin/env bash
# Publish production G2P pack onto JuiceFS (data disk).
#
# Split of responsibility:
#   - git: small code (english.py / slim_g2p_oov.py / this script)
#   - JuiceFS: large assets (openepd pickle, checkpoint20.npz, ONNX tars)
#
# Run on the physical host that owns /mnt/data (NOT Jetson).
# Jetson / leaderboard image only downloads:
#   http://172.28.4.81:34567/fanyi/phanthymotus_tts/melo-openepd-g2p-assets.tar.bz2
#
#   bash deploy/repack_g2p_slim_assets.sh
#
# Expects on data disk:
#   JUICE_ROOT/melo-openepd-g2p-assets.tar.bz2
#   JUICE_ROOT/checkpoint20.npz  (or /mnt/data/fanyi/tts/g2p/checkpoint20.npz)
# Slim .py: local deploy/melo_g2p_slim/ or GitHub raw.
set -euo pipefail

JUICE_ROOT="${MELO_JUICE_ROOT:-/mnt/data/fanyi/phanthymotus_tts}"
ASSETS_NAME="melo-openepd-g2p-assets"
TAR="${JUICE_ROOT}/${ASSETS_NAME}.tar.bz2"
CKPT_CANDIDATES=(
  "${JUICE_ROOT}/checkpoint20.npz"
  "/mnt/data/fanyi/tts/g2p/checkpoint20.npz"
)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLIM_DIR="${SCRIPT_DIR}/melo_g2p_slim"
GH_RAW="${GH_RAW:-https://raw.githubusercontent.com/F7089/phanthymotus/feat/fanyi-tts-lb/deploy/melo_g2p_slim}"
WORK="${WORK:-/tmp/repack_melo_g2p_$$}"

CKPT=""
for c in "${CKPT_CANDIDATES[@]}"; do
  if [[ -f "$c" ]]; then CKPT="$c"; break; fi
done
[[ -n "$CKPT" ]] || { echo "missing checkpoint20.npz under JuiceFS"; exit 1; }
[[ -f "$TAR" ]] || { echo "missing existing tar: $TAR"; exit 1; }

mkdir -p "$SLIM_DIR"
for f in english.py slim_g2p_oov.py; do
  if [[ ! -f "${SLIM_DIR}/${f}" ]]; then
    echo "fetch ${f} from GitHub raw..."
    curl -fsSL --noproxy '*' -o "${SLIM_DIR}/${f}" "${GH_RAW}/${f}"
  fi
done
[[ -f "${SLIM_DIR}/english.py" && -f "${SLIM_DIR}/slim_g2p_oov.py" ]] \
  || { echo "missing slim py sources"; exit 1; }

echo "== extract $TAR =="
rm -rf "$WORK"
mkdir -p "$WORK"
tar -xjf "$TAR" -C "$WORK"
PKG="${WORK}/${ASSETS_NAME}"
[[ -d "$PKG" ]] || PKG="$(find "$WORK" -maxdepth 1 -type d ! -path "$WORK" | head -1)"
TEXT="${PKG}/vendor/melo_g2p/text"
[[ -d "$TEXT" ]] || { echo "bad tar layout: no vendor/melo_g2p/text"; exit 1; }

echo "== overlay slim + ckpt =="
cp -a "${SLIM_DIR}/english.py" "${TEXT}/english.py"
cp -a "${SLIM_DIR}/slim_g2p_oov.py" "${TEXT}/slim_g2p_oov.py"
cp -a "$CKPT" "${TEXT}/checkpoint20.npz"

# Fail closed: must not ship g2p_en import
if grep -E 'from g2p_en import|G2p\(\)' "${TEXT}/english.py" >/dev/null; then
  echo "ERROR: english.py still references g2p_en"
  exit 1
fi

ls -lah "${TEXT}/english.py" "${TEXT}/slim_g2p_oov.py" "${TEXT}/checkpoint20.npz"

echo "== backup + write new tar =="
ts="$(date +%Y%m%d_%H%M%S)"
cp -a "$TAR" "${TAR}.bak_${ts}"
OUT_TAR="${JUICE_ROOT}/${ASSETS_NAME}.tar.bz2"
rm -f "$OUT_TAR"
tar -C "$WORK" -cjf "$OUT_TAR" "$(basename "$PKG")"
ls -lah "$OUT_TAR" "${TAR}.bak_${ts}"

echo "== HTTP check =="
url="http://172.28.4.81:34567/fanyi/phanthymotus_tts/${ASSETS_NAME}.tar.bz2"
curl -sI --noproxy '*' "$url" | head -5

echo
echo "OK: published $OUT_TAR"
echo "Jetson: rm -rf /models/melo-openepd-g2p-assets then re-download / restart TTS"
rm -rf "$WORK"
