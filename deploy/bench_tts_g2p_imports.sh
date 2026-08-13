#!/usr/bin/env bash
set -euo pipefail
NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
OUT_HOST="${OUT_HOST:-$HOME/fanyi/wav_out/g2p_imports.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$(dirname "$OUT_HOST")"
docker cp "${SCRIPT_DIR}/bench_tts_g2p_imports.py" "${NAME}:/tmp/bench_tts_g2p_imports.py"
docker exec -u 0 "${NAME}" python3 /tmp/bench_tts_g2p_imports.py | tee "$OUT_HOST"
echo "wrote $OUT_HOST"
