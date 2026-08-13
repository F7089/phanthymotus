#!/usr/bin/env bash
set -euo pipefail
NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
OUT="${OUT_HOST:-$HOME/fanyi/wav_out/oov_parity.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$(dirname "$OUT")"
docker cp "${SCRIPT_DIR}/bench_tts_oov_parity.py" "${NAME}:/tmp/bench_tts_oov_parity.py"
docker exec -u 0 "${NAME}" python3 /tmp/bench_tts_oov_parity.py | tee "$OUT"
echo "wrote $OUT"
