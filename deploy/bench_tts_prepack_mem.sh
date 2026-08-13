#!/usr/bin/env bash
# Jetson: Melo FP32 CUDA A/B with session.disable_prepacking.
set -euo pipefail
NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
OUT_HOST="${OUT_HOST:-$HOME/fanyi/wav_out/prepack_mem.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$(dirname "$OUT_HOST")"
docker cp "${SCRIPT_DIR}/bench_tts_prepack_mem.py" "${NAME}:/tmp/bench_tts_prepack_mem.py"
docker exec -u 0 "${NAME}" python3 /tmp/bench_tts_prepack_mem.py | tee "$OUT_HOST"
echo "wrote $OUT_HOST"
echo "production try: -e TTS_ORT_DISABLE_PREPACKING=1"
