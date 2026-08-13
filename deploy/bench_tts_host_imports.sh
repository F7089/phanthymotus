#!/usr/bin/env bash
# Measure host_base (~566MB) import-by-import. No ORT session / no finetune.
set -euo pipefail
NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
OUT_HOST="${OUT_HOST:-$HOME/fanyi/wav_out/host_imports.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$(dirname "$OUT_HOST")"
docker cp "${SCRIPT_DIR}/bench_tts_host_imports.py" "${NAME}:/tmp/bench_tts_host_imports.py"
# Fresh python process inside container
docker exec -u 0 "${NAME}" python3 /tmp/bench_tts_host_imports.py | tee "$OUT_HOST"
echo "wrote $OUT_HOST"
