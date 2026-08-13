#!/usr/bin/env bash
# Only measure RSS stages (host base vs session vs run). No finetune / re-export.
set -euo pipefail
NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
OUT_HOST="${OUT_HOST:-$HOME/fanyi/wav_out/rss_stages.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$(dirname "$OUT_HOST")"
docker cp "${SCRIPT_DIR}/bench_tts_rss_stages.py" "${NAME}:/tmp/bench_tts_rss_stages.py"

# external model optional; if missing, only onnx case matters
docker exec -u 0 \
  -e BENCH_CASES="${BENCH_CASES:-onnx,onnx_external}" \
  -e MELO_ORT_DIR="${MELO_ORT_DIR:-/tmp/melo_fp32_ort}" \
  "${NAME}" python3 /tmp/bench_tts_rss_stages.py | tee "$OUT_HOST"

echo "wrote $OUT_HOST"
