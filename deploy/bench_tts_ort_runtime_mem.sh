#!/usr/bin/env bash
# One-shot: Runtime-style ORT only (memory goal). Baseline onnx≈1341, external≈1271.
# Success if run2 RSS clearly < ~1200; else drop ORT-format line.
set -euo pipefail
NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
OUT_HOST="${OUT_HOST:-$HOME/fanyi/wav_out/ort_runtime_mem.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORT_DIR="/tmp/melo_fp32_ort_runtime"

mkdir -p "$(dirname "$OUT_HOST")"
docker cp "${SCRIPT_DIR}/convert_melo_ort_format.py" "${NAME}:/tmp/convert_melo_ort_format.py"
docker cp "${SCRIPT_DIR}/bench_tts_ortfmt_mem.py" "${NAME}:/tmp/bench_tts_ortfmt_mem.py"

echo "== convert Runtime-style ORT -> ${ORT_DIR} =="
docker exec -u 0 "${NAME}" python3 /tmp/convert_melo_ort_format.py \
  --style Runtime --out-dir "${ORT_DIR}"

echo "== single case: ort_path (Runtime) =="
docker exec -u 0 \
  -e MELO_ORT_DIR="${ORT_DIR}" \
  -e BENCH_CASES=ort_path \
  "${NAME}" python3 /tmp/bench_tts_ortfmt_mem.py | tee "$OUT_HOST"

echo
echo "wrote $OUT_HOST"
echo "compare to: onnx≈1341  onnx_external≈1271"
echo "drop ORT-format unless run2 rss is clearly lower (e.g. <1200)"
