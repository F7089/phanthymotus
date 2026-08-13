#!/usr/bin/env bash
# Jetson: convert Melo FP32 → ORT (+external) and A/B RSS vs plain ONNX.
set -euo pipefail
NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
OUT_HOST="${OUT_HOST:-$HOME/fanyi/wav_out/ortfmt_mem.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

mkdir -p "$(dirname "$OUT_HOST")"
docker cp "${SCRIPT_DIR}/convert_melo_ort_format.py" "${NAME}:/tmp/convert_melo_ort_format.py"
docker cp "${SCRIPT_DIR}/bench_tts_ortfmt_mem.py" "${NAME}:/tmp/bench_tts_ortfmt_mem.py"

echo "== convert ONNX → ORT (+ external data) under /tmp/melo_fp32_ort =="
docker exec -u 0 "${NAME}" bash -lc "
set -e
python3 -c 'import onnx' 2>/dev/null || pip3 install -q -i '${PIP_INDEX}' 'onnx>=1.14,<1.18'
python3 /tmp/convert_melo_ort_format.py --external --out-dir /tmp/melo_fp32_ort
ls -lah /tmp/melo_fp32_ort/model.ort /tmp/melo_fp32_ort/model.with_external.onnx* 2>/dev/null || true
"

echo "== RSS A/B =="
docker exec -u 0 "${NAME}" python3 /tmp/bench_tts_ortfmt_mem.py | tee "$OUT_HOST"
echo "wrote $OUT_HOST"
echo "production try (after model.ort exists in model_dir):"
echo "  -e TTS_ORT_USE_MODEL_BYTES=1 -e TTS_ORT_DISABLE_PREPACKING=1"
