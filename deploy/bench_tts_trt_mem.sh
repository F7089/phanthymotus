#!/usr/bin/env bash
# Jetson: shape-infer FP32 ONNX, then CUDA vs TensorRT EP RSS/RTF (fresh processes).
set -euo pipefail

NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
OUT_HOST="${OUT_HOST:-$HOME/fanyi/wav_out/trt_stages.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_IN="${MODEL_IN:-/models/vits-melo-longanlingxin-openepd-nobert-44100-fp32/model.onnx}"
MODEL_SHAPED="${MODEL_SHAPED:-/tmp/melo_fp32_shaped.onnx}"
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"

mkdir -p "$(dirname "$OUT_HOST")"
docker cp "${SCRIPT_DIR}/shape_infer_melo_onnx.py" "${NAME}:/tmp/shape_infer_melo_onnx.py"
docker cp "${SCRIPT_DIR}/bench_tts_trt_stages.py" "${NAME}:/tmp/bench_tts_trt_stages.py"

echo "== ensure onnx + shape-infer (must see 'wrote ...shaped.onnx') =="
docker exec -u 0 "${NAME}" bash -lc "
set -e
python3 -c 'import onnx' 2>/dev/null || pip3 install -q -i '${PIP_INDEX}' 'onnx>=1.14,<1.18'
rm -f '${MODEL_SHAPED}' '${MODEL_SHAPED}.ok'
python3 /tmp/shape_infer_melo_onnx.py --input '${MODEL_IN}' --output '${MODEL_SHAPED}'
test -f '${MODEL_SHAPED}.ok'
ls -lah '${MODEL_SHAPED}' '${MODEL_SHAPED}.ok'
"

echo "== CUDA vs TensorRT stage bench (subprocess-isolated) =="
docker exec -u 0 \
  -e TTS_ORT_TRT_WORKSPACE_MB="${TTS_ORT_TRT_WORKSPACE_MB:-512}" \
  -e MELO_ONNX_TRT="${MODEL_SHAPED}" \
  "${NAME}" python3 /tmp/bench_tts_trt_stages.py | tee "$OUT_HOST"

echo
echo "wrote $OUT_HOST"
echo "expect: TRT before_session RSS near g2p-only (~500MB), not ~1350MB (that meant old same-process script)"
