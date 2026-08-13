#!/usr/bin/env bash
# Jetson: CUDA EP vs TensorRT EP (FP32 Melo) RSS + RTF stage bench.
# Run on host next to running container phanthymotus-tts-melo.
set -euo pipefail

NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
OUT_HOST="${OUT_HOST:-$HOME/fanyi/wav_out/trt_stages.txt}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$(dirname "$OUT_HOST")"
docker cp "${SCRIPT_DIR}/bench_tts_trt_stages.py" "${NAME}:/tmp/bench_tts_trt_stages.py"

echo "== TensorRT FP32 stage bench (fresh process) =="
# First TRT run builds engine (can take minutes); cache reused for run2.
docker exec -u 0 \
  -e TTS_ORT_TRT_WORKSPACE_MB="${TTS_ORT_TRT_WORKSPACE_MB:-512}" \
  "${NAME}" python3 /tmp/bench_tts_trt_stages.py | tee "$OUT_HOST"

echo
echo "wrote $OUT_HOST"
echo "optional full-service TRT (env only, no rebuild):"
echo "  docker rm -f $NAME && ..."
echo "  -e TTS_ORT_USE_TRT=1 -e TTS_ORT_TRT_WORKSPACE_MB=512 \\"
echo "  -v .../config.yaml:/work/config.yaml:ro"
