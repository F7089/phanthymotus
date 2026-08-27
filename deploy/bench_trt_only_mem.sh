#!/usr/bin/env bash
# TensorRT-only memory (VITS2 runtime method): no sherpa / no ORT Session.
# Compare to tiny ORT CUDA Session ~786MB.
#
# trtexec runs in a throwaway container so build peak is NOT in the measured max.
#
#   IMAGE=$(docker inspect -f '{{.Config.Image}}' phanthymotus-perception-tts-0)
#   bash deploy/bench_trt_only_mem.sh "$IMAGE"
set -euo pipefail

IMAGE="${1:?image required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/perception/deploy/bench_trt_only_mem.py"
NAME="${TTS_CKPT_NAME:-phanthymotus-tts-ckpt}"
LIVE="${TTS_LIVE_CONTAINER:-phanthymotus-perception-tts-0}"
MODEL_DIR="${TTS_MODEL_DIR:-/models/matcha-kai-16k-e500}"
HOST_MODEL="${TTS_HOST_MODEL:-/tmp/matcha-kai-16k-e500-ckpt}"
CACHE_HOST="${TTS_VOCOS_TRT_CACHE_HOST:-/tmp/vocos_trt_cache}"
TINY_ONNX="/tmp/tiny_conv.onnx"
TINY_ENG="/opt/vocos_trt_cache/tiny_conv.fp16.ws64.engine"

mkdir -p "$CACHE_HOST"

VOLUME_ARGS=(-v "$CACHE_HOST":/opt/vocos_trt_cache)
if [[ -f "$HOST_MODEL/vocos-16khz-univ.onnx" ]]; then
  VOLUME_ARGS+=(-v "$HOST_MODEL:$MODEL_DIR:ro")
elif docker inspect "$LIVE" >/dev/null 2>&1 && \
     docker exec "$LIVE" test -f "$MODEL_DIR/vocos-16khz-univ.onnx"; then
  if [[ ! -f "$HOST_MODEL/vocos-16khz-univ.onnx" ]]; then
    echo "[ckpt] docker cp $LIVE:$MODEL_DIR -> $HOST_MODEL"
    rm -rf "$HOST_MODEL"
    docker cp "$LIVE:$MODEL_DIR" "$HOST_MODEL"
  fi
  VOLUME_ARGS+=(-v "$HOST_MODEL:$MODEL_DIR:ro")
fi

ensure_tiny_engine() {
  if [[ -f "$CACHE_HOST/tiny_conv.fp16.ws64.engine" ]]; then
    echo "[build] reuse tiny engine"
    return
  fi
  echo "[build] tiny engine (throwaway, not measured)"
  docker rm -f "$NAME-build" >/dev/null 2>&1 || true
  docker run --rm --name "$NAME-build" \
    --runtime nvidia --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    "${VOLUME_ARGS[@]}" \
    --entrypoint bash \
    "$IMAGE" \
    -lc '
set -e
python3 - <<'"'"'PY'"'"'
import os
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
w = numpy_helper.from_array(np.ones((8, 8, 1, 1), dtype=np.float32), name="W")
node = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[1, 1])
graph = helper.make_graph(
    [node], "tiny",
    [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 4, 4])],
    [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 4, 4])],
    [w],
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
model.ir_version = 8
onnx.save(model, "'"$TINY_ONNX"'")
print("wrote", "'"$TINY_ONNX"'", os.path.getsize("'"$TINY_ONNX"'"))
PY
exe=/usr/src/tensorrt/bin/trtexec
test -x "$exe" || exe=/usr/bin/trtexec
"$exe" --onnx='"$TINY_ONNX"' --saveEngine='"$TINY_ENG"'.tmp --fp16 --workspace=64
mv -f '"$TINY_ENG"'.tmp '"$TINY_ENG"'
ls -lah '"$TINY_ENG"'
'
}

ensure_vocos_engine() {
  if ls "$CACHE_HOST"/vocos-16khz-univ*.engine >/dev/null 2>&1; then
    echo "[build] reuse vocos engine $(ls "$CACHE_HOST"/vocos-16khz-univ*.engine | head -1)"
    return
  fi
  if [[ ! -f "$HOST_MODEL/vocos-16khz-univ.onnx" ]]; then
    echo "FATAL: no vocos onnx at $HOST_MODEL/vocos-16khz-univ.onnx" >&2
    exit 1
  fi
  echo "[build] vocos engine (throwaway, not measured)"
  docker rm -f "$NAME-build" >/dev/null 2>&1 || true
  docker run --rm --name "$NAME-build" \
    --runtime nvidia --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    "${VOLUME_ARGS[@]}" \
    --entrypoint bash \
    "$IMAGE" \
    -lc '
set -e
INP=$(python3 -c "import onnx; print(onnx.load(\"'"$MODEL_DIR"'/vocos-16khz-univ.onnx\").graph.input[0].name)")
echo "vocos_input=$INP"
exe=/usr/src/tensorrt/bin/trtexec
test -x "$exe" || exe=/usr/bin/trtexec
"$exe" --onnx='"$MODEL_DIR"'/vocos-16khz-univ.onnx \
  --saveEngine=/opt/vocos_trt_cache/vocos-16khz-univ.trt8.5.fp16.ws64.engine.tmp \
  --fp16 --workspace=64 \
  --minShapes=${INP}:1x80x16 --optShapes=${INP}:1x80x256 --maxShapes=${INP}:1x80x2000
mv -f /opt/vocos_trt_cache/vocos-16khz-univ.trt8.5.fp16.ws64.engine.tmp \
      /opt/vocos_trt_cache/vocos-16khz-univ.trt8.5.fp16.ws64.engine
ls -lah /opt/vocos_trt_cache/vocos-16khz-univ.trt8.5.fp16.ws64.engine
'
}

run_mode() {
  local mode="$1"
  local extra=("${@:2}")
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo
  echo "========== trt-only mode=$mode =========="
  docker run -d --name "$NAME" \
    --runtime nvidia --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e TTS_VOCOS_TRT_CACHE=/opt/vocos_trt_cache \
    "${VOLUME_ARGS[@]}" \
    -v "$PY":/deploy/bench_trt_only_mem.py:ro \
    --entrypoint python3 \
    "$IMAGE" \
    /deploy/bench_trt_only_mem.py --mode "$mode" "${extra[@]}"

  for _ in $(seq 1 90); do
    if docker logs "$NAME" 2>&1 | grep -qE 'CKPT_DONE|Traceback|SystemExit|engine missing'; then
      break
    fi
    if ! docker ps -q --filter "name=^/${NAME}$" | grep -q .; then
      break
    fi
    sleep 2
  done

  echo "----- logs -----"
  docker logs "$NAME" 2>&1 | grep -E 'CKPT |CKPT_DONE|imported |deserialize|warmup_|Traceback|Error|engine missing|SystemExit' || docker logs --tail 80 "$NAME"
  CID=$(docker inspect -f '{{.Id}}' "$NAME")
  CG="/sys/fs/cgroup/memory/docker/$CID"
  if [[ -f "$CG/memory.usage_in_bytes" ]]; then
    echo "----- host cgroup -----"
    awk '{printf "host_cgroup usage_MB=%.1f\n", $1/1024/1024}' "$CG/memory.usage_in_bytes"
    awk '{printf "host_cgroup max_MB=%.1f\n", $1/1024/1024}' "$CG/memory.max_usage_in_bytes"
  fi
  docker rm -f "$NAME" >/dev/null
}

ensure_tiny_engine
ensure_vocos_engine

run_mode import
run_mode tiny --engine "$TINY_ENG"
run_mode vocos

echo
echo "Baseline tiny ORT CUDA Session was ~786MB (same host cgroup max)."
echo "If tiny TRT D_warmup is still ~700MB+, dropping ORT will not cut the tax."
echo "If it is 150-400MB, Matcha-on-TRT (VITS2 method) is worth doing next."
