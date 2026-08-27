#!/usr/bin/env bash
# Compile Matcha acoustic ONNX -> TensorRT on the board (NOT docker build),
# then measure acoustic (+ Vocos) with zero ORT, same cgroup max as before.
#
# Engines stay on the host cache. Do not git them. Do not bake into the image.
#
# Build uses a lot of RAM: stop extra TTS replicas first if JP5 is tight.
# First fp16 compile can take 10-40+ minutes.
#
#   IMAGE=$(docker inspect -f '{{.Config.Image}}' phanthymotus-perception-tts-0)
#   bash deploy/bench_matcha_trt_mem.sh "$IMAGE"
set -euo pipefail

IMAGE="${1:?image required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/perception/deploy/bench_matcha_trt_mem.py"
SURGERY="$ROOT/perception/deploy/matcha_onnx_for_trt.py"
NAME="${TTS_CKPT_NAME:-phanthymotus-tts-ckpt}"
LIVE="${TTS_LIVE_CONTAINER:-phanthymotus-perception-tts-0}"
MODEL_DIR="${TTS_MODEL_DIR:-/models/matcha-kai-16k-e500}"
HOST_MODEL="${TTS_HOST_MODEL:-/tmp/matcha-kai-16k-e500-ckpt}"
MATCHA_CACHE_HOST="${TTS_MATCHA_TRT_CACHE_HOST:-/tmp/matcha_trt_cache}"
VOCOS_CACHE_HOST="${TTS_VOCOS_TRT_CACHE_HOST:-/tmp/vocos_trt_cache}"
WS="${TTS_TRT_WORKSPACE_MB:-4096}"
MAX_TOKENS="${TTS_TRT_MAX_TOKENS:-256}"
OPT_TOKENS="${TTS_TRT_OPT_TOKENS:-48}"
MIN_TOKENS="${TTS_TRT_MIN_TOKENS:-8}"
ONNX_NAME="${TTS_MATCHA_ONNX:-model-steps-3.onnx}"
MAX_MEL="${TTS_TRT_MAX_MEL:-2000}"
BUILD_LOG="${TTS_TRT_BUILD_LOG:-/tmp/matcha_trt_build.log}"
PATCHED_ONNX="/opt/matcha_trt_cache/model-steps-3.trtprep.L${MAX_TOKENS}.mel${MAX_MEL}.f32range.onnx"
ACOUSTIC_ENG_NAME="model-steps-3.trt8.5.fp16.ws${WS}.L${MAX_TOKENS}.mel${MAX_MEL}.f32range.engine"
ACOUSTIC_ENG="/opt/matcha_trt_cache/${ACOUSTIC_ENG_NAME}"

mkdir -p "$MATCHA_CACHE_HOST" "$VOCOS_CACHE_HOST"

if [[ ! -f "$PY" || ! -f "$SURGERY" ]]; then
  echo "missing $PY or $SURGERY" >&2
  exit 1
fi

VOLUME_ARGS=(
  -v "$MATCHA_CACHE_HOST":/opt/matcha_trt_cache
  -v "$VOCOS_CACHE_HOST":/opt/vocos_trt_cache
  -v "$SURGERY":/deploy/matcha_onnx_for_trt.py:ro
)
if [[ -f "$HOST_MODEL/$ONNX_NAME" ]]; then
  VOLUME_ARGS+=(-v "$HOST_MODEL:$MODEL_DIR:ro")
elif docker inspect "$LIVE" >/dev/null 2>&1 && \
     docker exec "$LIVE" test -f "$MODEL_DIR/$ONNX_NAME"; then
  if [[ ! -f "$HOST_MODEL/$ONNX_NAME" ]]; then
    echo "[ckpt] docker cp $LIVE:$MODEL_DIR -> $HOST_MODEL"
    rm -rf "$HOST_MODEL"
    docker cp "$LIVE:$MODEL_DIR" "$HOST_MODEL"
  fi
  VOLUME_ARGS+=(-v "$HOST_MODEL:$MODEL_DIR:ro")
else
  echo "FATAL: no $ONNX_NAME" >&2
  exit 1
fi

inspect_onnx() {
  echo "========== inspect $ONNX_NAME =========="
  docker run --rm \
    "${VOLUME_ARGS[@]}" \
    --entrypoint python3 \
    "$IMAGE" \
    -c '
import os, collections, onnx
p = os.path.join("'"$MODEL_DIR"'", "'"$ONNX_NAME"'")
m = onnx.load(p)
print("path", p)
print("bytes", os.path.getsize(p))
print("ir", m.ir_version)
print("opset", [(o.domain, o.version) for o in m.opset_import])
print("meta", {x.key: x.value for x in m.metadata_props})
for t in m.graph.input:
    dims = [d.dim_value or d.dim_param or "?" for d in t.type.tensor_type.shape.dim]
    et = t.type.tensor_type.elem_type
    print("IN", t.name, "elem", et, "shape", dims)
for t in m.graph.output:
    dims = [d.dim_value or d.dim_param or "?" for d in t.type.tensor_type.shape.dim]
    et = t.type.tensor_type.elem_type
    print("OUT", t.name, "elem", et, "shape", dims)
ops = collections.Counter(n.op_type for n in m.graph.node)
print("nodes", sum(ops.values()))
print("ops", ", ".join("%s=%s" % kv for kv in ops.most_common(25)))
'
}

prep_onnx() {
  local host_patched="$MATCHA_CACHE_HOST/$(basename "$PATCHED_ONNX")"
  if [[ -f "$host_patched" ]]; then
    echo "[prep] reuse $host_patched"
    return
  fi
  echo "[prep] fold length_scale=1.0 and static Range max_mel=$MAX_MEL"
  docker run --rm \
    "${VOLUME_ARGS[@]}" \
    --entrypoint python3 \
    "$IMAGE" \
    /deploy/matcha_onnx_for_trt.py \
    --in "$MODEL_DIR/$ONNX_NAME" \
    --out "$PATCHED_ONNX" \
    --length-scale 1.0 \
    --max-mel "$MAX_MEL"
}

build_acoustic() {
  if [[ -f "$MATCHA_CACHE_HOST/$ACOUSTIC_ENG_NAME" ]]; then
    echo "[build] reuse $MATCHA_CACHE_HOST/$ACOUSTIC_ENG_NAME"
    ls -lah "$MATCHA_CACHE_HOST/$ACOUSTIC_ENG_NAME"
    return
  fi
  prep_onnx
  echo "[build] acoustic trtexec fp16 workspace=${WS}MB maxL=${MAX_TOKENS} maxMel=${MAX_MEL}"
  echo "[build] onnx=$PATCHED_ONNX (not the raw sherpa graph)"
  echo "[build] log -> $BUILD_LOG"
  echo "[build] this is NOT docker build; uses the live TTS image GPU. 10-40+ min."
  docker rm -f "$NAME-build" >/dev/null 2>&1 || true
  set +e
  docker run --rm --name "$NAME-build" \
    --runtime nvidia --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    "${VOLUME_ARGS[@]}" \
    --entrypoint bash \
    "$IMAGE" \
    -lc '
set -e
exe=/usr/src/tensorrt/bin/trtexec
test -x "$exe" || exe=/usr/bin/trtexec
ls -lah '"$PATCHED_ONNX"'
"$exe" --onnx='"$PATCHED_ONNX"' \
  --saveEngine='"$ACOUSTIC_ENG"'.tmp \
  --fp16 \
  --workspace='"$WS"' \
  --minShapes=x:1x'"$MIN_TOKENS"',x_length:1 \
  --optShapes=x:1x'"$OPT_TOKENS"',x_length:1 \
  --maxShapes=x:1x'"$MAX_TOKENS"',x_length:1
mv -f '"$ACOUSTIC_ENG"'.tmp '"$ACOUSTIC_ENG"'
ls -lah '"$ACOUSTIC_ENG"'
' | tee "$BUILD_LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FATAL: trtexec failed rc=$rc ; see $BUILD_LOG" >&2
    echo "If it is an unsupported ONNX op, paste the last 80 lines." >&2
    exit $rc
  fi
}

find_vocos_eng() {
  local host
  host=$(ls -1 "$VOCOS_CACHE_HOST"/vocos-16khz-univ*.engine 2>/dev/null | head -1 || true)
  if [[ -n "$host" ]]; then
    echo "/opt/vocos_trt_cache/$(basename "$host")"
    return
  fi
  echo ""
}

run_measure() {
  local label="$1"
  local engines="$2"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo
  echo "========== measure $label =========="
  docker run -d --name "$NAME" \
    --runtime nvidia --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    "${VOLUME_ARGS[@]}" \
    -v "$PY":/deploy/bench_matcha_trt_mem.py:ro \
    --entrypoint python3 \
    "$IMAGE" \
    /deploy/bench_matcha_trt_mem.py --engines "$engines"

  for _ in $(seq 1 90); do
    if docker logs "$NAME" 2>&1 | grep -qE 'CKPT_DONE|Traceback|engine missing|execute_v2 failed'; then
      break
    fi
    if ! docker ps -q --filter "name=^/${NAME}$" | grep -q .; then
      break
    fi
    sleep 2
  done

  echo "----- logs -----"
  docker logs "$NAME" 2>&1 | grep -E 'CKPT |CKPT_DONE|imported |deserialize|warmup_|set_shape|input |output |Traceback|Error|engine missing' || docker logs --tail 80 "$NAME"
  CID=$(docker inspect -f '{{.Id}}' "$NAME")
  CG="/sys/fs/cgroup/memory/docker/$CID"
  if [[ -f "$CG/memory.usage_in_bytes" ]]; then
    echo "----- host cgroup -----"
    awk '{printf "host_cgroup usage_MB=%.1f\n", $1/1024/1024}' "$CG/memory.usage_in_bytes"
    awk '{printf "host_cgroup max_MB=%.1f\n", $1/1024/1024}' "$CG/memory.max_usage_in_bytes"
  fi
  docker rm -f "$NAME" >/dev/null
}

inspect_onnx
build_acoustic

VOCOS_ENG=$(find_vocos_eng)
run_measure acoustic-only "$ACOUSTIC_ENG"
if [[ -n "$VOCOS_ENG" ]]; then
  run_measure acoustic+vocos "${ACOUSTIC_ENG},${VOCOS_ENG}"
else
  echo "[skip] no vocos engine in $VOCOS_CACHE_HOST (run bench_trt_only_mem.sh once if needed)"
fi

echo
echo "Done. Ranking number is host_cgroup max_MB."
echo "Engine cache: $MATCHA_CACHE_HOST  (do not git add)"
echo "Build log: $BUILD_LOG"
