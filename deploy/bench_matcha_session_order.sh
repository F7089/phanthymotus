#!/usr/bin/env bash
# First-CUDA-session tax: Vocos-first vs acoustic-first vs tiny-first.
# Same image/models as bench_matcha_load_ckpt.sh. Does not rebuild the TTS image.
#
#   IMAGE=$(docker inspect -f '{{.Config.Image}}' phanthymotus-perception-tts-0)
#   bash deploy/bench_matcha_session_order.sh "$IMAGE"
set -euo pipefail

IMAGE="${1:?image required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY_ORDER="$ROOT/perception/deploy/bench_matcha_session_order.py"
PY_DUMP="$ROOT/perception/deploy/bench_matcha_load_ckpt.py"
NAME="${TTS_CKPT_NAME:-phanthymotus-tts-ckpt}"
LIVE="${TTS_LIVE_CONTAINER:-phanthymotus-perception-tts-0}"
MODEL_DIR="${TTS_MODEL_DIR:-/models/matcha-kai-16k-e500}"

if [[ ! -f "$PY_ORDER" || ! -f "$PY_DUMP" ]]; then
  echo "missing $PY_ORDER or $PY_DUMP" >&2
  exit 1
fi

VOLUME_ARGS=(-v /tmp/vocos_trt_cache:/opt/vocos_trt_cache)
HOST_MODEL="${TTS_HOST_MODEL:-/tmp/matcha-kai-16k-e500-ckpt}"
if docker inspect "$LIVE" >/dev/null 2>&1 && \
   docker exec "$LIVE" test -f "$MODEL_DIR/model-steps-3.onnx"; then
  if [[ ! -f "$HOST_MODEL/model-steps-3.onnx" ]]; then
    echo "[ckpt] docker cp $LIVE:$MODEL_DIR -> $HOST_MODEL"
    rm -rf "$HOST_MODEL"
    docker cp "$LIVE:$MODEL_DIR" "$HOST_MODEL"
  fi
  VOLUME_ARGS+=(-v "$HOST_MODEL:$MODEL_DIR:ro")
  echo "[ckpt] models $HOST_MODEL -> $MODEL_DIR"
elif [[ -d /models ]]; then
  VOLUME_ARGS+=(-v /models:/models)
else
  echo "FATAL: no model-steps-3.onnx" >&2
  exit 1
fi

run_order() {
  local order="$1"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo
  echo "========== order=$order =========="
  docker run -d --name "$NAME" \
    --runtime nvidia --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e TTS_DISABLE_TRT=1 \
    -e TTS_MODEL_DIR="$MODEL_DIR" \
    "${VOLUME_ARGS[@]}" \
    -v "$PY_DUMP":/deploy/bench_matcha_load_ckpt.py:ro \
    -v "$PY_ORDER":/deploy/bench_matcha_session_order.py:ro \
    --entrypoint python3 \
    "$IMAGE" \
    /deploy/bench_matcha_session_order.py --order "$order"

  for _ in $(seq 1 120); do
    if docker logs "$NAME" 2>&1 | grep -qE 'CKPT_DONE|Traceback|OrtGetApiBase NULL|GetApi\(16\) NULL|CreateSession_ok'; then
      break
    fi
    if ! docker ps -q --filter "name=^/${NAME}$" | grep -q .; then
      break
    fi
    sleep 2
  done

  echo "----- logs -----"
  docker logs "$NAME" 2>&1 | grep -E 'CKPT |CKPT_DONE|CreateSession|dlopen|wrote_tiny|calling |ort_api|order=|Traceback|RuntimeError|FATAL|OSError' || docker logs --tail 80 "$NAME"

  CID=$(docker inspect -f '{{.Id}}' "$NAME")
  CG="/sys/fs/cgroup/memory/docker/$CID"
  if [[ -f "$CG/memory.usage_in_bytes" ]]; then
    echo "----- host cgroup -----"
    awk '{printf "host_cgroup usage_MB=%.1f\n", $1/1024/1024}' "$CG/memory.usage_in_bytes"
    awk '{printf "host_cgroup max_MB=%.1f\n", $1/1024/1024}' "$CG/memory.max_usage_in_bytes"
  fi
  docker rm -f "$NAME" >/dev/null
}

run_order vocos,acoustic
run_order acoustic,vocos
run_order tiny,acoustic

echo
echo "Compare first-session CKPT B_* cgroup/heap across the three runs."
echo "If Vocos-first and tiny-first both jump ~800MB, that is CUDA first-session tax."
echo "If only acoustic jumps ~800MB, the Matcha graph is the fat one."
echo "Live TTS $LIVE is left running; docker stop it first if the board is tight."
