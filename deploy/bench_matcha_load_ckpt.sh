#!/usr/bin/env bash
# Throwaway containers: base / acoustic-only / full / full+warmup.
# Does not touch phanthymotus-perception-tts-0.
#
#   IMAGE=$(docker inspect -f '{{.Config.Image}}' phanthymotus-perception-tts-0)
#   bash deploy/bench_matcha_load_ckpt.sh "$IMAGE"
set -euo pipefail

IMAGE="${1:?image required, e.g. phanthymotus-perception-tts:e19aee0}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY_HOST="$ROOT/perception/deploy/bench_matcha_load_ckpt.py"
NAME="${TTS_CKPT_NAME:-phanthymotus-tts-ckpt}"
MODEL_DIR="${TTS_MODEL_DIR:-/models/matcha-kai-16k-e500}"

if [[ ! -f "$PY_HOST" ]]; then
  echo "missing $PY_HOST" >&2
  exit 1
fi

run_one() {
  local mode="$1"
  local extra="${2:-}"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo
  echo "========== mode=$mode $extra =========="
  docker run -d --name "$NAME" \
    --runtime nvidia --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e TTS_DISABLE_TRT=1 \
    -e TTS_VOCOS_TRT=0 \
    -e TTS_MODEL_DIR="$MODEL_DIR" \
    -e TTS_SHERPA_ORT_CONFIG=/deploy/ort_cuda_jp5.config \
    -e TTS_VOCOS_TRT_CACHE=/opt/vocos_trt_cache \
    -v /models:/models \
    -v /tmp/vocos_trt_cache:/opt/vocos_trt_cache \
    -v "$PY_HOST":/deploy/bench_matcha_load_ckpt.py:ro \
    --entrypoint python3 \
    "$IMAGE" \
    /deploy/bench_matcha_load_ckpt.py --mode "$mode" $extra

  for _ in $(seq 1 90); do
    if docker logs "$NAME" 2>&1 | grep -q 'CKPT_DONE'; then
      break
    fi
    sleep 2
  done

  echo "----- logs -----"
  docker logs "$NAME" 2>&1 | grep -E 'CKPT |CKPT_DONE|mode=|warmup_samples|disabled|FATAL|Error|Traceback' || docker logs --tail 40 "$NAME"

  CID=$(docker inspect -f '{{.Id}}' "$NAME")
  CG="/sys/fs/cgroup/memory/docker/$CID"
  if [[ -f "$CG/memory.usage_in_bytes" ]]; then
    echo "----- host cgroup -----"
    awk '{printf "host_cgroup usage_MB=%.1f\n", $1/1024/1024}' "$CG/memory.usage_in_bytes"
    awk '{printf "host_cgroup max_MB=%.1f\n", $1/1024/1024}' "$CG/memory.max_usage_in_bytes"
  fi
  docker rm -f "$NAME" >/dev/null
}

run_one base
run_one acoustic
run_one full
run_one full --warmup

echo
echo "Read CKPT B_acoustic vs C_full heap/dmabuf. Vocos ≈ full - acoustic."
echo "CUDA context is inside the acoustic stage (first Session)."
