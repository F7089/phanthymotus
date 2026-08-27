#!/usr/bin/env bash
# Canonical docker run for TTS leaderboard on Jetson (GPU required).
#
# Ranking: git clone this repo, models/WeText from the data disk (/models),
# JP5 TensorRT engines from host cache (not git). Default is Matcha TRT
# (cmpf32), not sherpa CUDA.
#
# judgeflow reference service should call this script (or equivalent flags)
# instead of plain: docker run --privileged --network=host ...
#
# Usage:
#   ./deploy/judgeflow_tts_run.sh <image> <container_name> <mcp_port> <ws_port>
#
# Example (instance 0):
#   ./deploy/judgeflow_tts_run.sh phanthymotus-perception-tts:e19aee0 \
#     phanthymotus-perception-tts-0 15720 15721
set -euo pipefail

IMAGE="${1:?image required}"
NAME="${2:?container name required}"
MCP_PORT="${3:?MCP_PORT required}"
WS_PORT="${4:?WS_PORT required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

pick_host_dir() {
  local d
  for d in "$@"; do
    if [[ -n "$d" && -d "$d" ]]; then
      printf '%s' "$d"
      return 0
    fi
  done
  return 1
}

MATCHA_CACHE_HOST="${TTS_MATCHA_TRT_CACHE_HOST:-}"
if [[ -z "$MATCHA_CACHE_HOST" ]]; then
  MATCHA_CACHE_HOST="$(pick_host_dir /models/matcha_trt_cache /tmp/matcha_trt_cache || true)"
  MATCHA_CACHE_HOST="${MATCHA_CACHE_HOST:-/tmp/matcha_trt_cache}"
fi
VOCOS_CACHE_HOST="${TTS_VOCOS_TRT_CACHE_HOST:-}"
if [[ -z "$VOCOS_CACHE_HOST" ]]; then
  VOCOS_CACHE_HOST="$(pick_host_dir /models/vocos_trt_cache /tmp/vocos_trt_cache || true)"
  VOCOS_CACHE_HOST="${VOCOS_CACHE_HOST:-/tmp/vocos_trt_cache}"
fi
mkdir -p "${MATCHA_CACHE_HOST}" "${VOCOS_CACHE_HOST}"

TTS_MATCHA_TRT="${TTS_MATCHA_TRT:-1}"
ACOUSTIC_HOST="${TTS_MATCHA_TRT_ENGINE_HOST:-}"
if [[ -z "$ACOUSTIC_HOST" ]]; then
  ACOUSTIC_HOST="$(ls -1t "${MATCHA_CACHE_HOST}"/model-steps-3.trt8.5*.cmpf32.engine 2>/dev/null | head -1 || true)"
fi
VOCOS_HOST="${TTS_VOCOS_TRT_ENGINE_HOST:-}"
if [[ -z "$VOCOS_HOST" ]]; then
  VOCOS_HOST="$(ls -1t "${VOCOS_CACHE_HOST}"/vocos-16khz-univ*.engine 2>/dev/null | head -1 || true)"
fi

if [[ "$TTS_MATCHA_TRT" == "1" ]]; then
  if [[ -z "$ACOUSTIC_HOST" || ! -f "$ACOUSTIC_HOST" ]]; then
    echo "FATAL: TTS_MATCHA_TRT=1 but no cmpf32 engine." >&2
    echo "  expected ${MATCHA_CACHE_HOST}/model-steps-3.trt8.5*.cmpf32.engine" >&2
    echo "  (JP5 trtexec, default tactics; do not use tacticSources/preview plans)" >&2
    exit 1
  fi
  if [[ -z "$VOCOS_HOST" || ! -f "$VOCOS_HOST" ]]; then
    echo "FATAL: TTS_MATCHA_TRT=1 but no Vocos engine in ${VOCOS_CACHE_HOST}" >&2
    exit 1
  fi
fi

RUN_ARGS=(
    docker run -d
    --name "${NAME}"
    --runtime nvidia
    --network host
    --privileged
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e MCP_PORT="${MCP_PORT}"
    -e WS_PORT="${WS_PORT}"
    -e TTS_DISABLE_TRT="${TTS_DISABLE_TRT:-1}"
    -e TTS_VOCOS_TRT="${TTS_VOCOS_TRT:-0}"
    -e TTS_VOCOS_TRT_CACHE="${TTS_VOCOS_TRT_CACHE:-/opt/vocos_trt_cache}"
    -e TTS_MATCHA_TRT="${TTS_MATCHA_TRT}"
    -e TTS_MATCHA_TRT_CACHE=/opt/matcha_trt_cache
    -e TTS_TRT_PREFER_TACTICS=0
    -e TTS_ORT_USE_TRT="${TTS_ORT_USE_TRT:-0}"
    -e TTS_ORT_CUDNN_MAX_WORKSPACE="${TTS_ORT_CUDNN_MAX_WORKSPACE:-0}"
    -e TTS_ORT_ARENA_EXTEND="${TTS_ORT_ARENA_EXTEND:-kSameAsRequested}"
    -e TTS_ORT_GPU_MEM_LIMIT_MB="${TTS_ORT_GPU_MEM_LIMIT_MB:-256}"
    -e TTS_SHERPA_ORT_CONFIG="${TTS_SHERPA_ORT_CONFIG:-/deploy/ort_cuda_jp5.config}"
    -v "${VOCOS_CACHE_HOST}:/opt/vocos_trt_cache"
    -v "${MATCHA_CACHE_HOST}:/opt/matcha_trt_cache"
)

if [[ "$TTS_MATCHA_TRT" == "1" ]]; then
  RUN_ARGS+=(
    -e "TTS_MATCHA_TRT_ENGINE=/opt/matcha_trt_cache/$(basename "$ACOUSTIC_HOST")"
    -e "TTS_VOCOS_TRT_ENGINE=/opt/vocos_trt_cache/$(basename "$VOCOS_HOST")"
  )
fi

# Git checkout overrides the image /work copies so ranking can git pull
# without rebuilding phanthymotus-perception-tts.
if [[ -f "$ROOT/perception/plugins/tts.py" ]]; then
  RUN_ARGS+=(
    -v "$ROOT/perception/plugins/tts.py:/work/plugins/tts.py:ro"
    -v "$ROOT/perception/utils/matcha_trt.py:/work/utils/matcha_trt.py:ro"
    -v "$ROOT/perception/utils/vocos_trt.py:/work/utils/vocos_trt.py:ro"
  )
fi
if [[ -f "$ROOT/perception/deploy/entrypoint.sh" ]]; then
  RUN_ARGS+=(-v "$ROOT/perception/deploy/entrypoint.sh:/deploy/entrypoint.sh:ro")
fi

# Optional host model cache (if mounted on eval Jetson)
if [ -d /models ]; then
    RUN_ARGS+=(-v /models:/models)
fi

# ROS_DOMAIN_ID / FASTDDS: set by judgeflow at docker run (not baked into image).
if [ -n "${ROS_DOMAIN_ID:-}" ]; then
    RUN_ARGS+=(-e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
fi
if [ -n "${FASTDDS_BUILTIN_TRANSPORTS:-}" ]; then
    RUN_ARGS+=(-e "FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS}")
fi

RUN_ARGS+=("${IMAGE}")

echo "[judgeflow_tts_run] TTS_MATCHA_TRT=${TTS_MATCHA_TRT} acoustic=${ACOUSTIC_HOST:-} vocos=${VOCOS_HOST:-}"
echo "[judgeflow_tts_run] ${RUN_ARGS[*]}"
"${RUN_ARGS[@]}"
