#!/usr/bin/env bash
# Canonical docker run for TTS leaderboard on Jetson (GPU required).
#
# judgeflow reference service should call this script (or equivalent flags)
# instead of plain: docker run --privileged --network=host ...
#
# Usage:
#   ./deploy/judgeflow_tts_run.sh <image> <container_name> <mcp_port> <ws_port>
#
# Example (instance 0):
#   ./deploy/judgeflow_tts_run.sh phanthymotus-perception-tts:3794763 \
#     phanthymotus-perception-tts-0 15720 15721
set -euo pipefail

IMAGE="${1:?image required}"
NAME="${2:?container name required}"
MCP_PORT="${3:?MCP_PORT required}"
WS_PORT="${4:?WS_PORT required}"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

CACHE_HOST="${TTS_VOCOS_TRT_CACHE_HOST:-/tmp/vocos_trt_cache}"
mkdir -p "${CACHE_HOST}"

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
    -e TTS_VOCOS_TRT="${TTS_VOCOS_TRT:-1}"
    -e TTS_VOCOS_TRT_CACHE="${TTS_VOCOS_TRT_CACHE:-/opt/vocos_trt_cache}"
    -e TTS_ORT_USE_TRT="${TTS_ORT_USE_TRT:-0}"
    -e TTS_ORT_CUDNN_MAX_WORKSPACE="${TTS_ORT_CUDNN_MAX_WORKSPACE:-0}"
    -e TTS_ORT_ARENA_EXTEND="${TTS_ORT_ARENA_EXTEND:-kSameAsRequested}"
    -e TTS_ORT_GPU_MEM_LIMIT_MB="${TTS_ORT_GPU_MEM_LIMIT_MB:-256}"
    -e TTS_SHERPA_ORT_CONFIG="${TTS_SHERPA_ORT_CONFIG:-/deploy/ort_cuda_jp5.config}"
    -v "${CACHE_HOST}:/opt/vocos_trt_cache"
)

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

echo "[judgeflow_tts_run] ${RUN_ARGS[*]}"
"${RUN_ARGS[@]}"
