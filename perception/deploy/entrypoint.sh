#!/bin/bash
# Gentleman TTS: Python onnxruntime-gpu CUDA EP only. No sherpa / TensorRT.
set -eo pipefail

log() { echo "[entrypoint] $*" >&2; }

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

log "starting (LD_PRELOAD=${LD_PRELOAD})"
log "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>} FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-<unset>}"

if [ "${TTS_REQUIRE_CUDA:-1}" = "1" ]; then
    log "checking onnxruntime CUDAExecutionProvider..."
    if ! cuda_ok="$(python3 - <<'PY'
import onnxruntime as ort
avail = ort.get_available_providers()
if "CUDAExecutionProvider" not in avail:
    raise SystemExit("available=%s" % avail)
print("onnxruntime", ort.__version__, "providers", ",".join(avail))
PY
    )"; then
        log "FATAL: hw_provider=cuda but onnxruntime CUDAExecutionProvider is missing."
        log "judgeflow must start the container with GPU runtime, for example:"
        log "  docker run --runtime nvidia \\"
        log "    -e NVIDIA_VISIBLE_DEVICES=all \\"
        log "    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \\"
        log "    ... <image>"
        log "See deploy/judgeflow_tts_run.sh in this repository."
        exit 125
    fi
    log "CUDA ok: ${cuda_ok}"
fi

log "sourcing /opt/ros/humble/install/setup.bash"
source /opt/ros/humble/install/setup.bash
log "sourcing /ros_ws/install/setup.bash"
source /ros_ws/install/setup.bash

log "launching /work/main.py"
exec python3 /work/main.py
