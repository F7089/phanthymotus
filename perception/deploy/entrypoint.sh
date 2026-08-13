#!/bin/bash
# Jetson TTS perception entrypoint — verify CUDA before loading models.
# Note: do not use "set -u" here; ROS setup.bash references optional vars (e.g. COLCON_TRACE).
set -eo pipefail

log() { echo "[entrypoint] $*" >&2; }

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1

# ROS DDS: do not set ROS_DOMAIN_ID in the image — judgeflow injects it at
# docker run (must match robot-tts-evaluation). Log runtime values for debug.
log "starting (LD_PRELOAD=${LD_PRELOAD})"
log "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>} FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-<unset>}"

# Prefer onnxruntime CUDA check over torch: avoids loading PyTorch CUDA
# (~40–50MiB) when TTS inference is ORT-only. Same GPU requirement.
if [ "${TTS_REQUIRE_CUDA:-1}" = "1" ]; then
    log "checking CUDA via onnxruntime..."
    if ! cuda_ok="$(python3 - <<'ORTPY'
import sys
try:
    import onnxruntime as ort
except Exception:
    sys.exit(1)
if "CUDAExecutionProvider" not in ort.get_available_providers():
    sys.exit(1)
print("CUDAExecutionProvider")
ORTPY
    )"; then
        log "FATAL: hw_provider=cuda but CUDAExecutionProvider is not available."
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

# Do NOT prepend sherpa_onnx/lib to LD_LIBRARY_PATH. Piper uses onnxruntime-gpu;
# putting sherpa's ORT .so first reintroduces the ABI mix that segfaults.
# Sherpa OfflineTts (Melo etc.) already finds its libs via RPATH.

log "launching /work/main.py"
exec python3 /work/main.py
