#!/bin/bash
# Jetson TTS perception entrypoint — verify CUDA before loading models.
# Note: do not use "set -u" here; ROS setup.bash references optional vars (e.g. COLCON_TRACE).
set -eo pipefail

log() { echo "[entrypoint] $*" >&2; }

export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
HOOK=/deploy/libort_cuda_mem_hook.so
if [ -f "$HOOK" ]; then
  export LD_PRELOAD="${HOOK}:${LD_PRELOAD}"
  log "ORT CUDA mem hook enabled (${HOOK})"
fi

# ROS DDS: do not set ROS_DOMAIN_ID in the image — judgeflow injects it at
# docker run (must match robot-tts-evaluation). Log runtime values for debug.
log "starting (LD_PRELOAD=${LD_PRELOAD})"
log "ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-<unset>} FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS:-<unset>}"

# Matcha uses sherpa-onnx's bundled CUDA ORT, not pip onnxruntime-gpu.
# Checking `import onnxruntime` would exit 125 after the official sherpa wheel.
#
# JP5: JetPack is CUDA 11.4 + TensorRT 8.5. Do not keep both EPs resident.
# Rename sherpa's TensorRT provider .so so ORT cannot register it.
if [ "${TTS_DISABLE_TRT:-1}" = "1" ]; then
    log "disabling TensorRT execution provider (TTS_DISABLE_TRT=1)..."
    python3 - <<'TRTPY' || true
import pathlib
import sherpa_onnx
libdir = pathlib.Path(sherpa_onnx.__file__).resolve().parent / "lib"
n = 0
for p in libdir.glob("libonnxruntime_providers_tensorrt*"):
    if p.suffix == ".disabled" or str(p).endswith(".disabled"):
        continue
    dest = p.with_name(p.name + ".disabled")
    try:
        p.rename(dest)
        print("disabled", dest.name)
        n += 1
    except OSError as e:
        print("skip", p.name, e)
print("tensorrt_disabled", n)
print("sherpa_lib", libdir)
print("so", sorted(x.name for x in libdir.glob("libonnxruntime_providers_*")))
if n == 0:
    print("note: no TensorRT provider .so in sherpa-onnx; dual-runtime is not this wheel")
TRTPY
fi

export TTS_ORT_USE_TRT="${TTS_ORT_USE_TRT:-0}"
export TTS_ORT_CUDNN_MAX_WORKSPACE="${TTS_ORT_CUDNN_MAX_WORKSPACE:-0}"
export TTS_ORT_ARENA_EXTEND="${TTS_ORT_ARENA_EXTEND:-kSameAsRequested}"
export TTS_ORT_GPU_MEM_LIMIT_MB="${TTS_ORT_GPU_MEM_LIMIT_MB:-256}"
export TTS_SHERPA_ORT_CONFIG="${TTS_SHERPA_ORT_CONFIG:-/deploy/ort_cuda_jp5.config}"
log "ort mem: TRT=${TTS_ORT_USE_TRT} workspace=${TTS_ORT_CUDNN_MAX_WORKSPACE} arena=${TTS_ORT_ARENA_EXTEND} gpu_mem_limit_mb=${TTS_ORT_GPU_MEM_LIMIT_MB}"

if [ "${TTS_REQUIRE_CUDA:-1}" = "1" ]; then
    log "checking CUDA via sherpa-onnx..."
    if ! cuda_ok="$(python3 - <<'ORTPY'
import pathlib
import sherpa_onnx
lib = pathlib.Path(sherpa_onnx.__file__).parent / "lib" / "libonnxruntime_providers_cuda.so"
if not lib.is_file():
    raise SystemExit(1)
print(str(lib))
ORTPY
    )"; then
        log "FATAL: hw_provider=cuda but sherpa CUDA provider library is missing."
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
