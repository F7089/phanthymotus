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

# Matcha ranking default is native TensorRT (TTS_MATCHA_TRT=1 from
# judgeflow_tts_run.sh) with host-cached cmpf32 + Vocos engines.
# sherpa dual CUDA ORT is TTS_MATCHA_TRT=0. Keep sherpa's ORT TensorRT EP
# disabled so ORT does not load a second nvinfer path.
export TTS_VOCOS_TRT="${TTS_VOCOS_TRT:-0}"
export TTS_VOCOS_TRT_CACHE="${TTS_VOCOS_TRT_CACHE:-/opt/vocos_trt_cache}"
export TTS_MATCHA_TRT="${TTS_MATCHA_TRT:-0}"
export TTS_MATCHA_TRT_CACHE="${TTS_MATCHA_TRT_CACHE:-/opt/matcha_trt_cache}"
mkdir -p "${TTS_VOCOS_TRT_CACHE}" "${TTS_MATCHA_TRT_CACHE}"
log "matcha TensorRT runtime enabled=${TTS_MATCHA_TRT} vocos_ort_hybrid=${TTS_VOCOS_TRT}"

# JP5 native TRT does not import sherpa / ORT. JP6 still uses sherpa CUDA
# Sessions — keep the ORT TensorRT EP disabled and the CUDA-provider file check.
if [ "${TTS_MATCHA_TRT}" = "1" ]; then
    log "skip sherpa/ORT probe (Matcha+Vocos TensorRT Runtime)"
    if [ "${TTS_REQUIRE_CUDA:-1}" = "1" ]; then
        log "checking TensorRT python module..."
        if ! trt_ok="$(python3 - <<'TRTCHK'
import tensorrt
print("tensorrt", getattr(tensorrt, "__version__", "?"))
TRTCHK
        )"; then
            log "FATAL: TTS_MATCHA_TRT=1 but import tensorrt failed."
            log "judgeflow must start the container with GPU runtime, for example:"
            log "  docker run --runtime nvidia \\"
            log "    -e NVIDIA_VISIBLE_DEVICES=all \\"
            log "    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \\"
            log "    ... <image>"
            exit 125
        fi
        log "TensorRT ok: ${trt_ok}"
    fi
else
    if [ "${TTS_DISABLE_TRT:-1}" = "1" ]; then
        log "disabling sherpa ORT TensorRT EP (native vocos uses TensorRT Runtime)..."
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
