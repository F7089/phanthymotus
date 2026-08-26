#!/bin/bash
# Build ONNX Runtime 1.16.3 CUDA EP against CUDA 11.8 (Orin SM 87).
# JP5 8GB: keep parallelism low or the compile will OOM.
set -eux
ORT_VER="${ORT_VER:-1.16.3}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
CUDNN_HOME="${CUDNN_HOME:-/usr}"
JOBS="${ORT_PARALLEL:-2}"
SRC=/opt/src/onnxruntime
WHEEL_DIR=/opt/cuda118_lazy/wheels
mkdir -p /opt/src "$WHEEL_DIR"

if [[ ! -d "$SRC/.git" ]]; then
  rm -rf "$SRC"
  git clone --recursive --branch "v${ORT_VER}" --depth 1 \
    https://ghfast.top/https://github.com/microsoft/onnxruntime.git "$SRC" \
    || git clone --recursive --branch "v${ORT_VER}" --depth 1 \
         https://github.com/microsoft/onnxruntime.git "$SRC"
fi

export PATH="${CUDA_HOME}/bin:${PATH}"
export CUDACXX="${CUDA_HOME}/bin/nvcc"
export CUDA_HOME
cd "$SRC"
# shellcheck disable=SC2086
./build.sh --config Release --update --build --build_wheel --build_shared_lib \
  --parallel ${JOBS} --skip_tests --allow_running_as_root \
  --use_cuda --cuda_home "${CUDA_HOME}" --cudnn_home "${CUDNN_HOME}" \
  --cuda_version 11.8 \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=87 \
    onnxruntime_NVCC_THREADS=1 \
    onnxruntime_BUILD_UNIT_TESTS=OFF

WHL=$(find "$SRC"/build/Linux/Release/dist -name 'onnxruntime_gpu-*.whl' -o -name 'onnxruntime-*.whl' | head -1)
test -n "$WHL"
cp -a "$WHL" "$WHEEL_DIR/"
pip3 install --no-cache-dir --force-reinstall "$WHL"
python3 - <<'PY'
import onnxruntime as ort
print("installed", ort.__version__, ort.get_available_providers())
assert "CUDAExecutionProvider" in ort.get_available_providers()
PY
echo "[cuda118] ORT wheel $WHL"
