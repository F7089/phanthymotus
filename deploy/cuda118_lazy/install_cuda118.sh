#!/bin/bash
# Install CUDA 11.8 toolkit + compat into a JP5 (r35) container.
# Host JetPack/BSP stays 11.4; userspace 11.8 is selected via LD_LIBRARY_PATH.
set -eux
export DEBIAN_FRONTEND=noninteractive
mkdir -p /usr/local/cuda-11.8/compat

if [[ ! -x /usr/local/cuda-11.8/bin/nvcc ]]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-keyring_1.1-1_all.deb \
    -O /tmp/cuda-keyring.deb || \
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-keyring_1.0-1_all.deb \
    -O /tmp/cuda-keyring.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get -o Acquire::AllowInsecureRepositories=true update
  # Pin 11.8. Do not apt-get install cuda (that pulls latest 12.x).
  apt-get install -y --no-install-recommends --allow-unauthenticated \
    cuda-toolkit-11-8 || apt-get install -y --no-install-recommends --allow-unauthenticated \
    cuda-nvcc-11-8 cuda-cudart-dev-11-8 cuda-nvrtc-11-8 cuda-cupti-11-8
  apt-get install -y --no-install-recommends --allow-unauthenticated \
    cuda-compat-11-8 || true
fi

# Some images put compat libs under /usr/lib/aarch64-linux-gnu/ or cuda-11.8/compat.
if [[ ! -e /usr/local/cuda-11.8/compat/libcuda.so ]] && \
   [[ ! -e /usr/local/cuda-11.8/compat/libcuda.so.1 ]]; then
  for d in /usr/local/cuda-11.8/compat \
           /usr/lib/aarch64-linux-gnu \
           /usr/local/cuda-11.8/lib64; do
    if ls "$d"/libcuda.so* >/dev/null 2>&1; then
      mkdir -p /usr/local/cuda-11.8/compat
      # shellcheck disable=SC2086
      cp -a "$d"/libcuda.so* /usr/local/cuda-11.8/compat/ 2>/dev/null || true
    fi
  done
fi

ls -l /usr/local/cuda-11.8/bin/nvcc
/usr/local/cuda-11.8/bin/nvcc --version
ls -l /usr/local/cuda-11.8/compat /usr/local/cuda-11.8/lib64 | head
ls /usr/local/cuda-11.8/compat/libcuda.so* || ls /usr/local/cuda-11.8/lib64/libcudart.so*
echo "[cuda118] toolkit ready"
