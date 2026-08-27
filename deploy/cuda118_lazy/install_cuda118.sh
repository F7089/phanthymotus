#!/bin/bash
# Install CUDA 11.8 toolkit + compat into a JP5 (r35) container.
# Host JetPack/BSP stays 11.4; userspace 11.8 is selected via LD_LIBRARY_PATH.
set -eux
export DEBIAN_FRONTEND=noninteractive
mkdir -p /usr/local/cuda-11.8/compat

if [[ ! -x /usr/local/cuda-11.8/bin/nvcc ]]; then
  # Jetson upgrade package first (tegra), then generic ubuntu2004/arm64.
  tegra_deb=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-tegra-repo-ubuntu2004-11-8-local_11.8.0-1_arm64.deb
  if wget -q -O /tmp/cuda-tegra-11-8.deb "$tegra_deb"; then
    dpkg -i /tmp/cuda-tegra-11-8.deb || true
    cp -a /var/cuda-tegra-repo-ubuntu2004-11-8-local/*.gpg /usr/share/keyrings/ 2>/dev/null || true
    apt-get -o Acquire::AllowInsecureRepositories=true update
    apt-get install -y --no-install-recommends --allow-unauthenticated \
      cuda-tegra-repo-ubuntu2004-11-8-local || true
    apt-get install -y --no-install-recommends --allow-unauthenticated \
      cuda-toolkit-11-8 cuda-compat-11-8 || true
  fi
fi

if [[ ! -x /usr/local/cuda-11.8/bin/nvcc ]]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-keyring_1.1-1_all.deb \
    -O /tmp/cuda-keyring.deb || \
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/arm64/cuda-keyring_1.0-1_all.deb \
    -O /tmp/cuda-keyring.deb
  dpkg -i /tmp/cuda-keyring.deb
  apt-get -o Acquire::AllowInsecureRepositories=true update
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
