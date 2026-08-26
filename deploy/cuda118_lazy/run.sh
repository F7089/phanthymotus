#!/usr/bin/env bash
# JP5.1.1 host unchanged. CUDA 11.8 userspace + lazy loading tiny-ORT experiment.
#
#   bash deploy/cuda118_lazy/run.sh probe    # CUDA 11.8 + cuModuleGetLoadingMode
#   bash deploy/cuda118_lazy/run.sh tiny     # rebuild ORT 11.8 then tiny Session
#
# Do not use the live TTS container. Do not flash JP6.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="${1:-probe}"
NAME="${CUDA118_NAME:-phanthymotus-cuda118-lazy}"
PROBE_TAG="${CUDA118_PROBE_TAG:-phanthymotus-cuda118-lazy:probe}"
ORT_TAG="${CUDA118_ORT_TAG:-phanthymotus-cuda118-lazy:ort}"
JP_VERSION="${JP_VERSION:-511}"

cd "$ROOT"

run_ctr() {
  local image="$1"
  shift
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" --runtime nvidia --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e CUDA_MODULE_LOADING=LAZY \
    -e LD_LIBRARY_PATH=/usr/local/cuda-11.8/compat:/usr/local/cuda-11.8/lib64:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu \
    "$image" "$@"
}

wait_done() {
  local pat="$1"
  for _ in $(seq 1 180); do
    if docker logs "$NAME" 2>&1 | grep -qE "$pat"; then
      break
    fi
    if ! docker ps -q --filter "name=^/${NAME}$" | grep -q .; then
      break
    fi
    sleep 2
  done
  echo "----- logs -----"
  docker logs "$NAME" 2>&1 | tail -n 80
  CID=$(docker inspect -f '{{.Id}}' "$NAME")
  CG="/sys/fs/cgroup/memory/docker/$CID"
  if [[ -f "$CG/memory.usage_in_bytes" ]]; then
    echo "----- host cgroup -----"
    awk '{printf "host_cgroup usage_MB=%.1f\n", $1/1024/1024}' "$CG/memory.usage_in_bytes"
    awk '{printf "host_cgroup max_MB=%.1f\n", $1/1024/1024}' "$CG/memory.max_usage_in_bytes"
  fi
}

case "$MODE" in
  probe)
    echo "[cuda118] build probe image (no ORT rebuild)"
    docker build -f perception/Dockerfile.jetson.cuda118 --network=host \
      --build-arg JP_VERSION="$JP_VERSION" --build-arg BUILD_ORT=0 \
      -t "$PROBE_TAG" .
    run_ctr "$PROBE_TAG" /opt/cuda118_lazy/probe_loading_mode
    wait_done 'PROBE_OK=|cuInit failed|cuModuleGetLoadingMode failed'
    echo
    echo "Need: CUDA Driver/Runtime 11.8 and Module Loading Mode = LAZY, PROBE_OK=1"
    echo "Then: bash deploy/cuda118_lazy/run.sh tiny"
    ;;
  tiny)
    echo "[cuda118] build ORT against CUDA 11.8 (slow, 8GB: ORT_PARALLEL=2)"
    docker build -f perception/Dockerfile.jetson.cuda118 --network=host \
      --build-arg JP_VERSION="$JP_VERSION" --build-arg BUILD_ORT=1 \
      --build-arg ORT_PARALLEL="${ORT_PARALLEL:-2}" \
      -t "$ORT_TAG" .
    run_ctr "$ORT_TAG" python3 /opt/cuda118_lazy/bench_tiny_session.py
    wait_done 'CKPT_DONE tiny|Traceback|PROBE_OK=0'
    echo
    echo "Compare host_cgroup max_MB to the 11.4 baseline tiny=786MB"
    echo "  150-300MB: continue, rebuild sherpa"
    echo "  ~500MB: still useful for 5 containers"
    echo "  700MB+: stop, not the main cause"
    ;;
  *)
    echo "usage: bash deploy/cuda118_lazy/run.sh probe|tiny" >&2
    exit 1
    ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true
