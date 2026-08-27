#!/usr/bin/env bash
# JP5.1.1 host unchanged. CUDA 11.8 userspace + ORT rebuild + lazy loading.
# Does not modify perception/Dockerfile.jetson or the live TTS image.
#
# Fast A/B (skip standalone probe image):
#   bash deploy/cuda118_lazy/run.sh all
#
# Tiny first. If tiny cgroup max > 650MB, stop. Else acoustic+vocos CUDA sessions.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE="${1:-all}"
NAME="${CUDA118_NAME:-phanthymotus-cuda118-lazy}"
ORT_TAG="${CUDA118_ORT_TAG:-phanthymotus-cuda118-lazy:ort}"
JP_VERSION="${JP_VERSION:-511}"
TINY_STOP_MB="${CUDA118_TINY_STOP_MB:-650}"

cd "$ROOT"

print_cgroup() {
  local cid cg
  cid=$(docker inspect -f '{{.Id}}' "$NAME" 2>/dev/null || true)
  [[ -n "$cid" ]] || return 0
  for cg in \
    "/sys/fs/cgroup/memory/docker/$cid" \
    "/sys/fs/cgroup/memory/system.slice/docker-${cid}.scope"; do
    if [[ -f "$cg/memory.max_usage_in_bytes" ]]; then
      echo "----- host cgroup -----"
      awk '{printf "host_cgroup usage_MB=%.1f\n", $1/1024/1024}' "$cg/memory.usage_in_bytes"
      awk '{printf "host_cgroup max_MB=%.1f\n", $1/1024/1024}' "$cg/memory.max_usage_in_bytes"
      return 0
    fi
  done
}

run_ctr() {
  local image="$1"
  shift
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  local args=(
    docker run -d --name "$NAME" --runtime nvidia --privileged
    --network host
    -e NVIDIA_VISIBLE_DEVICES=all
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e CUDA_MODULE_LOADING=LAZY
    -e CUDA_HOME=/usr/local/cuda-11.8
    -e LD_LIBRARY_PATH=/usr/local/cuda-11.8/compat:/usr/local/cuda-11.8/lib64:/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu
  )
  if [[ -d /models ]]; then
    args+=(-v /models:/models)
  fi
  args+=("$image" "$@")
  echo "[cuda118] ${args[*]}"
  "${args[@]}"
}

wait_pat() {
  local pat="$1"
  local n="${2:-180}"
  for _ in $(seq 1 "$n"); do
    if docker logs "$NAME" 2>&1 | grep -qE "$pat"; then
      break
    fi
    if ! docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true; then
      break
    fi
    sleep 2
  done
  echo "----- logs -----"
  docker logs "$NAME" 2>&1 | tail -n 120
  print_cgroup
}

read_tiny_peak() {
  local cid cg
  cid=$(docker inspect -f '{{.Id}}' "$NAME" 2>/dev/null || true)
  for cg in \
    "/sys/fs/cgroup/memory/docker/$cid" \
    "/sys/fs/cgroup/memory/system.slice/docker-${cid}.scope"; do
    if [[ -f "$cg/memory.max_usage_in_bytes" ]]; then
      awk '{printf "%.1f", $1/1024/1024}' "$cg/memory.max_usage_in_bytes"
      return 0
    fi
  done
  echo "NA"
}

build_ort_image() {
  echo "[cuda118] docker build BUILD_ORT=1 (ORT against CUDA 11.8; 1-3h on 8GB)"
  docker build -f perception/Dockerfile.jetson.cuda118 --network=host \
    --build-arg JP_VERSION="$JP_VERSION" \
    --build-arg BUILD_ORT=1 \
    --build-arg ORT_PARALLEL="${ORT_PARALLEL:-2}" \
    -t "$ORT_TAG" .
}

run_tiny() {
  run_ctr "$ORT_TAG" python3 /opt/cuda118_lazy/bench_tiny_session.py
  wait_pat 'CKPT_DONE tiny|Traceback|CUDA error|cuInit failed|ORT CUDA EP' 120
}

run_full() {
  run_ctr "$ORT_TAG" python3 /opt/cuda118_lazy/bench_full_ort_sessions.py
  wait_pat 'FULL_ORT_CUDA_DONE|Traceback|CUDA error' 180
}

case "$MODE" in
  all)
    build_ort_image
    echo "[cuda118] tiny CUDA Session (compare to 11.4 tiny ≈ 786MB)"
    run_tiny
    tiny_peak="$(read_tiny_peak)"
    echo "CUDA11.8_lazy_tiny_peak_MB=${tiny_peak}"
    echo "CUDA11.4_old_tiny_peak_MB=786"
    if docker logs "$NAME" 2>&1 | grep -qE 'Traceback|CUDA error 803|CUDAExecutionProvider.*not|cuInit failed'; then
      echo "CUDA118_RESULT=FAIL tiny (see logs). Stop. Do not flash JetPack."
      docker rm -f "$NAME" >/dev/null 2>&1 || true
      exit 1
    fi
    set +e
    python3 - <<PY
peak = "${tiny_peak}"
try:
    v = float(peak)
except Exception:
    v = 9999.0
stop = float("${TINY_STOP_MB}")
print("tiny_peak", v, "stop_if_gt", stop)
if v > stop:
    raise SystemExit(2)
PY
    tiny_rc=$?
    set -e
    if [[ "$tiny_rc" == "2" ]]; then
      echo "CUDA118_RESULT=STOP tiny ${tiny_peak}MB > ${TINY_STOP_MB}MB. Do not continue."
      docker rm -f "$NAME" >/dev/null 2>&1 || true
      exit 2
    fi
    echo "[cuda118] tiny under cap; fresh container for acoustic+vocos (not the 11.4 sherpa wheel)"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    run_full
    echo "CUDA118_RESULT=tiny_ok_full_ran tiny_MB=${tiny_peak}"
    ;;
  tiny)
    build_ort_image
    run_tiny
    ;;
  full)
    run_full
    ;;
  probe)
    echo "[cuda118] probe-only image skipped in A/B; use: bash deploy/cuda118_lazy/run.sh all"
    exit 1
    ;;
  *)
    echo "usage: bash deploy/cuda118_lazy/run.sh all|tiny|full" >&2
    exit 1
    ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true
