#!/usr/bin/env bash
# Jetson host: Nsight Systems profile of Melo FP32 CUDA process memory.
# Goal: process-scoped CUDA allocation peak (not cudaMemGetInfo whole-GPU).
#
#   bash deploy/bench_tts_nsys_cuda_mem.sh
#
# Needs nsys on the Jetson HOST (JetPack). Container only runs the worker.
set -euo pipefail

NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
OUT_DIR="${OUT_DIR:-$HOME/fanyi/wav_out}"
REP="${OUT_DIR}/melo_fp32_cuda_mem"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUT_DIR"

if ! command -v nsys >/dev/null 2>&1; then
  echo "FATAL: nsys not on PATH. On Jetson try:"
  echo "  export PATH=/opt/nvidia/nsight-systems/*/bin:\$PATH"
  echo "  # or: find /opt/nvidia -name nsys 2>/dev/null"
  exit 1
fi

echo "nsys=$(command -v nsys)"
nsys --version | head -3 || true

docker cp "${SCRIPT_DIR}/bench_tts_nsys_worker.py" "${NAME}:/tmp/bench_tts_nsys_worker.py"

echo "== nsys profile (cuda + cuda-memory-usage) =="
# Profile the docker-exec'd python (child) CUDA activity.
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output="$REP" \
  docker exec -u 0 "${NAME}" python3 /tmp/bench_tts_nsys_worker.py \
  | tee "${OUT_DIR}/melo_fp32_cuda_mem_stdout.txt"

echo
echo "report: ${REP}.nsys-rep"
echo "Open in Nsight Systems UI → Processes → CUDA → Memory / CUDA Memory Usage"
echo "Also try stats:"
echo "  nsys stats --report cuda_gpu_mem_size_sum ${REP}.nsys-rep | head -80"
echo "  nsys stats --report cuda_gpu_mem_time_sum ${REP}.nsys-rep | head -40"

if nsys stats --help 2>&1 | grep -q cuda_gpu_mem_size_sum; then
  nsys stats --report cuda_gpu_mem_size_sum "${REP}.nsys-rep" \
    | tee "${OUT_DIR}/melo_fp32_cuda_mem_stats.txt" || true
fi

echo "done"
