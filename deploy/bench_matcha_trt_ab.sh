#!/usr/bin/env bash
# A/B full-stack cgroup peak. One change per run. Judgeflow-equivalent flags.
#
#   IMAGE=phanthymotus-perception-tts:e19aee0
#   bash deploy/bench_matcha_trt_ab.sh "$IMAGE"
set -euo pipefail

IMAGE="${1:?image required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MCP_PORT="${MCP_PORT:-16720}"
export WS_PORT="${WS_PORT:-16721}"

run_case() {
  local label="$1"
  shift
  echo
  echo "######## A/B $label ########"
  TTS_BENCH_LABEL="$label" TTS_FULL_NAME="phanthymotus-tts-trt-ab" \
    "$@" bash "$ROOT/deploy/bench_matcha_trt_full.sh" "$IMAGE" \
    | tee "/tmp/tts_ab_${label}.log" \
    | grep -E 'host_cgroup max_MB=|EXTRA_TTS |FULLSTACK_INFER=|FULLSTACK_PEAK' || true
}

# Old fullstack behavior on the new host-network baseline.
run_case baseline \
  env TTS_VOCOS_PARSE_ONNX=1 TTS_ISTFT=scipy TTS_TRT_LOAD_ORDER=frontend_first \
      TTS_TRT_BUF_REUSE=0 TTS_RANKING_MODE=0

run_case no_onnx_load \
  env TTS_VOCOS_PARSE_ONNX=0 TTS_ISTFT=scipy TTS_TRT_LOAD_ORDER=frontend_first \
      TTS_TRT_BUF_REUSE=0 TTS_RANKING_MODE=0

run_case no_scipy \
  env TTS_VOCOS_PARSE_ONNX=0 TTS_ISTFT=numpy TTS_TRT_LOAD_ORDER=frontend_first \
      TTS_TRT_BUF_REUSE=0 TTS_RANKING_MODE=0

run_case load_order \
  env TTS_VOCOS_PARSE_ONNX=0 TTS_ISTFT=numpy TTS_TRT_LOAD_ORDER=trt_first \
      TTS_TRT_BUF_REUSE=0 TTS_RANKING_MODE=0

run_case buf_reuse \
  env TTS_VOCOS_PARSE_ONNX=0 TTS_ISTFT=numpy TTS_TRT_LOAD_ORDER=frontend_first \
      TTS_TRT_BUF_REUSE=1 TTS_RANKING_MODE=0

run_case ranking_mode \
  env TTS_VOCOS_PARSE_ONNX=0 TTS_ISTFT=numpy TTS_TRT_LOAD_ORDER=frontend_first \
      TTS_TRT_BUF_REUSE=0 TTS_RANKING_MODE=1

run_case combo \
  env TTS_VOCOS_PARSE_ONNX=0 TTS_ISTFT=numpy TTS_TRT_LOAD_ORDER=trt_first \
      TTS_TRT_BUF_REUSE=1 TTS_RANKING_MODE=1

echo
echo "======== A/B summary (grep max_MB) ========"
for f in /tmp/tts_ab_{baseline,no_onnx_load,no_scipy,load_order,buf_reuse,ranking_mode,combo}.log; do
  [[ -f "$f" ]] || continue
  printf '%-16s ' "$(basename "$f" .log | sed 's/tts_ab_//')"
  grep 'host_cgroup max_MB=' "$f" | tail -1
done
