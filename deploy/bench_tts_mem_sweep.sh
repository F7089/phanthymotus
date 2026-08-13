#!/usr/bin/env bash
# Sweep ORT CUDA memory knobs for FP32 Melo on Jetson (keep RTF, cut peak).
# GPT-suggested order:
#   A baseline
#   B cudnn_conv_use_max_workspace=0
#   C + kSameAsRequested
#   D/E/F + gpu_mem_limit 512/384/320
#
#   bash deploy/bench_tts_mem_sweep.sh
#   IMAGE=phanthymotus-perception-tts:927c9c6-jp61 bash deploy/bench_tts_mem_sweep.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-phanthymotus-perception-tts:927c9c6-jp61}"
NAME="${NAME:-phanthymotus-tts-melo}"
OUT="${OUT:-$HOME/fanyi/wav_out}"
mkdir -p "$OUT"
SUMMARY="$OUT/mem_sweep_summary.tsv"
echo -e "label\tavg_rtf\tcgroup_ready_peak_mib\tcgroup_infer_peak_mib\tcgroup_infer_current_mib" >"$SUMMARY"

run_one() {
  local label="$1"
  shift
  local args=()
  local e
  for e in "$@"; do
    args+=(--ort-env "$e")
  done
  echo "======== $label ======== ${args[*]-}"
  python3 deploy/bench_tts_peak_mem.py \
    --restart \
    --image "$IMAGE" \
    --container "$NAME" \
    --label "$label" \
    --warmup 1 \
    --runs 3 \
    --out-dir "$OUT" \
    "${args[@]}"
  python3 - <<PY
import json
from pathlib import Path
p = Path("$OUT") / "peak_mem_report_${label}.json"
d = json.loads(p.read_text())
row = [
    d.get("label"),
    f"{d.get('avg_rtf'):.4f}" if d.get("avg_rtf") is not None else "",
    str((d.get("cgroup_after_ready_mib") or {}).get("peak")),
    str((d.get("cgroup_after_infer_mib") or {}).get("peak")),
    str((d.get("cgroup_after_infer_mib") or {}).get("current")),
]
print("\t".join(row))
with open("$SUMMARY", "a") as f:
    f.write("\t".join(row) + "\n")
PY
}

# A: current defaults (max workspace=1)
run_one A

# B: cap cuDNN workspace
run_one B TTS_ORT_CUDNN_MAX_WORKSPACE=0

# C: B + arena extend
run_one C TTS_ORT_CUDNN_MAX_WORKSPACE=0 TTS_ORT_ARENA_EXTEND=kSameAsRequested

# D/E/F: add gpu_mem_limit ladder
run_one D TTS_ORT_CUDNN_MAX_WORKSPACE=0 TTS_ORT_ARENA_EXTEND=kSameAsRequested TTS_ORT_GPU_MEM_LIMIT_MB=512
run_one E TTS_ORT_CUDNN_MAX_WORKSPACE=0 TTS_ORT_ARENA_EXTEND=kSameAsRequested TTS_ORT_GPU_MEM_LIMIT_MB=384
run_one F TTS_ORT_CUDNN_MAX_WORKSPACE=0 TTS_ORT_ARENA_EXTEND=kSameAsRequested TTS_ORT_GPU_MEM_LIMIT_MB=320

echo
echo "=== sweep summary ==="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "wrote $SUMMARY"
