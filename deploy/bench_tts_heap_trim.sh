#!/usr/bin/env bash
# Cheap heap-side A/B before FP16 (GPT G/H/I).
# NOTE: production Melo already has enable_cpu_mem_arena=False (= G baseline).
#
#   bash deploy/bench_tts_heap_trim.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-phanthymotus-perception-tts:927c9c6-jp61}"
NAME="${NAME:-phanthymotus-tts-melo}"
OUT="${OUT:-$HOME/fanyi/wav_out}"
mkdir -p "$OUT"
SUMMARY="$OUT/heap_trim_summary.tsv"
echo -e "label\tavg_rtf\tcgroup_current_mib\tnote" >"$SUMMARY"

run_one() {
  local label="$1"
  shift
  echo "======== $label ========"
  python3 deploy/bench_tts_peak_mem.py \
    --restart \
    --image "$IMAGE" \
    --container "$NAME" \
    --label "$label" \
    --warmup 1 \
    --runs 3 \
    --out-dir "$OUT" \
    "$@"
  # smaps heap/anon snapshot
  docker cp deploy/smaps_rss.py "$NAME":/tmp/smaps_rss.py
  docker exec -u 0 "$NAME" python3 /tmp/smaps_rss.py 1 \
    | tee "$OUT/smaps_${label}.txt" \
    | grep -E '^\s+[0-9.]+ MiB Rss \|.*\[heap\]|aggregated|^\s+[0-9.]+ MiB Rss \|.*\[anon\]|sum Anon|Anonymous:' \
    || true
  python3 - <<PY
import json
from pathlib import Path
p = Path("$OUT") / "peak_mem_report_${label}.json"
d = json.loads(p.read_text())
cur = (d.get("cgroup_after_infer_mib") or {}).get("current")
row = [d.get("label"), f"{d.get('avg_rtf'):.4f}", str(cur), "${label}"]
print("\t".join(row))
with open("$SUMMARY", "a") as f:
    f.write("\t".join(row) + "\n")
PY
}

# G: already default (CPU arena off). Explicit for clarity.
run_one G \
  --ort-env TTS_ORT_CPU_ARENA=0 \
  --ort-env TTS_ORT_MEM_PATTERN=1

# H: also disable mem_pattern
run_one H \
  --ort-env TTS_ORT_CPU_ARENA=0 \
  --ort-env TTS_ORT_MEM_PATTERN=0

# I: G + malloc_trim after load/warmup
run_one I \
  --ort-env TTS_ORT_CPU_ARENA=0 \
  --ort-env TTS_ORT_MEM_PATTERN=1 \
  --ort-env TTS_MALLOC_TRIM=1

# H+I
run_one HI \
  --ort-env TTS_ORT_CPU_ARENA=0 \
  --ort-env TTS_ORT_MEM_PATTERN=0 \
  --ort-env TTS_MALLOC_TRIM=1

echo
echo "=== heap/trim summary ==="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo "smaps: $OUT/smaps_*.txt"
echo "look for [tts] malloc_trim in: docker logs $NAME"
