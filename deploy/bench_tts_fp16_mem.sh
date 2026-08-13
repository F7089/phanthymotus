#!/usr/bin/env bash
# Compare FP32 vs FP16 Melo on Jetson: RTF + cgroup current + smaps heap/anon.
# Prerequisite: JuiceFS has vits-melo-longanlingxin-openepd-nobert-44100-fp16.tar.bz2
#
#   bash deploy/bench_tts_fp16_mem.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-phanthymotus-perception-tts:927c9c6-jp61}"
NAME="${NAME:-phanthymotus-tts-melo}"
OUT="${OUT:-$HOME/fanyi/wav_out}"
mkdir -p "$OUT"
SUMMARY="$OUT/fp16_mem_summary.tsv"
echo -e "label\tavg_rtf\tcgroup_current_mib\theap_mib\tanon_mib" >"$SUMMARY"

patch_config() {
  local dtype="$1"  # fp32|fp16
  local tmp
  tmp="$(mktemp)"
  python3 - <<PY
from pathlib import Path
src = Path("perception/config.yaml")
text = src.read_text(encoding="utf-8")
# rewrite tts model_name / model_dir lines only
out = []
for line in text.splitlines(True):
    if line.strip().startswith("model_name:") and "tts_melo_openepd" in line:
        out.append(f"    model_name: tts_melo_openepd_${dtype}\n")
    elif line.strip().startswith("model_dir:") and "vits-melo-longanlingxin-openepd" in line:
        out.append(f"    model_dir: /models/vits-melo-longanlingxin-openepd-nobert-44100-${dtype}\n")
    else:
        out.append(line)
Path("$tmp").write_text("".join(out), encoding="utf-8")
print("wrote", "$tmp")
PY
  echo "$tmp"
}

run_one() {
  local label="$1"
  local dtype="$2"
  local cfg
  cfg="$(patch_config "$dtype")"
  # temporarily point mount at patched config via symlink in /tmp
  local mount_cfg="$OUT/config_${dtype}.yaml"
  cp -f "$cfg" "$mount_cfg"
  rm -f "$cfg"

  echo "======== $label ($dtype) ========"
  # Override restart mounts: copy patched config into place after run via env hack —
  # reuse bench script but replace host config mount by swapping perception/config.yaml briefly.
  cp -f perception/config.yaml "$OUT/config.yaml.bak"
  cp -f "$mount_cfg" perception/config.yaml
  python3 deploy/bench_tts_peak_mem.py \
    --restart \
    --image "$IMAGE" \
    --container "$NAME" \
    --label "$label" \
    --warmup 1 \
    --runs 3 \
    --out-dir "$OUT" || true
  mv -f "$OUT/config.yaml.bak" perception/config.yaml

  docker cp deploy/smaps_rss.py "$NAME":/tmp/smaps_rss.py
  docker exec -u 0 "$NAME" python3 /tmp/smaps_rss.py 1 | tee "$OUT/smaps_${label}.txt" >/dev/null
  python3 - <<PY
import json, re
from pathlib import Path
d = json.loads(Path("$OUT/peak_mem_report_${label}.json").read_text())
cur = (d.get("cgroup_after_infer_mib") or {}).get("current")
text = Path("$OUT/smaps_${label}.txt").read_text()
heap = anon = ""
for line in text.splitlines():
    if "MiB Rss |" in line and "| [heap]" in line:
        heap = line.split()[0]
    if line.strip().endswith("| [anon]") and "aggregated" not in line:
        # aggregated line: "   754.9 MiB Rss | ..."
        pass
for line in text.splitlines():
    if "MiB Rss |" in line and line.rstrip().endswith("| [anon]") and "Anon |" in line:
        # skip per-mapping; use aggregated section
        pass
agg_heap = agg_anon = None
in_agg = False
for line in text.splitlines():
    if "=== aggregated by name" in line:
        in_agg = True
        continue
    if in_agg and line.startswith("==="):
        break
    if in_agg and "| [heap]" in line:
        agg_heap = line.split()[0]
    if in_agg and line.rstrip().endswith("| [anon]"):
        agg_anon = line.split()[0]
row = [d.get("label"), f"{d.get('avg_rtf'):.4f}", str(cur), str(agg_heap), str(agg_anon)]
print("\t".join(row))
with open("$SUMMARY", "a") as f:
    f.write("\t".join(row) + "\n")
PY
}

run_one FP32 fp32
run_one FP16 fp16

echo
echo "=== FP32 vs FP16 ==="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
