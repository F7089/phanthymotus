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

write_config() {
  local dtype="$1"
  local dest="$2"
  python3 - "$dtype" "$dest" <<'PY'
import sys
from pathlib import Path
dtype, dest = sys.argv[1], Path(sys.argv[2])
src = Path("perception/config.yaml")
out = []
for line in src.read_text(encoding="utf-8").splitlines(True):
    if line.strip().startswith("model_name:") and "tts_melo_openepd" in line:
        out.append(f"    model_name: tts_melo_openepd_{dtype}\n")
    elif line.strip().startswith("model_dir:") and "vits-melo-longanlingxin-openepd" in line:
        out.append(
            f"    model_dir: /models/vits-melo-longanlingxin-openepd-nobert-44100-{dtype}\n"
        )
    else:
        out.append(line)
dest.write_text("".join(out), encoding="utf-8")
PY
}

run_one() {
  local label="$1"
  local dtype="$2"
  local mount_cfg="$OUT/config_${dtype}.yaml"
  write_config "$dtype" "$mount_cfg"

  echo "======== $label ($dtype) ========"
  cp -f perception/config.yaml "$OUT/config.yaml.bak"
  cp -f "$mount_cfg" perception/config.yaml

  set +e
  python3 deploy/bench_tts_peak_mem.py \
    --restart \
    --image "$IMAGE" \
    --container "$NAME" \
    --label "$label" \
    --warmup 1 \
    --runs 3 \
    --out-dir "$OUT"
  local rc=$?
  set -e
  mv -f "$OUT/config.yaml.bak" perception/config.yaml
  if [[ $rc -ne 0 ]]; then
    echo "WARN: bench $label failed rc=$rc" >&2
    return 0
  fi

  docker cp deploy/smaps_rss.py "$NAME":/tmp/smaps_rss.py
  docker exec -u 0 "$NAME" python3 /tmp/smaps_rss.py 1 | tee "$OUT/smaps_${label}.txt" >/dev/null

  python3 - "$OUT" "$label" "$SUMMARY" <<'PY'
import json, sys
from pathlib import Path
out, label, summary = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
d = json.loads((out / f"peak_mem_report_{label}.json").read_text())
cur = (d.get("cgroup_after_infer_mib") or {}).get("current")
text = (out / f"smaps_{label}.txt").read_text()
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
row = [
    str(d.get("label")),
    f"{d.get('avg_rtf'):.4f}" if d.get("avg_rtf") is not None else "",
    str(cur),
    str(agg_heap),
    str(agg_anon),
]
print("\t".join(row))
with summary.open("a") as f:
    f.write("\t".join(row) + "\n")
PY
}

run_one FP32 fp32
run_one FP16 fp16

echo
echo "=== FP32 vs FP16 ==="
column -t -s $'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
