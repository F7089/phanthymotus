#!/usr/bin/env bash
# Stop container → copy slim code from git → start → cold peak + RSS benches.
# Avoids "device or resource busy" on docker cp of running /work/*.py.
set -euo pipefail

NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT_HOST:-$HOME/fanyi/wav_out}"
mkdir -p "$OUT"

echo "== stop $NAME =="
docker stop "$NAME" >/dev/null

echo "== docker cp slim + plugins (container stopped) =="
docker cp "$ROOT/perception/melo_g2p_slim" "$NAME:/work/melo_g2p_slim"
docker cp "$ROOT/perception/plugins/tts.py" "$NAME:/work/plugins/tts.py"
docker cp "$ROOT/perception/utils/model_downloader.py" "$NAME:/work/utils/model_downloader.py"

# Overlay into downloaded G2P vendor (so benches that import vendor see slim)
TEXT="/models/melo-openepd-g2p-assets/vendor/melo_g2p/text"
docker start "$NAME" >/dev/null
# brief wait for docker daemon; then copy into models while running is OK for new files
sleep 2
docker cp "$ROOT/perception/melo_g2p_slim/english.py" "$NAME:$TEXT/english.py"
docker cp "$ROOT/perception/melo_g2p_slim/slim_g2p_oov.py" "$NAME:$TEXT/slim_g2p_oov.py"
docker cp "$ROOT/perception/melo_g2p_slim/openepd_compact.py" "$NAME:$TEXT/openepd_compact.py"

echo "== ensure OOV ckpt from JuiceFS (oedb built locally from pickle on first G2P load) =="
docker exec -u 0 -w /work "$NAME" python3 - <<'PY'
import os
from utils.model_downloader import ensure_model

text = "/models/melo-openepd-g2p-assets/vendor/melo_g2p/text"
ensure_model("tts_melo_g2p_oov_ckpt", text)
print("ckpt", os.path.join(text, "checkpoint20.npz"),
      "exists", os.path.isfile(os.path.join(text, "checkpoint20.npz")))
pkl = "/models/melo-openepd-g2p-assets/openepd_eng_dict.pickle"
print("pickle", pkl, "exists", os.path.isfile(pkl))
PY

echo "== restart so TTS process reloads overlay =="
docker restart "$NAME" >/dev/null
echo "waiting MCP..."
for i in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:15730/tts/test" -H 'Content-Type: application/json' \
      -d '{"text":"hi"}' >/dev/null 2>&1; then
    echo "MCP ready (${i}s)"
    break
  fi
  sleep 2
done

echo "== RSS stages =="
bash "$ROOT/deploy/bench_tts_rss_stages.sh" | tee "$OUT/rss_stages.txt"

echo "== cold peak mem =="
python3 "$ROOT/deploy/bench_tts_peak_mem.py" --restart --runs 3 --label cold_slim \
  | tee "$OUT/peak_mem_cold.txt"

echo "done. reports under $OUT"
