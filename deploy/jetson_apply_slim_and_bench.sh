#!/usr/bin/env bash
# Apply git slim G2P + bench. Host bind-mounts for tts.py / model_downloader.py
# mean those files are already the git checkout — do NOT docker-cp into them.
set -euo pipefail

NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT_HOST:-$HOME/fanyi/wav_out}"
mkdir -p "$OUT"

echo "== mounts =="
docker inspect -f '{{range .Mounts}}{{.Type}} {{.Source}} -> {{.Destination}}{{println}}{{end}}' "$NAME" || true

echo "== git bind-mounts already provide (no container rewrite) =="
echo "  $ROOT/perception/plugins/tts.py"
echo "  $ROOT/perception/utils/model_downloader.py"
ls -lah "$ROOT/perception/plugins/tts.py" "$ROOT/perception/utils/model_downloader.py"

echo "== install /work/melo_g2p_slim (not bind-mounted) =="
docker start "$NAME" >/dev/null 2>&1 || true
sleep 1
docker exec -u 0 "$NAME" rm -rf /work/melo_g2p_slim /tmp/_melo_g2p_slim
docker cp "$ROOT/perception/melo_g2p_slim" "${NAME}:/tmp/_melo_g2p_slim"
docker exec -u 0 "$NAME" bash -lc '
  set -e
  rm -rf /work/melo_g2p_slim
  mkdir -p /work/melo_g2p_slim
  # docker cp of a dir may nest; normalize
  if [[ -f /tmp/_melo_g2p_slim/english.py ]]; then
    cp -a /tmp/_melo_g2p_slim/. /work/melo_g2p_slim/
  else
    cp -a /tmp/_melo_g2p_slim/melo_g2p_slim/. /work/melo_g2p_slim/
  fi
  ls -lah /work/melo_g2p_slim
'

TEXT="/models/melo-openepd-g2p-assets/vendor/melo_g2p/text"
echo "== overlay vendor text under /models (writable) =="
docker exec -u 0 "$NAME" mkdir -p "$TEXT"
for f in english.py slim_g2p_oov.py openepd_compact.py; do
  docker cp "$ROOT/perception/melo_g2p_slim/$f" "$NAME:/tmp/_hotpatch_$f"
  docker exec -u 0 "$NAME" bash -lc "cat /tmp/_hotpatch_$f > '$TEXT/$f' && ls -lah '$TEXT/$f'"
done

echo "== ensure OOV ckpt from JuiceFS =="
docker exec -u 0 -w /work "$NAME" python3 - <<'PY'
import os, sys
sys.path.insert(0, "/work")
from utils.model_downloader import ensure_model
text = "/models/melo-openepd-g2p-assets/vendor/melo_g2p/text"
ensure_model("tts_melo_g2p_oov_ckpt", text)
print("ckpt ok", os.path.isfile(os.path.join(text, "checkpoint20.npz")))
print("pickle", os.path.isfile("/models/melo-openepd-g2p-assets/openepd_eng_dict.pickle"))
PY

echo "== restart to reload Python (bind-mounted tts.py from git) =="
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

echo "== peak mem (keep container; bind mounts already current) =="
python3 "$ROOT/deploy/bench_tts_peak_mem.py" --no-restart --runs 3 --label after_slim \
  | tee "$OUT/peak_mem_after_slim.txt"

echo "done. reports under $OUT"
