#!/usr/bin/env bash
# Apply git slim G2P into container, then cold RSS + peak benches.
# Handles bind-mounted / busy /work/*.py (docker cp unlink fails).
set -euo pipefail

NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT_HOST:-$HOME/fanyi/wav_out}"
mkdir -p "$OUT"

# Install one file into container without relying on docker-cp unlink.
# Strategy: docker cp → /tmp, then `cat > dst` (truncate-write, no unlink).
_install_file() {
  local src="$1" dst="$2"
  local base tmp
  base="$(basename "$dst")"
  tmp="/tmp/_hotpatch_${base}"
  echo "  install $src -> $dst"
  docker cp "$src" "${NAME}:${tmp}"
  docker exec -u 0 "$NAME" bash -lc "
    set -e
    mkdir -p \"\$(dirname '$dst')\"
    cat '$tmp' > '$dst'
    rm -f '$tmp'
    ls -lah '$dst'
  "
}

echo "== show mounts (busy often = single-file bind mount) =="
docker inspect -f '{{range .Mounts}}{{.Type}} {{.Source}} -> {{.Destination}}{{println}}{{end}}' "$NAME" || true

echo "== stop $NAME =="
docker stop "$NAME" >/dev/null || true
# ensure fully stopped
for _ in $(seq 1 30); do
  st=$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo missing)
  [[ "$st" == "exited" || "$st" == "created" || "$st" == "missing" ]] && break
  sleep 0.5
done

echo "== start (needed for exec install) =="
docker start "$NAME" >/dev/null
sleep 2

echo "== install slim + plugins via /tmp + cat =="
docker exec -u 0 "$NAME" mkdir -p /work/melo_g2p_slim /work/plugins /work/utils
# directory: cp works to /tmp then rsync-like
docker cp "$ROOT/perception/melo_g2p_slim/." "${NAME}:/tmp/_melo_g2p_slim/"
docker exec -u 0 "$NAME" bash -lc '
  set -e
  mkdir -p /work/melo_g2p_slim
  cp -a /tmp/_melo_g2p_slim/. /work/melo_g2p_slim/
  ls -lah /work/melo_g2p_slim
'
_install_file "$ROOT/perception/plugins/tts.py" "/work/plugins/tts.py"
_install_file "$ROOT/perception/utils/model_downloader.py" "/work/utils/model_downloader.py"

TEXT="/models/melo-openepd-g2p-assets/vendor/melo_g2p/text"
echo "== overlay vendor text =="
docker exec -u 0 "$NAME" mkdir -p "$TEXT"
_install_file "$ROOT/perception/melo_g2p_slim/english.py" "$TEXT/english.py"
_install_file "$ROOT/perception/melo_g2p_slim/slim_g2p_oov.py" "$TEXT/slim_g2p_oov.py"
_install_file "$ROOT/perception/melo_g2p_slim/openepd_compact.py" "$TEXT/openepd_compact.py"

echo "== ensure OOV ckpt from JuiceFS =="
docker exec -u 0 -w /work "$NAME" python3 - <<'PY'
import os
import sys
sys.path.insert(0, "/work")
from utils.model_downloader import ensure_model

text = "/models/melo-openepd-g2p-assets/vendor/melo_g2p/text"
ensure_model("tts_melo_g2p_oov_ckpt", text)
print("ckpt", os.path.join(text, "checkpoint20.npz"),
      "exists", os.path.isfile(os.path.join(text, "checkpoint20.npz")))
pkl = "/models/melo-openepd-g2p-assets/openepd_eng_dict.pickle"
print("pickle", pkl, "exists", os.path.isfile(pkl))
PY

echo "== restart so TTS process reloads =="
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

echo "== peak mem (no re-create: --restart would wipe /work hotpatch; we just restarted) =="
python3 "$ROOT/deploy/bench_tts_peak_mem.py" --no-restart --runs 3 --label after_slim \
  | tee "$OUT/peak_mem_after_slim.txt"

echo "done. reports under $OUT"
