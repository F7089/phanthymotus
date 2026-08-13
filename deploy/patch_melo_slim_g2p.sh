#!/usr/bin/env bash
# Hot-patch Jetson container: OpenEPD + slim g2p OOV (no g2p_en/nltk import).
set -euo pipefail
NAME="${TTS_CONTAINER:-phanthymotus-tts-melo}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SLIM_DIR="${SCRIPT_DIR}/melo_g2p_slim"
G2P_TEXT="/models/melo-openepd-g2p-assets/vendor/melo_g2p/text"

[[ -f "${SLIM_DIR}/slim_g2p_oov.py" ]] || { echo "missing ${SLIM_DIR}/slim_g2p_oov.py"; exit 1; }
[[ -f "${SLIM_DIR}/english.py" ]] || { echo "missing ${SLIM_DIR}/english.py"; exit 1; }

docker cp "${SLIM_DIR}/slim_g2p_oov.py" "${NAME}:${G2P_TEXT}/slim_g2p_oov.py"
docker cp "${SLIM_DIR}/english.py" "${NAME}:${G2P_TEXT}/english.py"

if [[ -f "${SLIM_DIR}/checkpoint20.npz" ]]; then
  docker cp "${SLIM_DIR}/checkpoint20.npz" "${NAME}:${G2P_TEXT}/checkpoint20.npz"
else
  echo "checkpoint20.npz not in repo; copy from container g2p_en package..."
  docker exec -u 0 "${NAME}" python3 - <<PY
import os, shutil
import g2p_en
src = os.path.join(os.path.dirname(g2p_en.__file__), "checkpoint20.npz")
dst = "${G2P_TEXT}/checkpoint20.npz"
shutil.copy2(src, dst)
print("copied", src, "->", dst, "bytes", os.path.getsize(dst))
PY
fi

docker exec -u 0 "${NAME}" bash -lc "
set -e
grep -E 'g2p_en|G2p\\(\\)' ${G2P_TEXT}/english.py && exit 1 || echo 'OK: english.py has no g2p_en'
ls -lah ${G2P_TEXT}/slim_g2p_oov.py ${G2P_TEXT}/checkpoint20.npz ${G2P_TEXT}/english.py
python3 - <<'PY'
import os, sys
os.environ['MELO_OPENEPD_DICT']='/models/melo-openepd-g2p-assets/openepd_eng_dict.pickle'
os.environ['MELO_SKIP_HF_TOKENIZER']='1'
sys.path.insert(0,'/models/melo-openepd-g2p-assets/vendor')
# Import english alone (should NOT pull nltk)
import melo_g2p.text.english as en
assert 'nltk' not in sys.modules, 'nltk was imported'
assert 'g2p_en' not in sys.modules, 'g2p_en was imported'
print('import english OK; nltk/g2p_en absent')
print('oov sample', en._g2p_oov('activationist'))
print('nltk after oov', 'nltk' in sys.modules)
PY
"

echo
echo "Verify RSS:"
echo "  bash deploy/bench_tts_host_imports.sh"
echo "  bash deploy/bench_tts_g2p_imports.sh"
