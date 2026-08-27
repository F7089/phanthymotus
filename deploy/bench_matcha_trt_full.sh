#!/usr/bin/env bash
# Full ranking-equivalent stack peak: ROS + MCP + WeText + lexicon + TRT infer.
# Same entrypoint/main.py as judgeflow TTS, opt-in TTS_MATCHA_TRT=1 (does not
# change the default ranking image). Peak = host cgroup memory.max_usage_in_bytes.
#
# Default engine is --tacticSources=-CUDNN,-JIT_CONVOLUTIONS. That plan has
# failed execute_v2 on JP5; the script then reruns with the working cmpf32
# engine unless TTS_TRT_NO_FALLBACK=1.
#
#   IMAGE=phanthymotus-perception-tts:e19aee0
#   bash deploy/bench_matcha_trt_full.sh "$IMAGE"
set -euo pipefail

IMAGE="${1:?image required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIVE="${TTS_LIVE_CONTAINER:-phanthymotus-perception-tts-0}"
NAME="${TTS_FULL_NAME:-phanthymotus-tts-trt-full}"
MODEL_DIR="${TTS_MODEL_DIR:-/models/matcha-kai-16k-e500}"
HOST_MODEL="${TTS_HOST_MODEL:-/tmp/matcha-kai-16k-e500-ckpt}"
WETEXT_HOST="${TTS_WETEXT_HOST:-/tmp/wetext-ckpt}"
MATCHA_CACHE_HOST="${TTS_MATCHA_TRT_CACHE_HOST:-/tmp/matcha_trt_cache}"
VOCOS_CACHE_HOST="${TTS_VOCOS_TRT_CACHE_HOST:-/tmp/vocos_trt_cache}"
MCP_PORT="${MCP_PORT:-16720}"
WS_PORT="${WS_PORT:-16721}"
EXTRA_TEXT="${TTS_FULL_TEXT:-今天是2024年3月1日，你好，我是陆风。}"

if [[ ! -d "$ROOT/perception" ]]; then
  echo "FATAL: missing $ROOT/perception" >&2
  exit 1
fi

ensure_dir_from_live() {
  local host="$1" ctr_path="$2" marker="$3"
  if [[ -f "$host/$marker" ]]; then
    return 0
  fi
  if docker inspect "$LIVE" >/dev/null 2>&1 && \
     docker exec "$LIVE" test -f "$ctr_path/$marker"; then
    echo "[full] docker cp $LIVE:$ctr_path -> $host"
    rm -rf "$host"
    docker cp "$LIVE:$ctr_path" "$host"
  fi
  if [[ ! -f "$host/$marker" ]]; then
    echo "FATAL: missing $host/$marker" >&2
    exit 1
  fi
}

ensure_dir_from_live "$HOST_MODEL" "$MODEL_DIR" "tokens.txt"
ensure_dir_from_live "$WETEXT_HOST" "/models/wetext" "zh_tn_tagger.fst"

TACTICS_HOST=$(ls -1t "$MATCHA_CACHE_HOST"/model-steps-3.trt8.5*.cudnn--jit_convolutions.engine 2>/dev/null | head -1 || true)
CMPF_HOST=$(ls -1t "$MATCHA_CACHE_HOST"/model-steps-3.trt8.5*.cmpf32.engine 2>/dev/null | head -1 || true)
VOCOS_HOST=$(ls -1 "$VOCOS_CACHE_HOST"/vocos-16khz-univ*.engine 2>/dev/null | head -1 || true)
if [[ -z "$VOCOS_HOST" ]]; then
  echo "FATAL: no Vocos engine in $VOCOS_CACHE_HOST (run bench_matcha_trt_mem.sh first)" >&2
  exit 1
fi

print_cgroup() {
  local cid cg
  cid=$(docker inspect -f '{{.Id}}' "$NAME" 2>/dev/null || true)
  [[ -n "$cid" ]] || return 0
  cg="/sys/fs/cgroup/memory/docker/$cid"
  if [[ -f "$cg/memory.max_usage_in_bytes" ]]; then
    echo "----- host cgroup -----"
    awk '{printf "host_cgroup usage_MB=%.1f\n", $1/1024/1024}' "$cg/memory.usage_in_bytes"
    awk '{printf "host_cgroup max_MB=%.1f\n", $1/1024/1024}' "$cg/memory.max_usage_in_bytes"
  fi
}

run_one() {
  local engine_host="$1" label="$2"
  local engine_ctr="/opt/matcha_trt_cache/$(basename "$engine_host")"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  echo "========== fullstack $label =========="
  echo "engine $engine_host"
  echo "wetext $WETEXT_HOST"
  echo "work   bind-mount tts.py + matcha_trt.py + vocos_trt.py"
  docker run -d --name "$NAME" \
    --runtime nvidia --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e MCP_PORT="$MCP_PORT" \
    -e WS_PORT="$WS_PORT" \
    -e TTS_DISABLE_TRT="${TTS_DISABLE_TRT:-1}" \
    -e TTS_VOCOS_TRT=0 \
    -e TTS_MATCHA_TRT=1 \
    -e TTS_MATCHA_TRT_ENGINE="$engine_ctr" \
    -e TTS_MATCHA_TRT_CACHE=/opt/matcha_trt_cache \
    -e TTS_VOCOS_TRT_CACHE=/opt/vocos_trt_cache \
    -e TTS_VOCOS_TRT_ENGINE="/opt/vocos_trt_cache/$(basename "$VOCOS_HOST")" \
    -e TTS_DUMP_CGROUP=1 \
    -v "$MATCHA_CACHE_HOST":/opt/matcha_trt_cache \
    -v "$VOCOS_CACHE_HOST":/opt/vocos_trt_cache \
    -v "$ROOT/perception/plugins/tts.py":/work/plugins/tts.py:ro \
    -v "$ROOT/perception/utils/matcha_trt.py":/work/utils/matcha_trt.py:ro \
    -v "$ROOT/perception/utils/vocos_trt.py":/work/utils/vocos_trt.py:ro \
    -v "$HOST_MODEL:$MODEL_DIR:ro" \
    -v "$WETEXT_HOST":/models/wetext:ro \
    "$IMAGE"

  local i infer_ok=0 infer_fail=0
  for i in $(seq 1 120); do
    if docker logs "$NAME" 2>&1 | grep -q 'FULLSTACK_PEAK tag=after_warmup'; then
      infer_ok=1
      break
    fi
    if docker logs "$NAME" 2>&1 | grep -q 'FULLSTACK_INFER_FAILED'; then
      infer_fail=1
      break
    fi
    if docker logs "$NAME" 2>&1 | grep -qE 'failed to load model|Traceback'; then
      if ! docker ps -q --filter "name=^/${NAME}$" | grep -q .; then
        break
      fi
    fi
    if ! docker ps -q --filter "name=^/${NAME}$" | grep -q .; then
      break
    fi
    sleep 2
  done

  echo "----- logs $label -----"
  docker logs "$NAME" 2>&1 | grep -E 'FULLSTACK_|Matcha TRT|wetext=|warmup |nvinfer_plugin|execute_v2|Traceback|Error|failed to load|plugin init' || docker logs --tail 80 "$NAME"

  if [[ "$infer_ok" == "1" ]]; then
    echo "[full] warmup infer ok, wait MCP then extra WeText generate"
    for i in $(seq 1 30); do
      if docker logs "$NAME" 2>&1 | grep -q 'MCP server'; then
        break
      fi
      sleep 1
    done
    docker exec \
      -e EXTRA_TEXT="$EXTRA_TEXT" \
      -e MCP_PORT="$MCP_PORT" \
      "$NAME" python3 -c '
import json, os, urllib.request
body = json.dumps({"text": os.environ.get("EXTRA_TEXT", "")}, ensure_ascii=False).encode()
req = urllib.request.Request(
    "http://127.0.0.1:%s/tts/test" % os.environ.get("MCP_PORT", "16720"),
    data=body,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    obj = json.loads(raw.decode())
    print("EXTRA_TTS ok=%s info=%s wav=%s" % (
        obj.get("ok"),
        obj.get("info", "")[:200],
        "yes" if obj.get("wav_b64") else "no",
    ), flush=True)
except Exception as e:
    print("EXTRA_TTS_FAILED", e, flush=True)
' || true
    sleep 2
    docker exec -w /work -e PYTHONPATH=/work "$NAME" python3 -c \
      'from utils.matcha_trt import dump_fullstack_peak; dump_fullstack_peak("after_extra_wetext")' || true
  fi

  print_cgroup
  if [[ "$infer_ok" == "1" ]]; then
    echo "FULLSTACK_INFER=ok label=$label"
    return 0
  fi
  echo "FULLSTACK_INFER=failed label=$label"
  return 1
}

echo "Peak is host_cgroup max_MB after warmup + extra WeText generate."
echo "This is ROS+MCP+WeText+lexicon+TRT in the ranking main.py process."

ok=1
if [[ -n "${TTS_MATCHA_TRT_ENGINE:-}" ]]; then
  run_one "${TTS_MATCHA_TRT_ENGINE}" "env-engine" || ok=0
elif [[ -n "$TACTICS_HOST" ]]; then
  if run_one "$TACTICS_HOST" "tacticSources=-CUDNN,-JIT_CONVOLUTIONS"; then
    ok=1
  else
    ok=0
    if [[ "${TTS_TRT_NO_FALLBACK:-0}" != "1" && -n "$CMPF_HOST" ]]; then
      echo "[full] tacticSources engine cannot infer; rerun working cmpf32 in a fresh container"
      run_one "$CMPF_HOST" "cmpf32-working" || ok=0
    elif [[ -z "$CMPF_HOST" ]]; then
      echo "FATAL: no cmpf32 engine to fall back to" >&2
    fi
  fi
elif [[ -n "$CMPF_HOST" ]]; then
  echo "[full] no tacticSources engine on disk; using cmpf32"
  run_one "$CMPF_HOST" "cmpf32-working" || ok=0
else
  echo "FATAL: no Matcha engine in $MATCHA_CACHE_HOST" >&2
  exit 1
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
if [[ "$ok" != "1" ]]; then
  echo "fullstack infer did not complete" >&2
  exit 1
fi
echo "Done. Compare host_cgroup max_MB to the ranking container max (same metric)."
