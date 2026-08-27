#!/usr/bin/env bash
# Full-stack peak with judgeflow-equivalent docker flags.
# Peak = host cgroup memory.max_usage_in_bytes.
#
# Matches deploy/judgeflow_tts_run.sh:
#   --runtime nvidia --network host --privileged
#   NVIDIA_VISIBLE_DEVICES / NVIDIA_DRIVER_CAPABILITIES
#   ROS_DOMAIN_ID / FASTDDS_BUILTIN_TRANSPORTS if set on the host
#   cmpf32 Matcha engine + Vocos engine (no tacticSources / preview)
#
#   IMAGE=phanthymotus-perception-tts:e19aee0
#   bash deploy/bench_matcha_trt_full.sh "$IMAGE"
set -euo pipefail

IMAGE="${1:?image required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${TTS_FULL_NAME:-phanthymotus-tts-trt-full}"
MODEL_DIR="${TTS_MODEL_DIR:-/models/matcha-kai-16k-e500}"
MATCHA_CACHE_HOST="${TTS_MATCHA_TRT_CACHE_HOST:-}"
VOCOS_CACHE_HOST="${TTS_VOCOS_TRT_CACHE_HOST:-}"
MCP_PORT="${MCP_PORT:-16720}"
WS_PORT="${WS_PORT:-16721}"
EXTRA_TEXT="${TTS_FULL_TEXT:-今天是2024年3月1日，你好，我是陆风。}"
LABEL="${TTS_BENCH_LABEL:-cmpf32-host}"

pick_host_dir() {
  local d
  for d in "$@"; do
    if [[ -n "$d" && -d "$d" ]]; then
      printf '%s' "$d"
      return 0
    fi
  done
  return 1
}

if [[ -z "$MATCHA_CACHE_HOST" ]]; then
  MATCHA_CACHE_HOST="$(pick_host_dir /models/matcha_trt_cache /tmp/matcha_trt_cache || true)"
  MATCHA_CACHE_HOST="${MATCHA_CACHE_HOST:-/tmp/matcha_trt_cache}"
fi
if [[ -z "$VOCOS_CACHE_HOST" ]]; then
  VOCOS_CACHE_HOST="$(pick_host_dir /models/vocos_trt_cache /tmp/vocos_trt_cache || true)"
  VOCOS_CACHE_HOST="${VOCOS_CACHE_HOST:-/tmp/vocos_trt_cache}"
fi

if [[ ! -d "$ROOT/perception" ]]; then
  echo "FATAL: missing $ROOT/perception" >&2
  exit 1
fi

CMPF_HOST="${TTS_MATCHA_TRT_ENGINE_HOST:-}"
if [[ -z "$CMPF_HOST" ]]; then
  CMPF_HOST=$(ls -1t "$MATCHA_CACHE_HOST"/model-steps-3.trt8.5*.cmpf32.engine 2>/dev/null | head -1 || true)
fi
VOCOS_HOST="${TTS_VOCOS_TRT_ENGINE_HOST:-}"
if [[ -z "$VOCOS_HOST" ]]; then
  VOCOS_HOST=$(ls -1t "$VOCOS_CACHE_HOST"/vocos-16khz-univ*.engine 2>/dev/null | head -1 || true)
fi
if [[ -z "$CMPF_HOST" || ! -f "$CMPF_HOST" ]]; then
  echo "FATAL: no cmpf32 engine in $MATCHA_CACHE_HOST" >&2
  exit 1
fi
if [[ -z "$VOCOS_HOST" || ! -f "$VOCOS_HOST" ]]; then
  echo "FATAL: no Vocos engine in $VOCOS_CACHE_HOST" >&2
  exit 1
fi

container_running() {
  docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true
}

print_cgroup() {
  local cid cg
  cid=$(docker inspect -f '{{.Id}}' "$NAME" 2>/dev/null || true)
  echo "[full] container=$NAME id=${cid:0:12} label=$LABEL"
  [[ -n "$cid" ]] || return 0
  for cg in \
    "/sys/fs/cgroup/memory/docker/$cid" \
    "/sys/fs/cgroup/memory/system.slice/docker-${cid}.scope"; do
    if [[ -f "$cg/memory.max_usage_in_bytes" ]]; then
      echo "----- host cgroup -----"
      awk '{printf "host_cgroup usage_MB=%.1f\n", $1/1024/1024}' "$cg/memory.usage_in_bytes"
      awk '{printf "host_cgroup max_MB=%.1f\n", $1/1024/1024}' "$cg/memory.max_usage_in_bytes"
      return 0
    fi
  done
  echo "[full] host cgroup file not found for $cid"
}

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "========== fullstack $LABEL =========="
echo "engine $CMPF_HOST"
echo "vocos  $VOCOS_HOST"
echo "flags  --network host (judgeflow-equivalent)"

RUN_ARGS=(
  docker run -d
  --name "$NAME"
  --runtime nvidia
  --network host
  --privileged
  -e NVIDIA_VISIBLE_DEVICES=all
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
  -e MCP_PORT="$MCP_PORT"
  -e WS_PORT="$WS_PORT"
  -e TTS_DISABLE_TRT="${TTS_DISABLE_TRT:-1}"
  -e TTS_VOCOS_TRT=0
  -e TTS_MATCHA_TRT=1
  -e TTS_MATCHA_TRT_ENGINE="/opt/matcha_trt_cache/$(basename "$CMPF_HOST")"
  -e TTS_MATCHA_TRT_CACHE=/opt/matcha_trt_cache
  -e TTS_VOCOS_TRT_CACHE=/opt/vocos_trt_cache
  -e TTS_VOCOS_TRT_ENGINE="/opt/vocos_trt_cache/$(basename "$VOCOS_HOST")"
  -e TTS_TRT_PREFER_TACTICS=0
  -e TTS_DUMP_CGROUP=1
  -e TTS_RANKING_MODE="${TTS_RANKING_MODE:-0}"
  -e TTS_ISTFT="${TTS_ISTFT:-numpy}"
  -e TTS_VOCOS_PARSE_ONNX="${TTS_VOCOS_PARSE_ONNX:-0}"
  -e TTS_TRT_LOAD_ORDER="${TTS_TRT_LOAD_ORDER:-trt_first}"
  -e TTS_TRT_BUF_REUSE="${TTS_TRT_BUF_REUSE:-1}"
  -v "$MATCHA_CACHE_HOST":/opt/matcha_trt_cache
  -v "$VOCOS_CACHE_HOST":/opt/vocos_trt_cache
  -v "$ROOT/perception/plugins/tts.py":/work/plugins/tts.py:ro
  -v "$ROOT/perception/utils/matcha_trt.py":/work/utils/matcha_trt.py:ro
  -v "$ROOT/perception/utils/vocos_trt.py":/work/utils/vocos_trt.py:ro
  -v "$ROOT/perception/main.py":/work/main.py:ro
  -v "$ROOT/perception/deploy/entrypoint.sh":/deploy/entrypoint.sh:ro
)

if [ -d /models ]; then
  RUN_ARGS+=(-v /models:/models)
fi
if [ -n "${ROS_DOMAIN_ID:-}" ]; then
  RUN_ARGS+=(-e "ROS_DOMAIN_ID=${ROS_DOMAIN_ID}")
fi
if [ -n "${FASTDDS_BUILTIN_TRANSPORTS:-}" ]; then
  RUN_ARGS+=(-e "FASTDDS_BUILTIN_TRANSPORTS=${FASTDDS_BUILTIN_TRANSPORTS}")
fi
RUN_ARGS+=("$IMAGE")

echo "[full] ${RUN_ARGS[*]}"
"${RUN_ARGS[@]}"

infer_ok=0
infer_fail=0
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
    if ! container_running; then
      break
    fi
  fi
  if ! container_running; then
    break
  fi
  sleep 2
done

echo "----- logs $LABEL -----"
docker logs "$NAME" 2>&1 | grep -E 'FULLSTACK_|Matcha TRT|wetext=|warmup |nvinfer_plugin|execute_v2|skip WebSocket|TTS_RANKING_MODE|TensorRT ok|skip sherpa|Traceback|Error|failed to load|istft=' || docker logs --tail 80 "$NAME"

ok=0
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
import json, os, time, urllib.request
text = os.environ.get("EXTRA_TEXT", "")
body = json.dumps({"text": text}, ensure_ascii=False).encode()
url = "http://127.0.0.1:%s/tts/test" % os.environ.get("MCP_PORT", "16720")
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
t0 = time.monotonic()
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = r.read()
    wall = time.monotonic() - t0
    obj = json.loads(raw.decode())
    wav_len = 0
    if obj.get("wav_b64"):
        import base64
        wav_len = len(base64.b64decode(obj["wav_b64"]))
    audio_s = max(0.0, (wav_len - 44) / 2.0 / 16000.0) if wav_len else 0.0
    rtf = (wall / audio_s) if audio_s > 0 else -1
    print("EXTRA_TTS ok=%s wall_s=%.3f audio_s=%.3f synth_RTF=%.4f wav=%s" % (
        obj.get("ok"), wall, audio_s, rtf, "yes" if obj.get("wav_b64") else "no",
    ), flush=True)
except Exception as e:
    print("EXTRA_TTS_FAILED", e, flush=True)
' || true
  sleep 2
  docker exec -w /work -e PYTHONPATH=/work "$NAME" python3 -c \
    'from utils.matcha_trt import dump_fullstack_peak; dump_fullstack_peak("after_extra_wetext")' || true
  ok=1
fi

print_cgroup
if [[ "$ok" == "1" ]]; then
  echo "FULLSTACK_INFER=ok label=$LABEL"
else
  echo "FULLSTACK_INFER=failed label=$LABEL" >&2
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  exit 1
fi

if [[ "${TTS_BENCH_KEEP:-0}" != "1" ]]; then
  docker rm -f "$NAME" >/dev/null 2>&1 || true
fi
echo "Done. Peak is host_cgroup max_MB (judgeflow-equivalent --network host)."
