#!/usr/bin/env bash
# Real wav via Matcha TRT + Vocos TRT. Reuses engines, no trtexec, no sherpa CUDA.
#
#   IMAGE=phanthymotus-perception-tts:e19aee0
#   bash deploy/bench_matcha_trt_wav.sh "$IMAGE"
set -euo pipefail

IMAGE="${1:?image required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY_WAV="$ROOT/perception/deploy/bench_matcha_trt_wav.py"
PY_MEM="$ROOT/perception/deploy/bench_matcha_trt_mem.py"
NAME="${TTS_CKPT_NAME:-phanthymotus-tts-ckpt}"
LIVE="${TTS_LIVE_CONTAINER:-phanthymotus-perception-tts-0}"
MODEL_DIR="${TTS_MODEL_DIR:-/models/matcha-kai-16k-e500}"
HOST_MODEL="${TTS_HOST_MODEL:-/tmp/matcha-kai-16k-e500-ckpt}"
MATCHA_CACHE_HOST="${TTS_MATCHA_TRT_CACHE_HOST:-/tmp/matcha_trt_cache}"
VOCOS_CACHE_HOST="${TTS_VOCOS_TRT_CACHE_HOST:-/tmp/vocos_trt_cache}"
TEXT="${TTS_TRT_TEXT:-你好，我是陆风。}"
OUT_HOST="${TTS_TRT_WAV:-/tmp/matcha_trt_lufeng.wav}"

ACOUSTIC_HOST=$(ls -1t "$MATCHA_CACHE_HOST"/model-steps-3.trt8.5*.engine 2>/dev/null | head -1 || true)
VOCOS_HOST=$(ls -1 "$VOCOS_CACHE_HOST"/vocos-16khz-univ*.engine 2>/dev/null | head -1 || true)
if [[ -z "$ACOUSTIC_HOST" || -z "$VOCOS_HOST" ]]; then
  echo "FATAL: need Matcha+Vocos engines. Run bench_matcha_trt_mem.sh first." >&2
  echo "matcha_cache=$MATCHA_CACHE_HOST vocos_cache=$VOCOS_CACHE_HOST" >&2
  exit 1
fi

VOLUME_ARGS=(
  -v "$MATCHA_CACHE_HOST":/opt/matcha_trt_cache
  -v "$VOCOS_CACHE_HOST":/opt/vocos_trt_cache
  -v "$PY_WAV":/deploy/bench_matcha_trt_wav.py:ro
  -v "$PY_MEM":/deploy/bench_matcha_trt_mem.py:ro
)
if [[ -f "$HOST_MODEL/tokens.txt" ]]; then
  VOLUME_ARGS+=(-v "$HOST_MODEL:$MODEL_DIR:ro")
elif docker inspect "$LIVE" >/dev/null 2>&1 && \
     docker exec "$LIVE" test -f "$MODEL_DIR/tokens.txt"; then
  if [[ ! -f "$HOST_MODEL/tokens.txt" ]]; then
    echo "[ckpt] docker cp $LIVE:$MODEL_DIR -> $HOST_MODEL"
    rm -rf "$HOST_MODEL"
    docker cp "$LIVE:$MODEL_DIR" "$HOST_MODEL"
  fi
  VOLUME_ARGS+=(-v "$HOST_MODEL:$MODEL_DIR:ro")
else
  echo "FATAL: no tokens.txt" >&2
  exit 1
fi

ACOUSTIC_ENG="/opt/matcha_trt_cache/$(basename "$ACOUSTIC_HOST")"
VOCOS_ENG="/opt/vocos_trt_cache/$(basename "$VOCOS_HOST")"
OUT_CTR="/opt/matcha_trt_cache/hello_lufeng.wav"

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "acoustic $ACOUSTIC_HOST"
echo "vocos    $VOCOS_HOST"
echo "text     $TEXT"
docker run -d --name "$NAME" \
  --runtime nvidia --privileged \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e TTS_TRT_MAX_MEL="${TTS_TRT_MAX_MEL:-2000}" \
  "${VOLUME_ARGS[@]}" \
  --entrypoint python3 \
  "$IMAGE" \
  /deploy/bench_matcha_trt_wav.py \
    --acoustic "$ACOUSTIC_ENG" \
    --vocos "$VOCOS_ENG" \
    --model-dir "$MODEL_DIR" \
    --out "$OUT_CTR" \
    --text "$TEXT"

for _ in $(seq 1 90); do
  if docker logs "$NAME" 2>&1 | grep -qE 'CKPT_DONE|Traceback|too few tokens'; then
    break
  fi
  if ! docker ps -q --filter "name=^/${NAME}$" | grep -q .; then
    break
  fi
  sleep 2
done

echo "----- logs -----"
docker logs "$NAME" 2>&1 | grep -E 'CKPT |CKPT_DONE|imported |deserialize|wrote_wav|n_tokens|mel_raw|vocos_s|text |phones |pad tokens|Traceback|Error|plugin_lib' || docker logs --tail 80 "$NAME"

CID=$(docker inspect -f '{{.Id}}' "$NAME")
CG="/sys/fs/cgroup/memory/docker/$CID"
if [[ -f "$CG/memory.usage_in_bytes" ]]; then
  echo "----- host cgroup -----"
  awk '{printf "host_cgroup usage_MB=%.1f\n", $1/1024/1024}' "$CG/memory.usage_in_bytes"
  awk '{printf "host_cgroup max_MB=%.1f\n", $1/1024/1024}' "$CG/memory.max_usage_in_bytes"
fi

if docker exec "$NAME" test -f "$OUT_CTR"; then
  docker cp "$NAME:$OUT_CTR" "$OUT_HOST"
  ls -lah "$OUT_HOST"
  echo "wav -> $OUT_HOST"
else
  echo "FATAL: no wav in container" >&2
fi
docker rm -f "$NAME" >/dev/null
echo "Peak is host_cgroup max_MB. Stages are CKPT tags A..I."
