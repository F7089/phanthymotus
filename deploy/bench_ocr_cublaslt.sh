#!/usr/bin/env bash
# OCR JP5 tactic A/B. Does not overwrite cmpf32. Does not change judgeflow.
#
# PPT rec is CUBLAS-only (not "keep CUDNN"). First run was only -CUBLAS_LT;
# that is a different experiment. Mode cublas_only is the PPT replay:
#   --tacticSources=-CUBLAS_LT,-CUDNN
#
#   IMAGE=phanthymotus-perception-tts:e19aee0
#   bash deploy/bench_ocr_cublaslt.sh "$IMAGE" cublas_only
#
# Other modes: nocublaslt | cublas_strict (also drop EDGE_MASK_CONVOLUTIONS)
set -euo pipefail

IMAGE="${1:?image required}"
MODE="${2:-${OCR_MODE:-cublas_only}}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIVE="${TTS_LIVE_CONTAINER:-phanthymotus-perception-tts-0}"
MODEL_DIR="${TTS_MODEL_DIR:-/models/matcha-kai-16k-e500}"
HOST_MODEL="${TTS_HOST_MODEL:-/tmp/matcha-kai-16k-e500-ckpt}"
MATCHA_CACHE_HOST="${TTS_MATCHA_TRT_CACHE_HOST:-/tmp/matcha_trt_cache}"
VOCOS_CACHE_HOST="${TTS_VOCOS_TRT_CACHE_HOST:-/tmp/vocos_trt_cache}"
NAME="${TTS_CKPT_NAME:-phanthymotus-tts-ckpt}"
WS="${TTS_TRT_WORKSPACE_MB:-256}"
export TTS_TRT_WORKSPACE_MB="$WS"

case "$MODE" in
  nocublaslt)
    TACTICS="-CUBLAS_LT"
    TAG="nocublaslt"
    ;;
  cublas_only)
    TACTICS="-CUBLAS_LT,-CUDNN"
    TAG="cublas_only"
    ;;
  cublas_strict)
    TACTICS="-CUBLAS_LT,-CUDNN,-EDGE_MASK_CONVOLUTIONS"
    TAG="cublas_strict"
    ;;
  *)
    echo "FATAL: mode must be nocublaslt | cublas_only | cublas_strict (got $MODE)" >&2
    exit 2
    ;;
esac

VOCOS_ENG_NAME="vocos-16khz-univ.trt8.5.fp16.ws64.${TAG}.engine"
BUILD_LOG="${TTS_VOCOS_TRT_BUILD_LOG:-/tmp/vocos_trt_build_${TAG}.log}"
MEASURE_LOG="/tmp/ocr_${TAG}_measure.log"
MATCHA_HOST="$MATCHA_CACHE_HOST/model-steps-3.trt8.5.fp16.ws${WS}.L256.mel2000.${TAG}.engine"

mkdir -p "$MATCHA_CACHE_HOST" "$VOCOS_CACHE_HOST"

echo "======== OCR tactic mode=$MODE --tacticSources=$TACTICS ========"
echo "workspace=${WS}MB  tag=$TAG"
echo "PPT rec = CUBLAS-only. nocublaslt keeps CUDNN; that is not the PPT case."
echo

TTS_TRT_TACTICS="$TACTICS" \
  TTS_TRT_ENG_TAG="$TAG" \
  TTS_TRT_BUILD_LOG="/tmp/matcha_trt_build_${TAG}.log" \
  TTS_TRT_WORKSPACE_MB="$WS" \
  bash "$ROOT/deploy/bench_matcha_trt_mem.sh" "$IMAGE" | tee "$MEASURE_LOG"

if [[ ! -f "$MATCHA_HOST" ]]; then
  echo "FATAL: missing $MATCHA_HOST" >&2
  exit 1
fi

if ! grep -q 'warmup_ok' "$MEASURE_LOG"; then
  echo
  echo "STOP: $TAG Matcha did not warmup_ok (divUp / execute_v2)."
  echo "PPT CUBLAS-only does not run on this Matcha graph if CUDNN is required."
  echo "Do not fullstack. Paste warmup_/divUp from $MEASURE_LOG."
  exit 1
fi

ensure_host_vocos_onnx() {
  if [[ -f "$HOST_MODEL/vocos-16khz-univ.onnx" ]]; then
    return
  fi
  if docker inspect "$LIVE" >/dev/null 2>&1 && \
     docker exec "$LIVE" test -f "$MODEL_DIR/vocos-16khz-univ.onnx"; then
    echo "[ckpt] docker cp $LIVE:$MODEL_DIR -> $HOST_MODEL"
    rm -rf "$HOST_MODEL"
    docker cp "$LIVE:$MODEL_DIR" "$HOST_MODEL"
    return
  fi
  if [[ -f /models/matcha-kai-16k-e500/vocos-16khz-univ.onnx ]]; then
    HOST_MODEL=/models/matcha-kai-16k-e500
    return
  fi
  echo "FATAL: no vocos-16khz-univ.onnx" >&2
  exit 1
}

echo
echo "======== Vocos --tacticSources=$TACTICS ========"
if [[ -f "$VOCOS_CACHE_HOST/$VOCOS_ENG_NAME" ]]; then
  echo "[build] reuse $VOCOS_CACHE_HOST/$VOCOS_ENG_NAME"
  ls -lah "$VOCOS_CACHE_HOST/$VOCOS_ENG_NAME"
else
  ensure_host_vocos_onnx
  docker rm -f "$NAME-vocos-build" >/dev/null 2>&1 || true
  echo "[build] vocos $TAG log -> $BUILD_LOG"
  docker run --rm --name "$NAME-vocos-build" \
    --runtime nvidia --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -v "$HOST_MODEL:$MODEL_DIR:ro" \
    -v "$VOCOS_CACHE_HOST":/opt/vocos_trt_cache \
    --entrypoint bash \
    "$IMAGE" \
    -lc '
set -e
INP=$(python3 -c "import onnx; print(onnx.load(\"'"$MODEL_DIR"'/vocos-16khz-univ.onnx\").graph.input[0].name)")
echo "vocos_input=$INP"
exe=/usr/src/tensorrt/bin/trtexec
test -x "$exe" || exe=/usr/bin/trtexec
"$exe" --onnx='"$MODEL_DIR"'/vocos-16khz-univ.onnx \
  --saveEngine=/opt/vocos_trt_cache/'"$VOCOS_ENG_NAME"'.tmp \
  --fp16 --workspace=64 \
  --tacticSources='"$TACTICS"' \
  --minShapes=${INP}:1x80x16 --optShapes=${INP}:1x80x256 --maxShapes=${INP}:1x80x2000
mv -f /opt/vocos_trt_cache/'"$VOCOS_ENG_NAME"'.tmp \
      /opt/vocos_trt_cache/'"$VOCOS_ENG_NAME"'
ls -lah /opt/vocos_trt_cache/'"$VOCOS_ENG_NAME"'
' > "$BUILD_LOG" 2>&1
  echo "----- vocos trtexec summary -----"
  grep -E '\[E\]|&&&& |PASSED|FAILED|tacticSources' "$BUILD_LOG" | grep -v 'onnx2trt_utils.cpp:403' || true
  if [[ ! -f "$VOCOS_CACHE_HOST/$VOCOS_ENG_NAME" ]]; then
    echo "FATAL: vocos $TAG build failed. log $BUILD_LOG" >&2
    exit 1
  fi
  echo "[build] ok $VOCOS_CACHE_HOST/$VOCOS_ENG_NAME"
fi

echo
echo "======== fullstack $TAG (judgeflow-equivalent) ========"
TTS_BENCH_LABEL="ocr-$TAG" \
  TTS_MATCHA_TRT_ENGINE_HOST="$MATCHA_HOST" \
  TTS_VOCOS_TRT_ENGINE_HOST="$VOCOS_CACHE_HOST/$VOCOS_ENG_NAME" \
  bash "$ROOT/deploy/bench_matcha_trt_full.sh" "$IMAGE"

echo
echo "Compare: default ~1345 | -CUBLAS_LT ~1269 | this $TAG ???"
echo "Need: warmup_ok, EXTRA_TTS ok=True, host_cgroup max_MB."
echo "If execute fails: Matcha needs CUDNN; PPT CUBLAS-only does not apply."
