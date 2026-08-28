#!/usr/bin/env bash
# OCR JP5 recipe on Matcha/Vocos: disable CUBLAS_LT, keep CUBLAS + CUDNN.
#
# OCR rec: cuBLAS_LT 1528MiB -> CUBLAS-only 699MiB. Cause is JP5 CUDA 11.4
# having no module lazy loading; a CUBLAS_LT tactic keeps the whole library
# resident. We never tried this flag. Previous attempts were different:
#   --tacticSources=-CUDNN          -> execute divUp n>0, peak unchanged
#   preview disableExternal...0805  -> same crash, peak unchanged
#
# Does not overwrite cmpf32 / default Vocos engines. Does not change judgeflow.
#
#   IMAGE=phanthymotus-perception-tts:e19aee0
#   bash deploy/bench_ocr_cublaslt.sh "$IMAGE"
set -euo pipefail

IMAGE="${1:?image required}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIVE="${TTS_LIVE_CONTAINER:-phanthymotus-perception-tts-0}"
MODEL_DIR="${TTS_MODEL_DIR:-/models/matcha-kai-16k-e500}"
HOST_MODEL="${TTS_HOST_MODEL:-/tmp/matcha-kai-16k-e500-ckpt}"
MATCHA_CACHE_HOST="${TTS_MATCHA_TRT_CACHE_HOST:-/tmp/matcha_trt_cache}"
VOCOS_CACHE_HOST="${TTS_VOCOS_TRT_CACHE_HOST:-/tmp/vocos_trt_cache}"
VOCOS_ENG_NAME="vocos-16khz-univ.trt8.5.fp16.ws64.nocublaslt.engine"
BUILD_LOG="${TTS_VOCOS_TRT_BUILD_LOG:-/tmp/vocos_trt_build_nocublaslt.log}"
NAME="${TTS_CKPT_NAME:-phanthymotus-tts-ckpt}"

mkdir -p "$MATCHA_CACHE_HOST" "$VOCOS_CACHE_HOST"

echo "======== OCR-style tactic: --tacticSources=-CUBLAS_LT ========"
echo "Keep CUBLAS and CUDNN. Do not use -CUDNN / preview."
echo "If CUDA 11.8 docker build is still running, stop it first (GPU+RAM)."
echo

# Acoustic: same patched ONNX / profiles as cmpf32, only CUBLAS_LT off.
MEASURE_LOG=/tmp/ocr_cublaslt_measure.log
TTS_TRT_TACTICS=-CUBLAS_LT \
  TTS_TRT_ENG_TAG=nocublaslt \
  TTS_TRT_BUILD_LOG=/tmp/matcha_trt_build_nocublaslt.log \
  bash "$ROOT/deploy/bench_matcha_trt_mem.sh" "$IMAGE" | tee "$MEASURE_LOG"

MATCHA_HOST="$MATCHA_CACHE_HOST/model-steps-3.trt8.5.fp16.ws4096.L256.mel2000.nocublaslt.engine"
if [[ ! -f "$MATCHA_HOST" ]]; then
  echo "FATAL: missing $MATCHA_HOST" >&2
  exit 1
fi

if ! grep -q 'warmup_ok' "$MEASURE_LOG"; then
  echo
  echo "STOP: nocublaslt Matcha engine did not warmup_ok."
  echo "Do not fullstack. Paste warmup_/divUp/host_cgroup from $MEASURE_LOG."
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
echo "======== Vocos engine with --tacticSources=-CUBLAS_LT ========"
if [[ -f "$VOCOS_CACHE_HOST/$VOCOS_ENG_NAME" ]]; then
  echo "[build] reuse $VOCOS_CACHE_HOST/$VOCOS_ENG_NAME"
  ls -lah "$VOCOS_CACHE_HOST/$VOCOS_ENG_NAME"
else
  ensure_host_vocos_onnx
  docker rm -f "$NAME-vocos-build" >/dev/null 2>&1 || true
  echo "[build] vocos nocublaslt log -> $BUILD_LOG"
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
  --tacticSources=-CUBLAS_LT \
  --minShapes=${INP}:1x80x16 --optShapes=${INP}:1x80x256 --maxShapes=${INP}:1x80x2000
mv -f /opt/vocos_trt_cache/'"$VOCOS_ENG_NAME"'.tmp \
      /opt/vocos_trt_cache/'"$VOCOS_ENG_NAME"'
ls -lah /opt/vocos_trt_cache/'"$VOCOS_ENG_NAME"'
' > "$BUILD_LOG" 2>&1
  echo "----- vocos trtexec summary -----"
  grep -E '\[E\]|&&&& |PASSED|FAILED|tacticSources' "$BUILD_LOG" | grep -v 'onnx2trt_utils.cpp:403' || true
  if [[ ! -f "$VOCOS_CACHE_HOST/$VOCOS_ENG_NAME" ]]; then
    echo "FATAL: vocos nocublaslt build failed. log $BUILD_LOG" >&2
    exit 1
  fi
  echo "[build] ok $VOCOS_CACHE_HOST/$VOCOS_ENG_NAME"
fi

echo
echo "======== fullstack nocublaslt (judgeflow-equivalent) ========"
TTS_BENCH_LABEL=ocr-nocublaslt \
  TTS_MATCHA_TRT_ENGINE_HOST="$MATCHA_HOST" \
  TTS_VOCOS_TRT_ENGINE_HOST="$VOCOS_CACHE_HOST/$VOCOS_ENG_NAME" \
  bash "$ROOT/deploy/bench_matcha_trt_full.sh" "$IMAGE"

echo
echo "Compare to default-tactic fullstack ~1345MB / sherpa ~1374MB."
echo "Need: warmup_ok, EXTRA_TTS ok=True, host_cgroup max_MB."
echo "If max still ~1.3GB, CUBLAS_LT is not the Matcha tax; stop, do not change judgeflow."
