#!/usr/bin/env bash
# Isolated Matcha native baseline on Jetson. Never mutates the Melo container.
#
# Usage (on Jetson, after copying this repo):
#   ./deploy/matcha_native/run_jetson.sh              # download + probe + bench-or-instruct
#   ./deploy/matcha_native/run_jetson.sh --download    # models only
#   ./deploy/matcha_native/run_jetson.sh --probe       # CUDA probe only
#   ./deploy/matcha_native/run_jetson.sh --bench       # require CUDA sherpa-onnx
#   ./deploy/matcha_native/run_jetson.sh --build-image # Dockerfile.jetson.matcha
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NATIVE="$(cd "$(dirname "$0")" && pwd)"
MELO_CONTAINER="${MELO_CONTAINER:-phanthymotus-tts-melo}"
MATCHA_IMAGE="${MATCHA_IMAGE:-phanthymotus-matcha-native:jp61}"
MATCHA_CONTAINER="${MATCHA_CONTAINER:-phanthymotus-tts-matcha}"
DEST="${MATCHA_MODEL_DIR:-$HOME/fanyi/matcha_native}"
WAV_OUT="${MATCHA_WAV_OUT:-$HOME/fanyi/wav_out/matcha_native}"
ACTION="${1:-all}"

download() {
  mkdir -p "$DEST" "$WAV_OUT"
  python3 "$NATIVE/download_models.py" --dest "$DEST"
}

probe_local() {
  python3 "$NATIVE/probe_sherpa_cuda.py"
}

probe_melo_readonly() {
  if ! docker inspect "$MELO_CONTAINER" >/dev/null 2>&1; then
    echo "[probe] Melo container $MELO_CONTAINER not found; skip read-only probe"
    return 2
  fi
  echo "[probe] read-only probe inside $MELO_CONTAINER (no pip, no restart)"
  docker start "$MELO_CONTAINER" >/dev/null 2>&1 || true
  docker cp "$NATIVE/probe_sherpa_cuda.py" "$MELO_CONTAINER:/tmp/probe_sherpa_cuda.py"
  docker exec -u 0 "$MELO_CONTAINER" python3 /tmp/probe_sherpa_cuda.py
}

build_image() {
  echo "[build] isolated Matcha CUDA image; Melo image is not modified"
  docker build \
    -f "$ROOT/perception/Dockerfile.jetson.matcha" \
    --build-arg JP_VERSION=61 \
    --network=host \
    -t "$MATCHA_IMAGE" \
    "$ROOT"
}

run_isolated_bench() {
  if ! docker image inspect "$MATCHA_IMAGE" >/dev/null 2>&1; then
    echo "[bench] image $MATCHA_IMAGE missing. Build with:"
    echo "  $0 --build-image"
    return 2
  fi
  docker rm -f "$MATCHA_CONTAINER" >/dev/null 2>&1 || true
  docker run --rm \
    --name "$MATCHA_CONTAINER" \
    --runtime nvidia \
    --network host \
    --privileged \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e MATCHA_MODEL_DIR=/models/matcha_native \
    -e MATCHA_WAV_OUT=/wav_out \
    -e MATCHA_PROVIDER=cuda \
    -e MATCHA_REQUIRE_CUDA=1 \
    -v "$DEST:/models/matcha_native" \
    -v "$WAV_OUT:/wav_out" \
    "$MATCHA_IMAGE"
}

case "$ACTION" in
  --download|download) download ;;
  --probe|probe)
    probe_melo_readonly || true
    echo "----- local python -----"
    probe_local || true
    ;;
  --build-image|--build) build_image ;;
  --bench|bench)
    download
    run_isolated_bench
    ;;
  all|"")
    download
    echo "===== probe Melo container (read-only; do not install) ====="
    if probe_melo_readonly; then
      echo "[note] Melo's sherpa-onnx reports CUDA. Still prefer isolated image so Melo env stays untouched."
    else
      echo "[note] Melo/local sherpa-onnx is not a CUDA build (expected for pip sherpa-onnx)."
      echo "       Do NOT pip install / rebuild inside $MELO_CONTAINER."
      echo "       Next: $0 --build-image && $0 --bench"
    fi
    if docker image inspect "$MATCHA_IMAGE" >/dev/null 2>&1; then
      run_isolated_bench
    else
      echo
      echo "Models are at: $DEST"
      echo "Isolated CUDA image not built yet. On Jetson:"
      echo "  $0 --build-image"
      echo "  $0 --bench"
    fi
    ;;
  *)
    echo "unknown action: $ACTION" >&2
    exit 2
    ;;
esac
