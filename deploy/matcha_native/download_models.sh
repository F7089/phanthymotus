#!/usr/bin/env bash
# Wrapper: official matcha-icefall-zh-en + vocos-16khz-univ (COS, GitHub fallback).
set -euo pipefail
exec python3 "$(cd "$(dirname "$0")" && pwd)/download_models.py" "$@"
