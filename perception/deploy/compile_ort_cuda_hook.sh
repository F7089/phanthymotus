#!/bin/bash
# Retired: Matcha Vocos now uses TensorRT Runtime (utils/vocos_trt.py),
# not an ORT EP LD_PRELOAD hook. Dockerfile no longer builds this .so.
echo "compile on image build only: see perception/Dockerfile.jetson" >&2
exit 1
