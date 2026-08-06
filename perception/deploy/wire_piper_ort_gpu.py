#!/usr/bin/env python3
"""Overlay sherpa-onnx GPU ORT .so into pip onnxruntime/capi (Jetson aarch64).

PyPI onnxruntime for aarch64 is CPU-only. Sherpa's JP6 build already installs
CUDA12 libonnxruntime*.so under site-packages/sherpa_onnx/lib/. Copy those into
onnxruntime/capi so Piper's InferenceSession can use CUDAExecutionProvider.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys


def main() -> int:
    import onnxruntime as ort
    import sherpa_onnx

    sherpa_lib = os.path.join(os.path.dirname(sherpa_onnx.__file__), "lib")
    capi = os.path.join(os.path.dirname(ort.__file__), "capi")
    if not os.path.isdir(sherpa_lib):
        print(f"missing sherpa ORT lib dir: {sherpa_lib}", file=sys.stderr)
        return 1
    if not os.path.isdir(capi):
        print(f"missing onnxruntime capi dir: {capi}", file=sys.stderr)
        return 1

    copied: list[str] = []
    for pattern in ("libonnxruntime.so*", "libonnxruntime_providers_*.so"):
        for src in glob.glob(os.path.join(sherpa_lib, pattern)):
            dst = os.path.join(capi, os.path.basename(src))
            shutil.copy2(src, dst)
            copied.append(os.path.basename(src))

    if not any("providers_cuda" in n for n in copied):
        print(
            f"CUDA provider lib not copied from {sherpa_lib}; got {copied}",
            file=sys.stderr,
        )
        return 1

    conf = "/etc/ld.so.conf.d/sherpa-onnx-ort.conf"
    try:
        with open(conf, "w", encoding="utf-8") as f:
            f.write(sherpa_lib + "\n")
        print("wrote", conf)
    except OSError as e:
        print(f"WARN: could not write {conf}: {e}", file=sys.stderr)

    print("overlaid GPU ORT libs into", capi, ":", sorted(copied))
    print("onnxruntime", ort.__version__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
