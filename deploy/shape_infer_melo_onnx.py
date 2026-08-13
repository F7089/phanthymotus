#!/usr/bin/env python3
"""Run ORT-recommended symbolic shape inference for TensorRT EP.

Usage (Jetson container or 74):
  python3 shape_infer_melo_onnx.py \\
    --input /models/.../model.onnx \\
    --output /tmp/model_shaped.onnx
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--auto-merge", action="store_true", default=True)
    args = ap.parse_args()

    inp = os.path.abspath(args.input)
    out = os.path.abspath(args.output)
    if not os.path.isfile(inp):
        raise SystemExit(f"missing input: {inp}")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    try:
        import onnx
    except ImportError as e:
        raise SystemExit(
            "need package 'onnx' for shape inference "
            f"(pip install onnx). import error: {e}"
        ) from e

    print(f"load {inp} ({os.path.getsize(inp)/1024/1024:.1f} MiB)")
    model = onnx.load(inp)

    # Prefer ORT symbolic shape infer (what TRT EP docs recommend).
    try:
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        print("using onnxruntime.tools.symbolic_shape_infer (auto_merge)")
        inferred = SymbolicShapeInference.infer_shapes(
            model, auto_merge=bool(args.auto_merge)
        )
    except Exception as e:
        print(f"symbolic_shape_infer failed ({e}); fallback onnx.shape_inference")
        from onnx import shape_inference

        inferred = shape_inference.infer_shapes(model)

    # Basic check: any ValueInfo still missing shape?
    missing = 0
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
        t = vi.type.tensor_type
        if not t.HasField("shape"):
            missing += 1
    print(f"value_info/output missing shape count≈{missing}")

    onnx.save(inferred, out)
    print(f"wrote {out} ({os.path.getsize(out)/1024/1024:.1f} MiB)")


if __name__ == "__main__":
    main()
    sys.exit(0)
