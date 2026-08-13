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
from pathlib import Path


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
        kwargs = {"auto_merge": bool(args.auto_merge)}
        # Newer ORT accepts guess_output_rank; ignore if unsupported.
        try:
            inferred = SymbolicShapeInference.infer_shapes(
                model, guess_output_rank=True, **kwargs
            )
        except TypeError:
            inferred = SymbolicShapeInference.infer_shapes(model, **kwargs)
    except Exception as e:
        print(f"symbolic_shape_infer failed ({e}); fallback onnx.shape_inference")
        from onnx import shape_inference

        try:
            inferred = shape_inference.infer_shapes(model, data_prop=True)
        except TypeError:
            inferred = shape_inference.infer_shapes(model)

    def _shape_str(vi) -> str:
        t = vi.type.tensor_type
        if not t.HasField("shape"):
            return "<no shape>"
        dims = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(str(d.dim_value))
            elif d.HasField("dim_param"):
                dims.append(d.dim_param)
            else:
                dims.append("?")
        return "[" + ",".join(dims) + "]"

    by_name = {vi.name: vi for vi in inferred.graph.value_info}
    by_name.update({o.name: o for o in inferred.graph.output})
    target = "/ConstantOfShape_output_0"
    if target in by_name:
        print(f"check {target}: {_shape_str(by_name[target])}")
    else:
        # search any ConstantOfShape outputs
        cos = [n for n in inferred.graph.node if n.op_type == "ConstantOfShape"]
        print(f"ConstantOfShape nodes={len(cos)}")
        for n in cos[:8]:
            for o in n.output:
                vi = by_name.get(o)
                print(f"  {o}: {_shape_str(vi) if vi else '<not in value_info>'}")

    missing = 0
    for vi in list(inferred.graph.value_info) + list(inferred.graph.output):
        t = vi.type.tensor_type
        if not t.HasField("shape"):
            missing += 1
    print(f"value_info/output missing shape count≈{missing}")

    onnx.save(inferred, out)
    print(f"wrote {out} ({os.path.getsize(out)/1024/1024:.1f} MiB)")
    Path(out + ".ok").write_text(
        f"source={inp}\nmissing_shapes≈{missing}\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
