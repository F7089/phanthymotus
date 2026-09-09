#!/usr/bin/env python3
"""Merge Matcha + BigVGAN ONNX into one graph: x/tones/languages/scales -> wav.

The merged .onnx is ~128MB and must stay in the model package / JuiceFS,
not in git. Ranking still runs without it (two CUDA sessions).
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcha", required=True)
    parser.add_argument("--bigvgan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import onnx
    from onnx.compose import merge_models
    from onnx import version_converter

    matcha = onnx.load(args.matcha)
    voc = onnx.load(args.bigvgan)
    matcha_ops = {oi.domain: oi.version for oi in matcha.opset_import}
    voc_ops = {oi.domain: oi.version for oi in voc.opset_import}
    default_m = matcha_ops.get("", matcha_ops.get("ai.onnx", 15))
    default_v = voc_ops.get("", voc_ops.get("ai.onnx", 17))
    if default_m != default_v:
        target = max(default_m, default_v)
        print("OPSET_ALIGN %s -> %s" % (default_m, target), flush=True)
        try:
            if default_m < target:
                matcha = version_converter.convert_version(matcha, target)
            if default_v < target:
                voc = version_converter.convert_version(voc, target)
        except Exception as exc:
            print("VERSION_CONVERTER_FAIL %s; bumping opset field only" % exc, flush=True)
            for model, want in ((matcha, target), (voc, target)):
                found = False
                for oi in model.opset_import:
                    if oi.domain in ("", "ai.onnx"):
                        oi.version = want
                        found = True
                if not found:
                    model.opset_import.extend([onnx.helper.make_opsetid("", want)])
    print("MATCHA ir=%s opsets=%s inputs=%s outputs=%s"
          % (matcha.ir_version,
             [(o.domain, o.version) for o in matcha.opset_import],
             [i.name for i in matcha.graph.input],
             [o.name for o in matcha.graph.output]), flush=True)
    print("BIGVGAN ir=%s opsets=%s inputs=%s outputs=%s"
          % (voc.ir_version,
             [(o.domain, o.version) for o in voc.opset_import],
             [i.name for i in voc.graph.input],
             [o.name for o in voc.graph.output]), flush=True)

    # Keep Matcha input names unprefixed. Prefix only BigVGAN to avoid collisions.
    merged = merge_models(
        matcha,
        voc,
        io_map=[("mel", "mels")],
        prefix1="",
        prefix2="v/",
    )
    in_names = [i.name for i in merged.graph.input]
    out_names = [o.name for o in merged.graph.output]
    print("MERGED ir=%s inputs=%s outputs=%s nodes=%s inits=%s"
          % (merged.ir_version, in_names, out_names,
             len(merged.graph.node), len(merged.graph.initializer)), flush=True)
    if not any(name.endswith("wav") for name in out_names):
        raise RuntimeError("merged graph missing wav output: %s" % out_names)
    onnx.checker.check_model(merged, full_check=False)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    onnx.save(merged, args.out)
    print("SAVED %s bytes=%s" % (args.out, os.path.getsize(args.out)), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("MERGE_FAIL %s: %s" % (type(exc).__name__, exc), flush=True)
        sys.exit(1)
