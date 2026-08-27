#!/usr/bin/env python3
"""Build Matcha TRT engine with Python BuilderConfig (TRT 8.5).

JP5 trtexec 8.5.2 has no --preview. The C++/Python enum may still exist:
  PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805

Same graph/profiles as the working cmpf32 trtexec command. Does not set
--tacticSources=-CUDNN (plugins can still receive cudnnContext).
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys


def _load_plugins(logger):
    import tensorrt as trt

    loaded = None
    for name in (
        "libnvinfer_plugin.so.8",
        "libnvinfer_plugin.so",
        "/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.8",
        "/usr/lib/aarch64-linux-gnu/tegra/libnvinfer_plugin.so.8",
    ):
        try:
            ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
            loaded = name
            break
        except OSError:
            continue
    print("plugin_lib", loaded or "NONE", flush=True)
    if hasattr(trt, "init_libnvinfer_plugins"):
        trt.init_libnvinfer_plugins(logger, "")
        print("init_libnvinfer_plugins ok", flush=True)


def _preview_enum(name: str):
    import tensorrt as trt

    pf = getattr(trt, "PreviewFeature", None)
    names = []
    if pf is not None:
        names = [x for x in dir(pf) if x[:1].isupper() or "_" in x]
        names = [x for x in names if not x.startswith("_")]
    print("PreviewFeature", names, flush=True)
    key = name.strip().lstrip("+-")
    aliases = {
        "disableExternalTacticSourcesForCore0805": "DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805",
        "DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805": "DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805",
        "fasterDynamicShapes0805": "FASTER_DYNAMIC_SHAPES_0805",
        "FASTER_DYNAMIC_SHAPES_0805": "FASTER_DYNAMIC_SHAPES_0805",
    }
    enum_name = aliases.get(key, key)
    feat = getattr(pf, enum_name, None) if pf is not None else None
    if feat is None:
        print("FATAL: tensorrt.PreviewFeature missing", enum_name, flush=True)
        raise SystemExit(2)
    print("preview_enum", enum_name, feat, flush=True)
    return feat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--preview", action="append", default=[])
    ap.add_argument("--workspace-mb", type=int, default=4096)
    ap.add_argument("--min-tokens", type=int, default=8)
    ap.add_argument("--opt-tokens", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    import tensorrt as trt

    print("trt", trt.__version__, flush=True)
    print("onnx", args.onnx, "bytes", os.path.getsize(args.onnx), flush=True)
    logger = trt.Logger(trt.Logger.INFO)
    _load_plugins(logger)

    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as f:
        ok = parser.parse(f.read())
    nerr = int(getattr(parser, "num_errors", 0) or 0)
    for i in range(nerr):
        print("parse_error", parser.get_error(i), flush=True)
    if not ok:
        raise SystemExit("ONNX parse failed")
    print("inputs", [network.get_input(i).name for i in range(network.num_inputs)], flush=True)
    print("outputs", [network.get_output(i).name for i in range(network.num_outputs)], flush=True)

    config = builder.create_builder_config()
    ws = int(args.workspace_mb) << 20
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws)
    else:
        config.max_workspace_size = ws
    if hasattr(trt, "BuilderFlag") and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
        print("fp16 on", flush=True)
    for raw in args.preview:
        for part in str(raw).split(","):
            part = part.strip()
            if not part:
                continue
            feat = _preview_enum(part)
            config.set_preview_feature(feat, True)
            print("set_preview_feature", part, True, flush=True)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "x",
        (1, args.min_tokens),
        (1, args.opt_tokens),
        (1, args.max_tokens),
    )
    profile.set_shape("x_length", (1,), (1,), (1,))
    config.add_optimization_profile(profile)
    print(
        "profile x 1x%s/1x%s/1x%s x_length 1"
        % (args.min_tokens, args.opt_tokens, args.max_tokens),
        flush=True,
    )
    print("build_start workspace_mb", args.workspace_mb, flush=True)

    blob = None
    if hasattr(builder, "build_serialized_network"):
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise SystemExit("build_serialized_network returned None")
        blob = bytes(blob)
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise SystemExit("build_engine returned None")
        blob = bytes(engine.serialize())
    tmp = args.out + ".tmp"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, args.out)
    print("wrote", args.out, "bytes", os.path.getsize(args.out), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise
