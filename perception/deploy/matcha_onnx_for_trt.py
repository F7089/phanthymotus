#!/usr/bin/env python3
"""Rewrite sherpa Matcha ONNX so TensorRT 8.5 can parse it.

TRT 8.5 classifies float `length_scale` as a shape tensor (it feeds
y_lengths -> Range) and then demands Int32:

  length_scale: network input that is shape tensor must have type Int32

Leaderboard uses speed=1, so fold length_scale to 1.0 and replace Range
limits on that path with a static INT32 max (padded decoder).

  python3 /deploy/matcha_onnx_for_trt.py \\
    --in /models/.../model-steps-3.onnx \\
    --out /opt/matcha_trt_cache/model-steps-3.trtprep.onnx
"""
from __future__ import annotations

import argparse

import numpy as np
import onnx
from onnx import numpy_helper


def _producer_map(graph):
    prod = {}
    for node in graph.node:
        for out in node.output:
            prod[out] = node
    return prod


def _depends_on(tensor, target, prod, seen=None):
    if tensor == target:
        return True
    if seen is None:
        seen = set()
    if not tensor or tensor in seen:
        return False
    seen.add(tensor)
    node = prod.get(tensor)
    if node is None:
        return False
    return any(_depends_on(inp, target, prod, seen) for inp in node.input)


def _drop_input(graph, name):
    keep = [i for i in graph.input if i.name != name]
    del graph.input[:]
    graph.input.extend(keep)


def _drop_initializer(graph, name):
    keep = [i for i in graph.initializer if i.name != name]
    del graph.initializer[:]
    graph.initializer.extend(keep)


def _add_const(graph, name, array):
    _drop_initializer(graph, name)
    graph.initializer.append(numpy_helper.from_array(np.asarray(array), name=name))


def fold_length_scale(graph, value=1.0):
    _drop_input(graph, "length_scale")
    _add_const(graph, "length_scale", np.array([value], dtype=np.float32))
    print("folded length_scale=%s" % value, flush=True)


def patch_duration_ranges(graph, max_mel):
    prod = _producer_map(graph)
    start_n = "trt_range_start0"
    limit_n = "trt_range_limit"
    delta_n = "trt_range_delta1"
    _add_const(graph, start_n, np.array(0, dtype=np.int32))
    _add_const(graph, limit_n, np.array(int(max_mel), dtype=np.int32))
    _add_const(graph, delta_n, np.array(1, dtype=np.int32))
    n_patch = 0
    for node in graph.node:
        if node.op_type != "Range" or len(node.input) < 3:
            continue
        if not _depends_on(node.input[1], "length_scale", prod):
            continue
        print(
            "patch Range %s limit %s -> %s=%s"
            % (node.name or "?", node.input[1], limit_n, max_mel),
            flush=True,
        )
        node.input[0] = start_n
        node.input[1] = limit_n
        node.input[2] = delta_n
        n_patch += 1
    print("patched_ranges", n_patch, flush=True)
    if n_patch == 0:
        print("WARN: no Range depends on length_scale", flush=True)
    return n_patch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--max-mel", type=int, default=2000)
    args = ap.parse_args()
    model = onnx.load(args.src)
    print("load", args.src, "bytes", __import__("os").path.getsize(args.src), flush=True)
    print("inputs_before", [i.name for i in model.graph.input], flush=True)
    fold_length_scale(model.graph, args.length_scale)
    patch_duration_ranges(model.graph, args.max_mel)
    print("inputs_after", [i.name for i in model.graph.input], flush=True)
    onnx.save(model, args.dst)
    print("wrote", args.dst, "bytes", __import__("os").path.getsize(args.dst), flush=True)


if __name__ == "__main__":
    main()
