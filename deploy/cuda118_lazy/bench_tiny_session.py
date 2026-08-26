#!/usr/bin/env python3
"""Tiny Conv ONNX + CUDA EP CreateSession. Requires ORT built against CUDA 11.8."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


def _mib(kb):
    return kb / 1024.0


def find_cgroup():
    usage = peak = None
    cg = ""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.strip().split(":")
            if len(parts) >= 3 and "memory" in parts[1]:
                cg = parts[2]
                break
            if line.startswith("0::"):
                cg = line.strip().split(":", 2)[-1]
    except Exception:
        cg = ""
    bases = []
    if cg:
        bases.append(Path("/sys/fs/cgroup") / cg.lstrip("/"))
        bases.append(Path("/sys/fs/cgroup/memory") / cg.lstrip("/"))
    bases.extend([Path("/sys/fs/cgroup/memory/docker"), Path("/sys/fs/cgroup/memory")])
    for base in bases:
        if not base.exists():
            continue
        for cur, pk in (
            (base / "memory.usage_in_bytes", base / "memory.max_usage_in_bytes"),
            (base / "memory.current", base / "memory.peak"),
        ):
            if cur.is_file():
                try:
                    usage = int(cur.read_text().split()[0]) / (1024.0 * 1024.0)
                except Exception:
                    usage = None
                if pk.is_file():
                    try:
                        peak = int(pk.read_text().split()[0]) / (1024.0 * 1024.0)
                    except Exception:
                        peak = None
                return usage, peak
    return usage, peak


def smaps_buckets(pid=None):
    pid = pid or os.getpid()
    rss = defaultdict(int)
    pss = defaultdict(int)
    cur = "unknown"
    path = Path("/proc/%d/smaps" % pid)
    for line in path.read_text(errors="replace").splitlines():
        if " kB" not in line[:30] and line[:1] in "0123456789abcdef":
            name = line.split()[-1] if len(line.split()) >= 6 else "[anon]"
            if name.startswith("/"):
                name = name.rsplit("/", 1)[-1]
            elif not (
                name.startswith("[")
                or name.startswith("anon")
                or name.startswith("dmabuf")
            ):
                name = "[anon]"
            cur = name
            continue
        if line.startswith("Rss:"):
            rss[cur] += int(line.split()[1])
        elif line.startswith("Pss:"):
            pss[cur] += int(line.split()[1])
    buckets = defaultdict(lambda: [0, 0])
    for name, kb in rss.items():
        if name == "[heap]":
            k = "heap"
        elif name.startswith("dmabuf"):
            k = "dmabuf"
        elif name in ("[anon]",):
            k = "anon"
        elif "cudnn" in name.lower():
            k = "cudnn"
        elif "cublas" in name.lower():
            k = "cublas"
        else:
            k = "other"
        buckets[k][0] += kb
        buckets[k][1] += pss[name]
    return buckets, sum(rss.values()), sum(pss.values())


def dump(tag):
    time.sleep(1.0)
    buckets, tot_rss, tot_pss = smaps_buckets()
    cg_u, cg_p = find_cgroup()
    print("=== CKPT %s ===" % tag, flush=True)
    print(
        "CKPT tag=%s cgroup_usage_mib=%s cgroup_max_mib=%s smaps_rss_mib=%.1f smaps_pss_mib=%.1f"
        % (
            tag,
            ("%.1f" % cg_u) if cg_u is not None else "NA",
            ("%.1f" % cg_p) if cg_p is not None else "NA",
            _mib(tot_rss),
            _mib(tot_pss),
        ),
        flush=True,
    )
    for k in ("heap", "dmabuf", "anon", "cudnn", "cublas", "other"):
        a, b = buckets.get(k, [0, 0])
        print("CKPT %s %-8s rss=%.1f pss=%.1f" % (tag, k, _mib(a), _mib(b)), flush=True)


def print_cuda_libs():
    print("--- /proc/self/maps cuda/cudart/cudnn ---", flush=True)
    for line in Path("/proc/self/maps").read_text(errors="replace").splitlines():
        if any(x in line for x in ("libcuda", "libcudart", "libcudnn", "libcublas")):
            print(line, flush=True)


def write_tiny(path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(np.ones((8, 8, 1, 1), dtype=np.float32), name="W")
    node = helper.make_node("Conv", ["X", "W"], ["Y"], kernel_shape=[1, 1])
    graph = helper.make_graph(
        [node],
        "tiny",
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 8, 4, 4])],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 8, 4, 4])],
        [w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, path)
    print("wrote_tiny", path, "bytes", os.path.getsize(path), flush=True)


def main():
    probe = os.environ.get("CUDA118_PROBE", "/opt/cuda118_lazy/probe_loading_mode")
    if os.path.isfile(probe):
        print("=== probe_loading_mode ===", flush=True)
        subprocess.check_call([probe])
    dump("A_base")
    import onnxruntime as ort

    print("onnxruntime", ort.__version__, "providers", ort.get_available_providers(), flush=True)
    dump("A2_import_ort")
    tiny = "/tmp/tiny_conv.onnx"
    write_tiny(tiny)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    cuda_opts = {
        "device_id": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": 1,
    }
    print("CreateSession tiny CUDA EP", cuda_opts, flush=True)
    sess = ort.InferenceSession(
        tiny,
        sess_options=so,
        providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"],
    )
    print("CreateSession_ok providers", sess.get_providers(), flush=True)
    dump("B_tiny")
    print_cuda_libs()
    print("CKPT_DONE tiny", flush=True)
    time.sleep(8)
    return sess


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        time.sleep(3)
        sys.exit(1)
