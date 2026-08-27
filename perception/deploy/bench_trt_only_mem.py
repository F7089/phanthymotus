#!/usr/bin/env python3
"""VITS2-style TensorRT-only load: no sherpa, no ORT CUDA Session.

Same method as 4paradigm vits2_tts_trt/runtime/backends/trt_cuda_session.py:
import tensorrt -> deserialize_cuda_engine -> create_execution_context ->
cudaMalloc + execute. Compare host cgroup to tiny ORT CreateSession ~786MB.

Do not run trtexec in this process (it shares the container cgroup).
Build engines in a throwaway container first.

  python3 /deploy/bench_trt_only_mem.py --mode import
  python3 /deploy/bench_trt_only_mem.py --mode tiny --engine /opt/vocos_trt_cache/tiny_conv.fp16.ws64.engine
  python3 /deploy/bench_trt_only_mem.py --mode vocos --engine /opt/vocos_trt_cache/vocos-....engine
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_


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
        elif "cudnn" in name.lower() or "nvinfer" in name.lower():
            k = "cudnn_trt"
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
    for k in ("heap", "dmabuf", "anon", "cudnn_trt", "cublas", "other"):
        a, b = buckets.get(k, [0, 0])
        print("CKPT %s %-8s rss=%.1f pss=%.1f" % (tag, k, _mib(a), _mib(b)), flush=True)


def find_vocos_engine(cache):
    cache = Path(cache)
    if not cache.is_dir():
        return None
    cands = sorted(cache.glob("vocos-16khz-univ*.engine"), key=lambda p: -p.stat().st_size)
    return str(cands[0]) if cands else None


class _Cudart:
    def __init__(self):
        lib = ctypes.CDLL("libcudart.so")
        self._c_void_p = ctypes.c_void_p
        self._size = ctypes.c_size_t
        self._int = ctypes.c_int
        lib.cudaSetDevice.argtypes = [self._int]
        lib.cudaSetDevice.restype = self._int
        lib.cudaMalloc.argtypes = [ctypes.POINTER(self._c_void_p), self._size]
        lib.cudaMalloc.restype = self._int
        lib.cudaFree.argtypes = [self._c_void_p]
        lib.cudaFree.restype = self._int
        lib.cudaMemcpy.argtypes = [
            self._c_void_p,
            self._c_void_p,
            self._size,
            self._int,
        ]
        lib.cudaMemcpy.restype = self._int
        self.lib = lib
        err = lib.cudaSetDevice(0)
        if err:
            raise RuntimeError("cudaSetDevice(0) failed err=%s" % err)

    def malloc(self, n):
        p = self._c_void_p()
        err = self.lib.cudaMalloc(ctypes.byref(p), n)
        if err or not p.value:
            raise RuntimeError("cudaMalloc(%s) failed err=%s" % (n, err))
        return int(p.value)

    def h2d(self, dst, host):
        buf = np.ascontiguousarray(host)
        err = self.lib.cudaMemcpy(
            self._c_void_p(dst),
            buf.ctypes.data_as(self._c_void_p),
            buf.nbytes,
            1,
        )
        if err:
            raise RuntimeError("cudaMemcpy H2D failed err=%s" % err)


def deserialize(engine_path, keep):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open(engine_path, "rb") as f:
        blob = f.read()
    print(
        "deserialize",
        engine_path,
        "bytes",
        len(blob),
        "trt",
        trt.__version__,
        flush=True,
    )
    engine = runtime.deserialize_cuda_engine(blob)
    if engine is None:
        raise RuntimeError("deserialize failed: %s" % engine_path)
    ctx = engine.create_execution_context()
    if ctx is None:
        raise RuntimeError("create_execution_context failed")
    keep.extend([runtime, engine, ctx])
    print("deserialize_ok", engine_path, flush=True)
    return engine, ctx


def _numpy_dtype(trt_dtype):
    import tensorrt as trt

    try:
        return np.dtype(trt.nptype(trt_dtype))
    except Exception:
        return np.dtype(np.float32)


def warmup(engine, ctx, keep):
    """One execute like VITS2 TensorRTCudaSession.run (keep device buffers)."""
    import tensorrt as trt  # noqa: F401

    cuda = _Cudart()
    keep.append(cuda)
    nbind = int(engine.num_bindings)
    ptrs = []
    print("warmup_bindings", nbind, flush=True)
    for i in range(nbind):
        if engine.binding_is_input(i):
            shape = tuple(ctx.get_binding_shape(i))
            if any(d < 0 for d in shape):
                ctx.set_binding_shape(i, (1, 80, 16))
        shape = tuple(ctx.get_binding_shape(i))
        if any(d < 0 for d in shape):
            raise RuntimeError("unresolved binding %d shape=%s" % (i, shape))
        dtype = _numpy_dtype(engine.get_binding_dtype(i))
        nbytes = int(np.prod(shape)) * dtype.itemsize
        dptr = cuda.malloc(nbytes)
        keep.append(dptr)
        ptrs.append(dptr)
        if engine.binding_is_input(i):
            host = np.zeros(shape, dtype=dtype)
            cuda.h2d(dptr, host)
    if not ctx.execute_v2(ptrs):
        raise RuntimeError("execute_v2 failed")
    print("warmup_ok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("tiny", "vocos", "import"), default="tiny")
    ap.add_argument("--engine", default="")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()
    keep = []
    dump("A_base")
    import tensorrt as trt  # noqa: F401

    print("imported tensorrt", trt.__version__, flush=True)
    dump("B_import_trt")
    if args.mode == "import":
        print("CKPT_DONE mode=import", flush=True)
        time.sleep(8)
        return
    cache = os.environ.get("TTS_VOCOS_TRT_CACHE", "/opt/vocos_trt_cache")
    engine_path = args.engine.strip()
    if not engine_path:
        if args.mode == "tiny":
            engine_path = os.path.join(cache, "tiny_conv.fp16.ws64.engine")
        else:
            engine_path = find_vocos_engine(cache) or ""
    if not engine_path or not os.path.isfile(engine_path):
        raise SystemExit(
            "engine missing (build in a throwaway container first): %s" % engine_path
        )
    engine, ctx = deserialize(engine_path, keep)
    dump("C_deserialize")
    if not args.no_warmup:
        warmup(engine, ctx, keep)
        dump("D_warmup")
    print("CKPT_DONE mode=%s engine=%s" % (args.mode, engine_path), flush=True)
    time.sleep(8)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        time.sleep(3)
        raise
