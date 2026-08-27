#!/usr/bin/env python3
"""Load Matcha acoustic + Vocos TensorRT engines (no sherpa / no ORT).

Build engines in a throwaway container first (see bench_matcha_trt_mem.sh).

  python3 /deploy/bench_matcha_trt_mem.py --engines /opt/matcha_trt_cache/acoustic.engine
  python3 /deploy/bench_matcha_trt_mem.py --engines acoustic.engine,vocos.engine
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
        if n <= 0:
            return 0
        p = self._c_void_p()
        err = self.lib.cudaMalloc(ctypes.byref(p), n)
        if err:
            raise RuntimeError("cudaMalloc(%s) failed err=%s" % (n, err))
        if not p.value:
            raise RuntimeError("cudaMalloc(%s) returned NULL" % n)
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


_PLUGINS_LOADED = False


def _load_trt_plugins():
    """trtexec links InstanceNorm etc. via nvinfer_plugin; Python Runtime does not."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    import ctypes
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
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        print("init_libnvinfer_plugins ok", flush=True)
    _PLUGINS_LOADED = True


def deserialize(engine_path, keep):
    import tensorrt as trt

    _load_trt_plugins()
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
    print("deserialize_ok", os.path.basename(engine_path), flush=True)
    return engine, ctx


def _numpy_dtype(trt_dtype):
    import tensorrt as trt

    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT32: np.int32,
        trt.DataType.INT8: np.int8,
        trt.DataType.BOOL: np.bool_,
    }
    if hasattr(trt.DataType, "INT64"):
        mapping[trt.DataType.INT64] = np.int64
    if trt_dtype in mapping:
        return np.dtype(mapping[trt_dtype])
    try:
        return np.dtype(trt.nptype(trt_dtype))
    except Exception:
        return np.dtype(np.float32)


def _guess_shape(name, shape):
    name = (name or "").lower()
    if not any(d < 1 for d in shape):
        return tuple(shape)
    if name in ("x", "tokens", "token_ids") and len(shape) == 2:
        return (1, 32)
    if "length" in name and "scale" not in name and len(shape) == 1:
        return (1,)
    if name in ("mel",) or len(shape) == 3:
        t = int(os.environ.get("TTS_TRT_MAX_MEL", "2000"))
        return (1, 80, t)
    return tuple(1 if d < 1 else d for d in shape)


def _fill_input(name, shape, dtype):
    name = (name or "").lower()
    host = np.zeros(shape, dtype=dtype)
    if "scale" in name:
        host[...] = 1.0
    elif name in ("x", "tokens", "token_ids"):
        host[...] = 1
    elif "length" in name:
        host[...] = 32
    return host


def warmup(engine, ctx, keep, cuda, tag):
    nbind = int(engine.num_bindings)
    print("warmup_start", tag, "bindings", nbind, flush=True)
    for i in range(nbind):
        if not engine.binding_is_input(i):
            continue
        name = engine.get_binding_name(i)
        shape = tuple(ctx.get_binding_shape(i))
        new = _guess_shape(name, shape)
        if new != shape:
            ok = ctx.set_binding_shape(i, new)
            print("set_shape", tag, name, shape, "->", new, "ok", ok, flush=True)
    ptrs = []
    for i in range(nbind):
        name = engine.get_binding_name(i)
        shape = tuple(ctx.get_binding_shape(i))
        if any(d < 1 for d in shape):
            new = _guess_shape(name, shape)
            try:
                ctx.set_binding_shape(i, new)
            except Exception:
                pass
            shape = tuple(d if d > 0 else nd for d, nd in zip(shape, new))
            if any(d < 1 for d in shape):
                shape = new
            print("fix_shape", tag, name, "->", shape, flush=True)
        dtype = _numpy_dtype(engine.get_binding_dtype(i))
        nbytes = int(np.prod(shape)) * dtype.itemsize
        dptr = cuda.malloc(nbytes)
        keep.append(dptr)
        ptrs.append(dptr)
        if engine.binding_is_input(i) and dptr:
            host = _fill_input(name, shape, dtype)
            cuda.h2d(dptr, host)
            print("input", tag, name, "shape", shape, "dtype", dtype, flush=True)
        else:
            print("output", tag, name, "shape", shape, "dtype", dtype, "nbytes", nbytes, flush=True)
    if not ctx.execute_v2(ptrs):
        raise RuntimeError("execute_v2 failed: %s" % tag)
    print("warmup_ok", tag, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", required=True, help="comma-separated engine paths")
    args = ap.parse_args()
    paths = [p.strip() for p in args.engines.split(",") if p.strip()]
    if not paths:
        raise SystemExit("no engines")
    for p in paths:
        if not os.path.isfile(p):
            raise SystemExit("engine missing: %s" % p)
    keep = []
    dump("A_base")
    import tensorrt as trt  # noqa: F401

    print("imported tensorrt", trt.__version__, flush=True)
    dump("B_import_trt")
    loaded = []
    for i, path in enumerate(paths):
        tag = "C%d_%s" % (i, os.path.basename(path)[:40])
        engine, ctx = deserialize(path, keep)
        loaded.append((tag, engine, ctx, path))
        dump(tag)
    cuda = _Cudart()
    keep.append(cuda)
    dump("C_cuda_ctx")
    for tag, engine, ctx, path in loaded:
        warmup(engine, ctx, keep, cuda, os.path.basename(path))
        dump("D_%s" % os.path.basename(path)[:40])
    print("CKPT_DONE engines=%s" % ",".join(paths), flush=True)
    time.sleep(8)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        time.sleep(3)
        raise
