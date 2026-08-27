"""Matcha acoustic TensorRT runtime (no sherpa / no ORT CUDA).

Used when TTS_MATCHA_TRT=1. Vocos stays in utils.vocos_trt.VocosTRT.
Frontend (WeText + lexicon) stays in the TTS adapter.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_

log = logging.getLogger(__name__)

MAX_MEL = int(os.environ.get("TTS_TRT_MAX_MEL", "2000"))
MIN_TOKENS = int(os.environ.get("TTS_TRT_MIN_TOKENS", "8"))
MAX_TOKENS = int(os.environ.get("TTS_TRT_MAX_TOKENS", "256"))


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
    bases.extend(
        [Path("/sys/fs/cgroup/memory/docker"), Path("/sys/fs/cgroup/memory")]
    )
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


def dump_fullstack_peak(tag: str) -> None:
    u, p = find_cgroup()
    usage = "%.1f" % u if u is not None else "NA"
    peak = "%.1f" % p if p is not None else "NA"
    line = "FULLSTACK_PEAK tag=%s cgroup_usage_MB=%s cgroup_max_MB=%s" % (
        tag,
        usage,
        peak,
    )
    print(line, flush=True)
    log.info("[tts] %s", line)


def _load_trt_plugins():
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
    log.info("[tts] nvinfer_plugin=%s", loaded or "NONE")
    if hasattr(trt, "init_libnvinfer_plugins"):
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")


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


class AcousticTRT:
    def __init__(self, engine_path: str, cudart):
        import tensorrt as trt

        _load_trt_plugins()
        self._cudart = cudart
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            blob = f.read()
        engine = runtime.deserialize_cuda_engine(blob)
        if engine is None:
            raise RuntimeError("deserialize failed: %s" % engine_path)
        ctx = engine.create_execution_context()
        if ctx is None:
            raise RuntimeError("create_execution_context failed: %s" % engine_path)
        self._runtime = runtime
        self._engine = engine
        self._ctx = ctx
        self.engine_path = engine_path
        log.info(
            "[tts] matcha TRT engine=%s bytes=%s trt=%s",
            engine_path,
            len(blob),
            trt.__version__,
        )

    def infer(self, feeds: dict) -> dict:
        engine, ctx, cuda = self._engine, self._ctx, self._cudart
        nbind = int(engine.num_bindings)
        for i in range(nbind):
            if not engine.binding_is_input(i):
                continue
            name = engine.get_binding_name(i)
            if name not in feeds:
                raise RuntimeError("missing input %s" % name)
            arr = np.ascontiguousarray(feeds[name])
            feeds[name] = arr
            ctx.set_binding_shape(i, tuple(arr.shape))
        ptrs = []
        host_out = []
        alloc = []
        try:
            for i in range(nbind):
                name = engine.get_binding_name(i)
                shape = tuple(int(d) for d in ctx.get_binding_shape(i))
                if any(d < 1 for d in shape):
                    if (not engine.binding_is_input(i)) and len(shape) == 3:
                        shape = (1, 80, MAX_MEL)
                    else:
                        shape = tuple(max(d, 1) for d in shape)
                dtype = _numpy_dtype(engine.get_binding_dtype(i))
                nbytes = int(np.prod(shape)) * dtype.itemsize
                dptr = cuda.malloc(nbytes)
                alloc.append(dptr)
                ptrs.append(dptr)
                if engine.binding_is_input(i):
                    cuda.h2d(dptr, np.ascontiguousarray(feeds[name], dtype=dtype))
                else:
                    host_out.append((name, np.empty(shape, dtype=dtype), dptr))
            if not ctx.execute_v2(ptrs):
                raise RuntimeError("execute_v2 failed")
            outs = {}
            for name, buf, dptr in host_out:
                cuda.d2h(dptr, buf)
                outs[name] = buf
            return outs
        finally:
            for p in alloc:
                cuda.free(p)


def resolve_acoustic_engine(cache_dir: str) -> str:
    explicit = os.environ.get("TTS_MATCHA_TRT_ENGINE", "").strip()
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError("TTS_MATCHA_TRT_ENGINE missing: %s" % explicit)
        return explicit
    cache = Path(cache_dir)
    tactics = sorted(
        cache.glob("model-steps-3.trt8.5*.cudnn--jit_convolutions.engine")
    )
    prefer = os.environ.get("TTS_TRT_PREFER_TACTICS", "1") == "1"
    if prefer and tactics:
        return str(tactics[-1])
    cmpf = sorted(cache.glob("model-steps-3.trt8.5*.cmpf32.engine"))
    if cmpf:
        return str(cmpf[-1])
    if tactics:
        return str(tactics[-1])
    any_eng = sorted(cache.glob("model-steps-3.trt8.5*.engine"))
    if any_eng:
        return str(any_eng[-1])
    raise FileNotFoundError("no Matcha TRT engine under %s" % cache_dir)


def load_tokens(path: str) -> dict:
    tok2id = {}
    for line in open(path, encoding="utf-8"):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) == 1:
            tok2id[" "] = int(parts[0])
            continue
        a, b = parts[0], parts[-1]
        if b.lstrip("-").isdigit() and not a.lstrip("-").isdigit():
            tok2id[a] = int(b)
        elif a.lstrip("-").isdigit():
            tok2id[b] = int(a)
        else:
            tok2id[a] = int(b)
    return tok2id


def load_lexicon(path: str) -> dict:
    lex = {}
    for line in open(path, encoding="utf-8"):
        parts = line.strip().split()
        if len(parts) >= 2:
            lex[parts[0]] = parts[1:]
    return lex


def text_to_ids(text: str, lex: dict, tok2id: dict):
    phones = []
    skipped = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or ch in "，。！？、；：,.!?;:\"'“”‘’":
            i += 1
            continue
        if ch.isascii() and (ch.isalpha() or ch == "'"):
            j = i + 1
            while j < len(text) and text[j].isascii() and (text[j].isalpha() or text[j] == "'"):
                j += 1
            word = text[i:j]
            key = word.lower()
            if word in lex:
                phones.extend(lex[word])
            elif key in lex:
                phones.extend(lex[key])
            elif word in tok2id or key in tok2id:
                phones.append(word if word in tok2id else key)
            else:
                for c in word:
                    cu = c.upper()
                    cl = c.lower()
                    if c in lex:
                        phones.extend(lex[c])
                    elif cu in lex:
                        phones.extend(lex[cu])
                    elif cl in tok2id:
                        phones.append(cl)
                    elif cu in tok2id:
                        phones.append(cu)
                    else:
                        skipped.append(c)
            i = j
            continue
        if ch in lex:
            phones.extend(lex[ch])
        elif ch in tok2id:
            phones.append(ch)
        else:
            skipped.append(ch)
        i += 1
    ids = []
    unk = tok2id.get("<unk>", tok2id.get("UNK"))
    missing = []
    for p in phones:
        if p in tok2id:
            ids.append(tok2id[p])
        elif unk is not None:
            ids.append(unk)
            missing.append(p)
        else:
            missing.append(p)
    for name in ("<sos>", "<bos>", "sos", "<s>"):
        if name in tok2id:
            ids = [tok2id[name]] + ids
            break
    for name in ("<eos>", "</s>", "eos"):
        if name in tok2id:
            ids = ids + [tok2id[name]]
            break
    return ids, phones, skipped, missing


def pad_token_ids(ids, tok2id):
    real_len = len(ids)
    if real_len > MAX_TOKENS:
        ids = ids[:MAX_TOKENS]
        real_len = MAX_TOKENS
    pad_id = tok2id.get("<pad>", tok2id.get("pad", 0))
    if real_len < MIN_TOKENS:
        ids = ids + [pad_id] * (MIN_TOKENS - real_len)
    return ids, real_len


def crop_mel(mel, n_tokens: int):
    m = np.asarray(mel, dtype=np.float32)
    if m.ndim == 3:
        m = m[0]
    e = (m.astype(np.float64) ** 2).mean(axis=0)
    mx = float(e.max()) if e.size else 0.0
    if mx > 0:
        hits = np.where(e > mx * 0.02)[0]
        if hits.size:
            end = min(m.shape[1], int(hits[-1]) + 8)
            start = max(0, int(hits[0]))
            m = m[:, start:end]
    cap = max(80, min(MAX_MEL, int(n_tokens) * 24))
    if m.shape[1] > cap:
        m = m[:, :cap]
    return m
