"""Vocos via TensorRT Runtime (no ONNX Runtime session).

vocos-16khz-univ.onnx emits STFT (mag, x, y); PCM is CPU iSTFT, same as sherpa.
Engine is built once with trtexec (JP5 = TRT 8.5) into TTS_VOCOS_TRT_CACHE.

Ranking path with a prebuilt engine must not import onnx or scipy.
"""
from __future__ import annotations

import fcntl
import gc
import logging
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

# TensorRT 8.5 Python still references numpy.bool (removed in NumPy 1.24).
if not hasattr(np, "bool"):
    np.bool = np.bool_

log = logging.getLogger(__name__)

N_MELS = 80
N_FFT = 1024
HOP = 256
WIN = 1024
SAMPLE_RATE = 16000


def _hann_periodic(n: int) -> np.ndarray:
    """scipy.get_window('hann', n, fftbins=True)."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n))


def istft_numpy(mag: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CPU iSTFT matching scipy.signal.istft for Vocos 16 kHz.

    n_fft=1024, hop=256, win=1024, periodic hann, onesided, boundary=True.
    """
    spec = np.asarray(mag * (x + 1j * y), dtype=np.complex128)
    n_bins, nseg = spec.shape
    if n_bins != N_FFT // 2 + 1:
        raise RuntimeError("vocos istft expected %s bins, got %s" % (N_FFT // 2 + 1, n_bins))
    win = _hann_periodic(WIN)
    frames = np.fft.irfft(spec, n=N_FFT, axis=0).real[:WIN, :]
    out_len = WIN + (nseg - 1) * HOP
    acc = np.zeros(out_len, dtype=np.float64)
    w2 = np.zeros(out_len, dtype=np.float64)
    for t in range(nseg):
        off = t * HOP
        acc[off : off + WIN] += frames[:, t] * win
        w2[off : off + WIN] += win * win
    acc /= np.where(w2 > 1e-10, w2, 1.0)
    acc = acc[WIN // 2 : acc.size - WIN // 2]
    return acc.astype(np.float32)


def istft_scipy(mag: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Debug/fallback only. Ranking default must not import scipy."""
    from scipy.signal import istft as _scipy_istft

    spec = mag * (x + 1j * y)
    _, pcm = _scipy_istft(
        spec,
        fs=SAMPLE_RATE,
        nperseg=WIN,
        noverlap=WIN - HOP,
        nfft=N_FFT,
        window="hann",
        input_onesided=True,
        boundary=True,
    )
    return np.asarray(pcm, dtype=np.float32)


def _istft(mag: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    mode = os.environ.get("TTS_ISTFT", "numpy").strip().lower()
    if mode == "scipy":
        return istft_scipy(mag, x, y)
    return istft_numpy(mag, x, y)


class _Cudart:
    def __init__(self):
        import ctypes

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
        lib.cudaMemcpy.argtypes = [self._c_void_p, self._c_void_p, self._size, self._int]
        lib.cudaMemcpy.restype = self._int
        lib.cudaStreamSynchronize.argtypes = [self._c_void_p]
        lib.cudaStreamSynchronize.restype = self._int
        self.lib = lib
        err = lib.cudaSetDevice(0)
        if err:
            raise RuntimeError("cudaSetDevice(0) failed err=%s" % err)

    def malloc(self, n: int) -> int:
        import ctypes

        p = self._c_void_p()
        err = self.lib.cudaMalloc(ctypes.byref(p), n)
        if err or not p.value:
            raise RuntimeError("cudaMalloc(%s) failed err=%s" % (n, err))
        return int(p.value)

    def free(self, p: int) -> None:
        if p:
            self.lib.cudaFree(self._c_void_p(p))

    def h2d(self, dst: int, host: np.ndarray) -> None:
        import ctypes

        buf = np.ascontiguousarray(host)
        err = self.lib.cudaMemcpy(
            self._c_void_p(dst), buf.ctypes.data_as(self._c_void_p), buf.nbytes, 1
        )
        if err:
            raise RuntimeError("cudaMemcpy H2D failed err=%s" % err)

    def d2h(self, src: int, host: np.ndarray) -> None:
        import ctypes

        buf = np.ascontiguousarray(host)
        err = self.lib.cudaMemcpy(
            buf.ctypes.data_as(self._c_void_p), self._c_void_p(src), buf.nbytes, 2
        )
        if err:
            raise RuntimeError("cudaMemcpy D2H failed err=%s" % err)


class CudaBufferPool:
    """Grow-on-demand device buffers. Reuse when nbytes <= capacity."""

    def __init__(self, cudart: _Cudart):
        self._cudart = cudart
        self._slots = []

    def ensure(self, nbytes_list):
        while len(self._slots) < len(nbytes_list):
            self._slots.append({"ptr": 0, "cap": 0})
        ptrs = []
        for i, n in enumerate(nbytes_list):
            n = int(n)
            slot = self._slots[i]
            if n > slot["cap"]:
                if slot["ptr"]:
                    self._cudart.free(slot["ptr"])
                slot["ptr"] = self._cudart.malloc(n) if n > 0 else 0
                slot["cap"] = n
            ptrs.append(slot["ptr"])
        return ptrs

    def close(self) -> None:
        for slot in self._slots:
            if slot["ptr"]:
                self._cudart.free(slot["ptr"])
                slot["ptr"] = 0
                slot["cap"] = 0
        self._slots = []


def deserialize_cuda_engine(runtime, engine_path: str):
    """Read engine bytes, deserialize, drop the Python blob immediately."""
    n = os.path.getsize(engine_path)
    with open(engine_path, "rb") as f:
        blob = f.read()
    try:
        engine = runtime.deserialize_cuda_engine(blob)
    finally:
        del blob
        gc.collect()
    if engine is None:
        raise RuntimeError("TensorRT deserialize failed: %s" % engine_path)
    return engine, n


def _input_name_from_engine(engine) -> str:
    nbind = int(engine.num_bindings)
    for i in range(nbind):
        if engine.binding_is_input(i):
            return engine.get_binding_name(i)
    raise RuntimeError("vocos engine has no input bindings")


def _input_name_onnx(onnx_path: str) -> str:
    import onnx

    m = onnx.load(onnx_path)
    if not m.graph.input:
        raise RuntimeError("vocos ONNX has no inputs")
    return m.graph.input[0].name


def _trtexec() -> str:
    for p in (
        os.environ.get("TRTEXEC", "").strip(),
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/bin/trtexec",
        shutil.which("trtexec") or "",
    ):
        if p and os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    raise FileNotFoundError("trtexec not found (JP5 TensorRT 8.5 expected)")


def build_engine(onnx_path: str, engine_path: str, inp: str) -> None:
    exe = _trtexec()
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = engine_path + ".tmp"
    cmd = [
        exe,
        "--onnx=" + onnx_path,
        "--saveEngine=" + tmp,
        "--fp16",
        "--workspace=64",
        "--minShapes=%s:1x%dx16" % (inp, N_MELS),
        "--optShapes=%s:1x%dx256" % (inp, N_MELS),
        "--maxShapes=%s:1x%dx2000" % (inp, N_MELS),
    ]
    log.info("[tts] vocos trtexec: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    os.replace(tmp, engine_path)
    log.info("[tts] vocos engine saved %s (%.1fMB)", engine_path, os.path.getsize(engine_path) / 1048576.0)


class VocosTRT:
    def __init__(self, vocos_onnx: str, cache_dir: str, cudart=None):
        import tensorrt as trt

        self._onnx = vocos_onnx
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        trt_ver = getattr(trt, "__version__", "0")
        env_eng = os.environ.get("TTS_VOCOS_TRT_ENGINE", "").strip()
        if env_eng:
            self._engine_path = env_eng
        else:
            self._engine_path = str(
                self._cache / ("vocos-16khz-univ.trt%s.fp16.ws64.engine" % trt_ver)
            )
        engine_ready = (
            os.path.isfile(self._engine_path) and os.path.getsize(self._engine_path) > 0
        )
        force_onnx = os.environ.get("TTS_VOCOS_PARSE_ONNX", "0") == "1"
        if engine_ready and not force_onnx:
            self._inp = None
        else:
            self._inp = _input_name_onnx(vocos_onnx)
            if not engine_ready:
                self._ensure_engine()
        self._cudart = cudart or _Cudart()
        self._bufs = CudaBufferPool(self._cudart)
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine, nbytes = deserialize_cuda_engine(runtime, self._engine_path)
        self._runtime = runtime
        self._engine = engine
        self._ctx = engine.create_execution_context()
        if self._inp is None:
            self._inp = _input_name_from_engine(engine)
        try:
            major = int(str(trt_ver).split(".")[0])
        except ValueError:
            major = 0
        self._trt10 = major >= 10
        self._reuse = os.environ.get("TTS_TRT_BUF_REUSE", "1") != "0"
        log.info(
            "[tts] vocos TensorRT runtime engine=%s bytes=%s trt=%s api=%s input=%s reuse=%s istft=%s",
            self._engine_path,
            nbytes,
            trt_ver,
            "10" if self._trt10 else "8",
            self._inp,
            self._reuse,
            os.environ.get("TTS_ISTFT", "numpy"),
        )

    def _ensure_engine(self) -> None:
        if os.path.isfile(self._engine_path) and os.path.getsize(self._engine_path) > 0:
            log.info("[tts] vocos engine reuse %s", self._engine_path)
            return
        lock_p = self._cache / "vocos.build.lock"
        with open(lock_p, "w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if os.path.isfile(self._engine_path) and os.path.getsize(self._engine_path) > 0:
                log.info("[tts] vocos engine reuse after wait %s", self._engine_path)
                return
            build_engine(self._onnx, self._engine_path, self._inp)

    def close(self) -> None:
        if getattr(self, "_bufs", None) is not None:
            self._bufs.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def mel_flat_to_pcm(self, flat) -> np.ndarray:
        mel = np.asarray(flat, dtype=np.float32).reshape(-1)
        if mel.size % N_MELS != 0:
            raise RuntimeError("mel size %s not divisible by %s" % (mel.size, N_MELS))
        t = mel.size // N_MELS
        return self.infer(mel.reshape(1, N_MELS, t))

    def infer(self, mel: np.ndarray) -> np.ndarray:
        mel = np.ascontiguousarray(mel, dtype=np.float32)
        if mel.ndim != 3 or mel.shape[0] != 1 or mel.shape[1] != N_MELS:
            raise RuntimeError("vocos mel expected (1, %s, T), got %s" % (N_MELS, mel.shape))
        log.info("[tts] vocos infer start shape=%s api=%s", mel.shape, "10" if self._trt10 else "8")
        if self._trt10:
            mag, x, y = self._infer10(mel)
        else:
            mag, x, y = self._infer8(mel)
        return _istft(mag, x, y)

    def _infer8(self, mel: np.ndarray):
        engine, ctx, cuda = self._engine, self._ctx, self._cudart
        in_idx = None
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i) and engine.get_binding_name(i) == self._inp:
                in_idx = i
                break
        if in_idx is None:
            in_idx = 0
        ctx.set_binding_shape(in_idx, tuple(mel.shape))
        nbytes_list = []
        meta = []
        for i in range(engine.num_bindings):
            shape = tuple(ctx.get_binding_shape(i))
            dtype = np.dtype(self._nptype(engine.get_binding_dtype(i)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            nbytes_list.append(nbytes)
            meta.append((i, shape, dtype, bool(engine.binding_is_input(i))))
        if self._reuse:
            ptrs = self._bufs.ensure(nbytes_list)
            alloc = []
        else:
            ptrs = [cuda.malloc(n) for n in nbytes_list]
            alloc = list(ptrs)
        try:
            host_out = []
            for (i, shape, dtype, is_in), dptr in zip(meta, ptrs):
                if is_in:
                    cuda.h2d(dptr, mel)
                else:
                    host_out.append((np.empty(shape, dtype=dtype), dptr))
            if not ctx.execute_v2(ptrs):
                raise RuntimeError("TensorRT execute_v2 failed")
            outs = []
            for buf, dptr in host_out:
                cuda.d2h(dptr, buf)
                outs.append(buf)
        finally:
            for p in alloc:
                cuda.free(p)
        if len(outs) < 3:
            raise RuntimeError("vocos engine expected 3 outputs, got %s" % len(outs))
        return outs[0][0], outs[1][0], outs[2][0]

    def _infer10(self, mel: np.ndarray):
        engine, ctx, cuda = self._engine, self._ctx, self._cudart
        ctx.set_input_shape(self._inp, tuple(mel.shape))
        nbytes_list = []
        meta = []
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(ctx.get_tensor_shape(name))
            dtype = np.dtype(self._nptype(engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            nbytes_list.append(nbytes)
            mode = engine.get_tensor_mode(name)
            is_input = int(mode) == 0 or str(mode).endswith("INPUT")
            meta.append((name, shape, dtype, is_input))
        if self._reuse:
            ptrs = self._bufs.ensure(nbytes_list)
            alloc = []
        else:
            ptrs = [cuda.malloc(n) for n in nbytes_list]
            alloc = list(ptrs)
        try:
            host_out = []
            for (name, shape, dtype, is_input), dptr in zip(meta, ptrs):
                ctx.set_tensor_address(name, dptr)
                if name == self._inp or is_input and name == self._inp:
                    cuda.h2d(dptr, mel)
                elif not is_input:
                    host_out.append((np.empty(shape, dtype=dtype), dptr))
            if hasattr(ctx, "execute_async_v3"):
                ok = ctx.execute_async_v3(0)
                self._cudart.lib.cudaStreamSynchronize(self._cudart._c_void_p())
            else:
                ok = ctx.execute_v2([0])
            if not ok:
                raise RuntimeError("TensorRT execute failed")
            outs = []
            for buf, dptr in host_out:
                cuda.d2h(dptr, buf)
                outs.append(buf)
        finally:
            for p in alloc:
                cuda.free(p)
        if len(outs) < 3:
            raise RuntimeError("vocos engine expected 3 outputs, got %s" % len(outs))
        return outs[0][0], outs[1][0], outs[2][0]

    @staticmethod
    def _nptype(trt_dtype):
        import tensorrt as trt

        return trt.nptype(trt_dtype)
