"""Vocos via TensorRT Runtime (no ONNX Runtime session).

vocos-16khz-univ.onnx emits STFT (mag, x, y); PCM is CPU iSTFT, same as sherpa.
Engine is built once with trtexec (JP5 = TRT 8.5) into TTS_VOCOS_TRT_CACHE.
"""
from __future__ import annotations

import fcntl
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


def _istft(mag: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """mag/x/y: (n_bins, n_frames) as in sherpa vocos-vocoder.cc."""
    spec = mag * (x + 1j * y)
    try:
        from scipy.signal import istft

        _, pcm = istft(
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
    except Exception:
        n_bins, n_frames = spec.shape
        window = np.hanning(WIN).astype(np.float32)
        expected = n_frames * HOP
        acc = np.zeros(expected + WIN, dtype=np.float32)
        wsum = np.zeros_like(acc)
        for t in range(n_frames):
            frame = np.fft.irfft(spec[:, t], n=N_FFT).real[:WIN] * window
            off = t * HOP
            acc[off : off + WIN] += frame
            wsum[off : off + WIN] += window
        nz = wsum > 1e-6
        acc[nz] /= wsum[nz]
        return acc[: n_frames * HOP].astype(np.float32)


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


def _input_name(onnx_path: str) -> str:
    import onnx

    m = onnx.load(onnx_path)
    if not m.graph.input:
        raise RuntimeError("vocos ONNX has no inputs")
    return m.graph.input[0].name


def build_engine(onnx_path: str, engine_path: str, inp: str) -> None:
    exe = _trtexec()
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = engine_path + ".tmp"
    cmd = [
        exe,
        "--onnx=" + onnx_path,
        "--saveEngine=" + tmp,
        "--fp16",
        "--workspace=256",
        "--minShapes=%s:1x%dx16" % (inp, N_MELS),
        "--optShapes=%s:1x%dx256" % (inp, N_MELS),
        "--maxShapes=%s:1x%dx2000" % (inp, N_MELS),
    ]
    log.info("[tts] vocos trtexec: %s", " ".join(cmd))
    subprocess.check_call(cmd)
    os.replace(tmp, engine_path)
    log.info("[tts] vocos engine saved %s (%.1fMB)", engine_path, os.path.getsize(engine_path) / 1048576.0)


class VocosTRT:
    def __init__(self, vocos_onnx: str, cache_dir: str):
        import tensorrt as trt

        self._onnx = vocos_onnx
        self._cache = Path(cache_dir)
        self._cache.mkdir(parents=True, exist_ok=True)
        trt_ver = getattr(trt, "__version__", "0")
        self._engine_path = str(self._cache / ("vocos-16khz-univ.trt%s.fp16.engine" % trt_ver))
        self._inp = _input_name(vocos_onnx)
        self._ensure_engine()
        self._cudart = _Cudart()
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(self._engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("TensorRT deserialize failed: %s" % self._engine_path)
        self._engine = engine
        self._ctx = engine.create_execution_context()
        try:
            major = int(str(trt_ver).split(".")[0])
        except ValueError:
            major = 0
        # 8.5 may expose num_io_tensors; execute_v2/bindings is the real 8.x path.
        self._trt10 = major >= 10
        log.info(
            "[tts] vocos TensorRT runtime engine=%s trt=%s api=%s input=%s",
            self._engine_path,
            trt_ver,
            "10" if self._trt10 else "8",
            self._inp,
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
        ptrs = []
        host_out = []
        alloc = []
        try:
            for i in range(engine.num_bindings):
                shape = tuple(ctx.get_binding_shape(i))
                dtype = np.dtype(self._nptype(engine.get_binding_dtype(i)))
                nbytes = int(np.prod(shape)) * dtype.itemsize
                dptr = cuda.malloc(nbytes)
                alloc.append(dptr)
                ptrs.append(dptr)
                if engine.binding_is_input(i):
                    cuda.h2d(dptr, mel)
                else:
                    host_out.append((i, np.empty(shape, dtype=dtype), dptr))
            if not ctx.execute_v2(ptrs):
                raise RuntimeError("TensorRT execute_v2 failed")
            outs = []
            for _, buf, dptr in host_out:
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
        alloc = []
        host_out = []
        try:
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = tuple(ctx.get_tensor_shape(name))
                dtype = np.dtype(self._nptype(engine.get_tensor_dtype(name)))
                nbytes = int(np.prod(shape)) * dtype.itemsize
                dptr = cuda.malloc(nbytes)
                alloc.append(dptr)
                ctx.set_tensor_address(name, dptr)
                mode = engine.get_tensor_mode(name)
                is_input = int(mode) == 0 or str(mode).endswith("INPUT")
                if name == self._inp or is_input and name == self._inp:
                    cuda.h2d(dptr, mel)
                elif not is_input:
                    host_out.append((name, np.empty(shape, dtype=dtype), dptr))
            if hasattr(ctx, "execute_async_v3"):
                ok = ctx.execute_async_v3(0)
                self._cudart.lib.cudaStreamSynchronize(self._cudart._c_void_p())
            else:
                ok = ctx.execute_v2([0])
            if not ok:
                raise RuntimeError("TensorRT execute failed")
            outs = []
            for _, buf, dptr in host_out:
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
