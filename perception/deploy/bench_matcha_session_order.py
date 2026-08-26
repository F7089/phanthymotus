#!/usr/bin/env python3
"""Create ORT CUDA sessions in a chosen order using sherpa's libonnxruntime.

This is not OfflineTts(). It uses the same ORT 1.16 CUDA EP as sherpa Matcha,
so we can put Vocos or a tiny Conv *before* acoustic and see who pays the
first-session tax.

  python3 /deploy/bench_matcha_session_order.py --order vocos,acoustic
  python3 /deploy/bench_matcha_session_order.py --order acoustic,vocos
  python3 /deploy/bench_matcha_session_order.py --order tiny,acoustic
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

sys.path.insert(0, "/deploy")
from bench_matcha_load_ckpt import disable_sherpa_trt, dump  # noqa: E402

IDX_CREATE_STATUS = 0
IDX_GET_ERROR_CODE = 1
IDX_GET_ERROR_MESSAGE = 2
IDX_CREATE_ENV = 3
IDX_CREATE_SESSION = 7
IDX_CREATE_SESSION_OPTS = 10
IDX_SET_INTRA = 24
IDX_CUDA_V1 = 152
ORT_LOGGING_WARNING = 2


class OrtCUDAProviderOptions(ctypes.Structure):
    _fields_ = [
        ("device_id", ctypes.c_int),
        ("cudnn_conv_algo_search", ctypes.c_int),
        ("gpu_mem_limit", ctypes.c_size_t),
        ("arena_extend_strategy", ctypes.c_int),
        ("do_copy_in_default_stream", ctypes.c_int),
        ("has_user_compute_stream", ctypes.c_int),
        ("user_compute_stream", ctypes.c_void_p),
        ("default_memory_arena_cfg", ctypes.c_void_p),
        ("tunable_op_enable", ctypes.c_int),
        ("tunable_op_tuning_enable", ctypes.c_int),
        ("tunable_op_max_tuning_duration_ms", ctypes.c_int),
    ]


def _fn(api, idx, restype, *argtypes):
    ptr = api[idx]
    if not ptr:
        raise RuntimeError("OrtApi slot %d is NULL" % idx)
    return ctypes.CFUNCTYPE(restype, *argtypes)(ptr)


def load_ort_api():
    import sherpa_onnx

    libdir = Path(sherpa_onnx.__file__).resolve().parent / "lib"
    so = next(libdir.glob("libonnxruntime.so*"))
    for name in (
        so.name,
        "libonnxruntime_providers_shared.so",
        "libonnxruntime_providers_cuda.so",
    ):
        path = libdir / name if name != so.name else so
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            print("dlopen", path.name, flush=True)
    lib = ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
    lib.OrtGetApiBase.restype = ctypes.c_void_p
    base = lib.OrtGetApiBase()
    if not base:
        raise RuntimeError("OrtGetApiBase NULL")
    GetApi = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32)(
        ctypes.cast(base, ctypes.POINTER(ctypes.c_void_p))[0]
    )
    api_ptr = GetApi(16)
    if not api_ptr:
        raise RuntimeError("GetApi(16) NULL")
    api = ctypes.cast(api_ptr, ctypes.POINTER(ctypes.c_void_p))
    print("ort_api", hex(api_ptr), "lib", so, flush=True)
    return api


def check(api, st, what):
    if not st:
        return
    GetErrorMessage = _fn(api, IDX_GET_ERROR_MESSAGE, ctypes.c_char_p, ctypes.c_void_p)
    msg = GetErrorMessage(st)
    raise RuntimeError("%s: %s" % (what, msg.decode("utf-8", "replace") if msg else "status"))


def make_tiny_onnx(path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = numpy_helper.from_array(
        np.ones((8, 8, 1, 1), dtype=np.float32), name="W"
    )
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


def create_cuda_session(api, env, onnx_path):
    CreateSO = _fn(
        api, IDX_CREATE_SESSION_OPTS, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)
    )
    SetIntra = _fn(api, IDX_SET_INTRA, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    AppendCUDA = _fn(
        api,
        IDX_CUDA_V1,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(OrtCUDAProviderOptions),
    )
    CreateSession = _fn(
        api,
        IDX_CREATE_SESSION,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    opts = ctypes.c_void_p()
    check(api, CreateSO(ctypes.byref(opts)), "CreateSessionOptions")
    check(api, SetIntra(opts, 2), "SetIntraOpNumThreads")
    cuda_opts = OrtCUDAProviderOptions()
    cuda_opts.device_id = 0
    cuda_opts.cudnn_conv_algo_search = 1  # HEURISTIC
    cuda_opts.gpu_mem_limit = ctypes.c_size_t(-1).value
    cuda_opts.arena_extend_strategy = 0
    cuda_opts.do_copy_in_default_stream = 1
    check(api, AppendCUDA(opts, ctypes.byref(cuda_opts)), "Append CUDA EP")
    sess = ctypes.c_void_p()
    print("CreateSession", onnx_path, flush=True)
    check(
        api,
        CreateSession(env, onnx_path.encode("utf-8"), opts, ctypes.byref(sess)),
        "CreateSession %s" % onnx_path,
    )
    print("CreateSession_ok", onnx_path, hex(sess.value or 0), flush=True)
    return sess


def resolve_paths(order):
    model_dir = os.environ.get("TTS_MODEL_DIR", "/models/matcha-kai-16k-e500")
    mapping = {
        "acoustic": os.path.join(model_dir, "model-steps-3.onnx"),
        "vocos": os.path.join(model_dir, "vocos-16khz-univ.onnx"),
        "tiny": "/tmp/tiny_conv.onnx",
    }
    names = [x.strip() for x in order.split(",") if x.strip()]
    paths = []
    for name in names:
        if name not in mapping:
            raise SystemExit("unknown model %s (acoustic|vocos|tiny)" % name)
        path = mapping[name]
        if name == "tiny" and not os.path.isfile(path):
            make_tiny_onnx(path)
        if not os.path.isfile(path):
            raise SystemExit("missing %s %s" % (name, path))
        paths.append((name, path))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", required=True, help="comma list: vocos,acoustic")
    args = ap.parse_args()

    disable_sherpa_trt()
    dump("A_base")
    api = load_ort_api()
    CreateEnv = _fn(
        api,
        IDX_CREATE_ENV,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    env = ctypes.c_void_p()
    check(api, CreateEnv(ORT_LOGGING_WARNING, b"session_order", ctypes.byref(env)), "CreateEnv")
    dump("A2_ort_env")
    keep = []
    letters = "BCDEFG"
    for i, (name, path) in enumerate(resolve_paths(args.order)):
        sess = create_cuda_session(api, env, path)
        keep.append(sess)
        dump("%s_%s" % (letters[i], name))
    print("CKPT_DONE order=%s" % args.order, flush=True)
    import time

    time.sleep(8)


if __name__ == "__main__":
    main()
