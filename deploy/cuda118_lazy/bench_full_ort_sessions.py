#!/usr/bin/env python3
"""Matcha acoustic + Vocos CUDA EP sessions on the CUDA 11.8 ORT wheel.

Fresh process (do not run after tiny in the same container). Same two ONNX
files sherpa Matcha uses. No JP5 11.4 sherpa wheel.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/opt/cuda118_lazy")
from bench_tiny_session import dump, find_cgroup, print_cuda_libs  # noqa: E402


def _session(onnx_path, tag):
    import numpy as np
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    cuda_opts = {
        "device_id": 0,
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": 1,
    }
    print("CreateSession", tag, onnx_path, flush=True)
    t0 = time.monotonic()
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=[("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"],
    )
    print(
        "CreateSession_ok",
        tag,
        "wall_s=%.2f" % (time.monotonic() - t0),
        "providers",
        sess.get_providers(),
        flush=True,
    )
    dump(tag)
    return sess


def _first_infer(ac, vocos):
    import numpy as np

    feeds = {}
    for inp in ac.get_inputs():
        name = inp.name
        ty = inp.type or ""
        shape = []
        for d in inp.shape or []:
            if isinstance(d, int) and d > 0:
                shape.append(d)
            else:
                shape.append(1)
        if len(shape) < 1:
            shape = [1]
        if "int64" in ty:
            arr = np.ones(shape, dtype=np.int64)
        elif "int" in ty:
            arr = np.ones(shape, dtype=np.int32)
        else:
            arr = np.ones(shape, dtype=np.float32)
        if name == "x" and arr.ndim == 2:
            arr = np.ones((1, 16), dtype=arr.dtype)
        if name == "x_length":
            arr = np.asarray([16], dtype=arr.dtype)
        if name == "noise_scale":
            arr = np.asarray([0.667], dtype=np.float32)
        feeds[name] = arr
    t0 = time.monotonic()
    outs = ac.run(None, feeds)
    t_ac = time.monotonic() - t0
    print("acoustic_infer_s=%.3f out_shapes=%s" % (t_ac, [np.asarray(o).shape for o in outs]), flush=True)
    mel = None
    for o, meta in zip(outs, ac.get_outputs()):
        if "mel" in meta.name:
            mel = np.asarray(o, dtype=np.float32)
            break
    if mel is None:
        mel = np.asarray(outs[0], dtype=np.float32)
    if mel.ndim == 2:
        mel = mel[None, ...]
    if mel.ndim == 3 and mel.shape[1] != 80 and mel.shape[2] == 80:
        mel = np.transpose(mel, (0, 2, 1))
    vin = vocos.get_inputs()[0].name
    t1 = time.monotonic()
    vout = vocos.run(None, {vin: np.ascontiguousarray(mel, dtype=np.float32)})
    t_v = time.monotonic() - t1
    wall = t_ac + t_v
    print("vocos_infer_s=%.3f vocos_out=%s TTFT_s=%.3f" % (t_v, [np.asarray(o).shape for o in vout], wall), flush=True)
    audio_s = 0.5
    n = int(np.asarray(vout[0]).size) if vout else 0
    if n > 1000:
        audio_s = max(0.05, n / 16000.0)
    print("synth_RTF=%.4f audio_s_est=%.3f" % (wall / audio_s, audio_s), flush=True)


def main():
    model_dir = os.environ.get("TTS_MODEL_DIR", "/models/matcha-kai-16k-e500")
    ac_path = os.path.join(model_dir, "model-steps-3.onnx")
    voc_path = os.path.join(model_dir, "vocos-16khz-univ.onnx")
    if not os.path.isfile(ac_path) or not os.path.isfile(voc_path):
        raise SystemExit("missing ONNX under %s" % model_dir)
    dump("F0_base")
    import onnxruntime as ort

    print("onnxruntime", ort.__version__, ort.get_available_providers(), flush=True)
    ac = _session(ac_path, "F1_acoustic")
    vocos = _session(voc_path, "F2_vocos")
    _first_infer(ac, vocos)
    dump("F3_after_infer")
    print_cuda_libs()
    u, p = find_cgroup()
    print(
        "FULL_ORT_CUDA_DONE cgroup_usage_MB=%s cgroup_max_MB=%s"
        % (
            ("%.1f" % u) if u is not None else "NA",
            ("%.1f" % p) if p is not None else "NA",
        ),
        flush=True,
    )
    time.sleep(5)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        time.sleep(3)
        sys.exit(1)
