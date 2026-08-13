#!/usr/bin/env python3
"""Minimal Melo FP32 CUDA session for Nsight CUDA-memory profiling.

Stages print RSS only; CUDA peak comes from nsys --cuda-memory-usage.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return -1.0


def main() -> None:
    g2p = Path("/models/melo-openepd-g2p-assets")
    model = Path(
        "/models/vits-melo-longanlingxin-openepd-nobert-44100-fp32/model.onnx"
    )
    text = os.environ.get(
        "BENCH_TEXT", "你好，这是榜单模拟测试。今天天气怎么样？"
    )
    os.environ["MELO_OPENEPD_DICT"] = str(g2p / "openepd_eng_dict.pickle")
    os.environ.setdefault("MELO_SKIP_HF_TOKENIZER", "1")
    sys.path.insert(0, str(g2p / "vendor"))

    import onnxruntime as ort
    from melo_g2p.encode import encode_phones_tones

    print(f"pid={os.getpid()} rss_start={rss_mb():.1f}", flush=True)
    with open(g2p / "config.json", encoding="utf-8") as f:
        meta = json.load(f)
    phone_ids, tone_ids = encode_phones_tones(
        text,
        list(meta["symbols"]),
        add_blank=bool(meta.get("add_blank", True)),
        language=str(meta.get("language") or "ZH_MIX_EN"),
    )
    feed = {
        "x": np.array([phone_ids], dtype=np.int64),
        "x_lengths": np.array([len(phone_ids)], dtype=np.int64),
        "tones": np.array([tone_ids], dtype=np.int64),
        "sid": np.array([0], dtype=np.int64),
        "noise_scale": np.array([0.6], dtype=np.float32),
        "length_scale": np.array([1.0 / 0.9], dtype=np.float32),
        "noise_scale_w": np.array([0.8], dtype=np.float32),
    }
    print(f"g2p_ready rss={rss_mb():.1f}", flush=True)

    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "cudnn_conv_use_max_workspace": "1",
                "cudnn_conv_algo_search": "HEURISTIC",
            },
        ),
        "CPUExecutionProvider",
    ]
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model), sess_options=so, providers=providers)
    print(
        f"after_session rss={rss_mb():.1f} load={time.perf_counter()-t0:.2f}s "
        f"providers={sess.get_providers()}",
        flush=True,
    )
    for i in range(2):
        t1 = time.perf_counter()
        outs = sess.run(None, feed)
        dt = time.perf_counter() - t1
        audio = np.asarray(outs[0]).squeeze()
        dur = float(audio.size) / 44100.0
        print(
            f"after_run{i+1} rss={rss_mb():.1f} ort={dt:.3f}s rtf={dt/dur:.3f}",
            flush=True,
        )
    # Hold briefly so nsys captures steady state
    time.sleep(1.0)
    print(f"done rss={rss_mb():.1f}", flush=True)


if __name__ == "__main__":
    main()
