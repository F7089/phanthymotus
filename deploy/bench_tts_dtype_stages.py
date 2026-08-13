#!/usr/bin/env python3
"""Stage-wise RSS for Melo FP32 vs FP16 (run inside Jetson TTS container).

Stages per dtype:
  1) after imports / G2P encode ready
  2) after InferenceSession create
  3) after run #1
  4) after run #2

Usage (on Jetson host):
  docker cp deploy/bench_tts_dtype_stages.py phanthymotus-tts-melo:/tmp/
  docker exec -u 0 phanthymotus-tts-melo python3 /tmp/bench_tts_dtype_stages.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def rss_mb() -> float:
    # VmRSS kB
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return -1.0


def main() -> None:
    g2p = Path("/models/melo-openepd-g2p-assets")
    text = os.environ.get(
        "BENCH_TEXT", "你好，这是榜单模拟测试。今天天气怎么样？"
    )
    models = {
        "fp32": Path(
            "/models/vits-melo-longanlingxin-openepd-nobert-44100-fp32/model.onnx"
        ),
        "fp16": Path(
            "/models/vits-melo-longanlingxin-openepd-nobert-44100-fp16/model.onnx"
        ),
    }

    rows = []
    print(f"pid={os.getpid()} text={text!r}")
    print(f"stage0_import_start rss={rss_mb():.1f} MiB")

    os.environ["MELO_OPENEPD_DICT"] = str(g2p / "openepd_eng_dict.pickle")
    os.environ.setdefault("MELO_SKIP_HF_TOKENIZER", "1")
    sys.path.insert(0, str(g2p / "vendor"))

    import onnxruntime as ort
    from melo_g2p.encode import encode_phones_tones

    with open(g2p / "config.json", encoding="utf-8") as f:
        meta = json.load(f)
    symbols = list(meta["symbols"])
    add_blank = bool(meta.get("add_blank", True))
    language = str(meta.get("language") or "ZH_MIX_EN")

    t0 = time.perf_counter()
    phone_ids, tone_ids = encode_phones_tones(
        text, symbols, add_blank=add_blank, language=language
    )
    g2p_s = time.perf_counter() - t0
    rss_g2p = rss_mb()
    print(f"stage1_g2p_ready rss={rss_g2p:.1f} MiB g2p={g2p_s:.3f}s phones={len(phone_ids)}")

    speed = 0.9
    feed = {
        "x": np.array([phone_ids], dtype=np.int64),
        "x_lengths": np.array([len(phone_ids)], dtype=np.int64),
        "tones": np.array([tone_ids], dtype=np.int64),
        "sid": np.array([0], dtype=np.int64),
        "noise_scale": np.array([0.6], dtype=np.float32),
        "length_scale": np.array([1.0 / speed], dtype=np.float32),
        "noise_scale_w": np.array([0.8], dtype=np.float32),
    }

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "cudnn_conv_use_max_workspace": os.environ.get(
                    "TTS_ORT_CUDNN_MAX_WORKSPACE", "1"
                ),
                "cudnn_conv_algo_search": "HEURISTIC",
            },
        ),
        "CPUExecutionProvider",
    ]

    for dtype, path in models.items():
        if not path.is_file():
            print(f"SKIP {dtype}: missing {path}")
            continue
        # Drop previous session aggressively
        import gc

        gc.collect()
        before = rss_mb()
        print(f"\n== {dtype} == onnx={path} size_mb={path.stat().st_size/1024/1024:.1f}")
        print(f"  before_session rss={before:.1f}")

        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.enable_mem_pattern = os.environ.get("TTS_ORT_MEM_PATTERN", "1") != "0"
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        t1 = time.perf_counter()
        sess = ort.InferenceSession(str(path), sess_options=so, providers=providers)
        load_s = time.perf_counter() - t1
        after_sess = rss_mb()
        print(
            f"  after_session  rss={after_sess:.1f} (+{after_sess-before:.1f}) "
            f"load={load_s:.2f}s providers={sess.get_providers()}"
        )

        times = []
        for i in range(2):
            t2 = time.perf_counter()
            outs = sess.run(None, feed)
            dt = time.perf_counter() - t2
            times.append(dt)
            audio = np.asarray(outs[0]).squeeze()
            dur = float(audio.size) / 44100.0
            now = rss_mb()
            print(
                f"  after_run{i+1}    rss={now:.1f} (+{now-before:.1f} from pre) "
                f"ort={dt:.3f}s audio={dur:.3f}s rtf={dt/dur:.3f}"
            )
            rows.append(
                {
                    "dtype": dtype,
                    "stage": f"after_run{i+1}",
                    "rss_mib": round(now, 1),
                    "delta_from_pre_session": round(now - before, 1),
                    "ort_s": round(dt, 3),
                    "rtf": round(dt / dur, 3) if dur else None,
                }
            )

        rows.append(
            {
                "dtype": dtype,
                "stage": "after_session",
                "rss_mib": round(after_sess, 1),
                "delta_from_pre_session": round(after_sess - before, 1),
                "load_s": round(load_s, 2),
            }
        )
        del sess, outs, audio
        gc.collect()

    out = Path("/tmp/dtype_stages.json")
    summary = {
        "g2p_rss_mib": round(rss_g2p, 1),
        "providers_avail": ort.get_available_providers(),
        "rows": rows,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
