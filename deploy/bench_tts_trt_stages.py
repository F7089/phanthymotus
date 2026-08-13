#!/usr/bin/env python3
"""Compare ORT CUDA vs TensorRT EP RSS for Melo FP32 (inside Jetson container).

Fresh process stages per backend:
  before_session / after_session / after_run1 / after_run2

Host:
  docker cp deploy/bench_tts_trt_stages.py phanthymotus-tts-melo:/tmp/
  docker exec -u 0 phanthymotus-tts-melo python3 /tmp/bench_tts_trt_stages.py \\
    | tee ~/fanyi/wav_out/trt_stages.txt
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


def build_providers(backend: str):
    import onnxruntime as ort

    available = ort.get_available_providers()
    cuda_opts = {
        "device_id": 0,
        "cudnn_conv_use_max_workspace": "1",
        "cudnn_conv_algo_search": "HEURISTIC",
    }
    if backend == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"no CUDA EP: {available}")
        return ort, [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]

    if backend == "trt":
        if "TensorrtExecutionProvider" not in available:
            raise RuntimeError(f"no TensorRT EP: {available}")
        cache = "/tmp/ort_trt_cache_melo_bench"
        os.makedirs(cache, exist_ok=True)
        ws_mb = int(os.environ.get("TTS_ORT_TRT_WORKSPACE_MB", "512"))
        trt_opts = {
            "device_id": 0,
            "trt_max_workspace_size": ws_mb * 1024 * 1024,
            "trt_fp16_enable": False,  # keep FP32
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache,
        }
        providers = [("TensorrtExecutionProvider", trt_opts)]
        if "CUDAExecutionProvider" in available:
            providers.append(("CUDAExecutionProvider", cuda_opts))
        providers.append("CPUExecutionProvider")
        return ort, providers

    raise ValueError(backend)


def main() -> None:
    g2p = Path("/models/melo-openepd-g2p-assets")
    model = Path(
        "/models/vits-melo-longanlingxin-openepd-nobert-44100-fp32/model.onnx"
    )
    text = os.environ.get(
        "BENCH_TEXT", "你好，这是榜单模拟测试。今天天气怎么样？"
    )
    backends = ["cuda", "trt"]
    if not model.is_file():
        raise SystemExit(f"missing {model}")

    os.environ["MELO_OPENEPD_DICT"] = str(g2p / "openepd_eng_dict.pickle")
    os.environ.setdefault("MELO_SKIP_HF_TOKENIZER", "1")
    sys.path.insert(0, str(g2p / "vendor"))

    import onnxruntime as ort
    from melo_g2p.encode import encode_phones_tones

    print("available_providers=", ort.get_available_providers())
    print(f"model={model} size_mb={model.stat().st_size/1024/1024:.1f}")

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
    print(f"g2p_ready rss={rss_mb():.1f} phones={len(phone_ids)}")

    summary = []
    for backend in backends:
        import gc

        gc.collect()
        ort_mod, providers = build_providers(backend)
        before = rss_mb()
        print(f"\n== backend={backend} ==")
        print(f"  providers_cfg={providers}")
        print(f"  before_session rss={before:.1f}")

        so = ort_mod.SessionOptions()
        so.enable_cpu_mem_arena = False
        so.intra_op_num_threads = 2
        so.inter_op_num_threads = 1
        so.execution_mode = ort_mod.ExecutionMode.ORT_SEQUENTIAL
        # Help see EP assignment if needed
        if os.environ.get("TTS_ORT_VERBOSE", "0") == "1":
            so.log_severity_level = 0
            so.log_verbosity_level = 1

        t0 = time.perf_counter()
        try:
            sess = ort_mod.InferenceSession(
                str(model), sess_options=so, providers=providers
            )
        except Exception as e:
            print(f"  ERROR creating session: {e}")
            summary.append({"backend": backend, "error": str(e)})
            continue
        load_s = time.perf_counter() - t0
        after = rss_mb()
        print(
            f"  after_session  rss={after:.1f} (+{after-before:.1f}) "
            f"load={load_s:.2f}s active={sess.get_providers()}"
        )

        for i in range(2):
            t1 = time.perf_counter()
            outs = sess.run(None, feed)
            dt = time.perf_counter() - t1
            audio = np.asarray(outs[0]).squeeze()
            dur = float(audio.size) / 44100.0
            now = rss_mb()
            print(
                f"  after_run{i+1}    rss={now:.1f} (+{now-before:.1f}) "
                f"ort={dt:.3f}s rtf={dt/dur:.3f}"
            )
            summary.append(
                {
                    "backend": backend,
                    "stage": f"run{i+1}",
                    "rss_mib": round(now, 1),
                    "delta": round(now - before, 1),
                    "ort_s": round(dt, 3),
                    "rtf": round(dt / dur, 3),
                    "active": sess.get_providers(),
                }
            )

        summary.append(
            {
                "backend": backend,
                "stage": "session",
                "rss_mib": round(after, 1),
                "delta": round(after - before, 1),
                "load_s": round(load_s, 2),
                "active": sess.get_providers(),
            }
        )
        del sess, outs
        gc.collect()

    out = Path("/tmp/trt_stages.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
