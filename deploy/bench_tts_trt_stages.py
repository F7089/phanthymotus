#!/usr/bin/env python3
"""Compare ORT CUDA vs TensorRT EP RSS for Melo FP32 (inside Jetson container).

Each backend runs in a **fresh subprocess** so RSS baselines stay clean.

Host:
  # 1) shape-infer (required for TRT EP on this Melo graph)
  docker cp deploy/shape_infer_melo_onnx.py phanthymotus-tts-melo:/tmp/
  docker exec -u 0 phanthymotus-tts-melo bash -lc \\
    'python3 -c "import onnx" 2>/dev/null || pip3 install -q onnx; \\
     python3 /tmp/shape_infer_melo_onnx.py \\
       --input /models/vits-melo-longanlingxin-openepd-nobert-44100-fp32/model.onnx \\
       --output /tmp/melo_fp32_shaped.onnx'

  # 2) stage bench
  docker cp deploy/bench_tts_trt_stages.py phanthymotus-tts-melo:/tmp/
  docker exec -u 0 -e MELO_ONNX_TRT=/tmp/melo_fp32_shaped.onnx \\
    phanthymotus-tts-melo python3 /tmp/bench_tts_trt_stages.py \\
    | tee ~/fanyi/wav_out/trt_stages.txt
"""
from __future__ import annotations

import json
import os
import subprocess
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


def _default_model(backend: str) -> Path:
    raw = Path(
        "/models/vits-melo-longanlingxin-openepd-nobert-44100-fp32/model.onnx"
    )
    if backend == "trt":
        shaped = Path(
            os.environ.get("MELO_ONNX_TRT", "/tmp/melo_fp32_shaped.onnx")
        )
        if shaped.is_file():
            return shaped
        # fall back; will likely fail with the known ConstantOfShape error
        return Path(os.environ.get("MELO_ONNX", str(raw)))
    return Path(os.environ.get("MELO_ONNX", str(raw)))


def run_one_backend(backend: str) -> dict:
    g2p = Path("/models/melo-openepd-g2p-assets")
    model = _default_model(backend)
    text = os.environ.get(
        "BENCH_TEXT", "你好，这是榜单模拟测试。今天天气怎么样？"
    )
    if not model.is_file():
        return {
            "backend": backend,
            "error": f"missing model {model}"
            + (
                " (run shape_infer_melo_onnx.py first for TRT)"
                if backend == "trt"
                else ""
            ),
        }

    os.environ["MELO_OPENEPD_DICT"] = str(g2p / "openepd_eng_dict.pickle")
    os.environ.setdefault("MELO_SKIP_HF_TOKENIZER", "1")
    if str(g2p / "vendor") not in sys.path:
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
        return {"backend": backend, "error": str(e), "model": str(model)}

    load_s = time.perf_counter() - t0
    after = rss_mb()
    print(
        f"  after_session  rss={after:.1f} (+{after-before:.1f}) "
        f"load={load_s:.2f}s active={sess.get_providers()}"
    )

    rows = [
        {
            "backend": backend,
            "stage": "session",
            "rss_mib": round(after, 1),
            "delta": round(after - before, 1),
            "load_s": round(load_s, 2),
            "active": sess.get_providers(),
            "model": str(model),
        }
    ]
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
        rows.append(
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
    return {"backend": backend, "ok": True, "rows": rows}


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        result = run_one_backend(sys.argv[2])
        Path("/tmp/trt_worker_result.json").write_text(
            json.dumps(result, ensure_ascii=False) + "\n"
        )
        if "error" in result and not result.get("ok"):
            sys.exit(2)
        return

    backends = os.environ.get("BENCH_BACKENDS", "cuda,trt").split(",")
    summary = []
    for backend in [b.strip() for b in backends if b.strip()]:
        print(f"\n######## spawn worker backend={backend} ########")
        env = os.environ.copy()
        # Force unbuffered child logs
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.run(
            [sys.executable, __file__, "--worker", backend],
            env=env,
            check=False,
        )
        result_path = Path("/tmp/trt_worker_result.json")
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("rows"):
                summary.extend(result["rows"])
            else:
                summary.append(result)
            print(f"worker_exit={p.returncode} backend={backend}")
        else:
            summary.append(
                {
                    "backend": backend,
                    "error": f"no worker result, exit={p.returncode}",
                }
            )

    out = Path("/tmp/trt_stages.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
