#!/usr/bin/env python3
"""A/B host RSS: plain ONNX vs ORT-format + model-bytes initializers (CUDA EP).

Fresh subprocess per case. Convert first:
  python3 /tmp/convert_melo_ort_format.py
  python3 /tmp/bench_tts_ortfmt_mem.py
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


def run_case(case: str) -> dict:
    g2p = Path("/models/melo-openepd-g2p-assets")
    onnx_dir = Path(
        "/models/vits-melo-longanlingxin-openepd-nobert-44100-fp32"
    )
    ort_dir = Path(os.environ.get("MELO_ORT_DIR", "/tmp/melo_fp32_ort"))
    text = os.environ.get(
        "BENCH_TEXT", "你好，这是榜单模拟测试。今天天气怎么样？"
    )
    os.environ["MELO_OPENEPD_DICT"] = str(g2p / "openepd_eng_dict.pickle")
    os.environ.setdefault("MELO_SKIP_HF_TOKENIZER", "1")
    sys.path.insert(0, str(g2p / "vendor"))

    import onnxruntime as ort
    from melo_g2p.encode import encode_phones_tones

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
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    model_bytes = None
    model_arg: str | bytes
    if case == "onnx":
        model_arg = str(onnx_dir / "model.onnx")
    elif case == "ort_path":
        p = ort_dir / "model.ort"
        if not p.is_file():
            return {"case": case, "error": f"missing {p}; run convert first"}
        so.add_session_config_entry("session.load_model_format", "ORT")
        model_arg = str(p)
    elif case == "ort_bytes_direct":
        # Safer: use buffer directly but still copy initializers (no for_initializers).
        p = ort_dir / "model.ort"
        if not p.is_file():
            return {"case": case, "error": f"missing {p}; run convert first"}
        so.add_session_config_entry("session.load_model_format", "ORT")
        so.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
        so.add_session_config_entry("session.disable_prepacking", "1")
        with open(p, "rb") as f:
            model_bytes = f.read()
        model_arg = model_bytes
    elif case == "ort_mmap":
        # Path load + mmap (if this ORT build supports the config key).
        p = ort_dir / "model.ort"
        if not p.is_file():
            return {"case": case, "error": f"missing {p}; run convert first"}
        so.add_session_config_entry("session.load_model_format", "ORT")
        so.add_session_config_entry("session.use_memory_mapped_ort_model", "1")
        so.add_session_config_entry("session.disable_prepacking", "1")
        model_arg = str(p)
    elif case == "ort_bytes":
        # Known to SIGSEGV on Jetson ORT 1.23 + CUDA EP with Melo; keep opt-in only.
        if os.environ.get("TTS_ORT_ALLOW_BYTES_INIT", "0") != "1":
            return {
                "case": case,
                "error": "skipped: use_ort_model_bytes_for_initializers SIGSEGV "
                "on this ORT/CUDA; set TTS_ORT_ALLOW_BYTES_INIT=1 to force",
            }
        p = ort_dir / "model.ort"
        if not p.is_file():
            return {"case": case, "error": f"missing {p}; run convert first"}
        so.add_session_config_entry("session.load_model_format", "ORT")
        so.add_session_config_entry("session.use_ort_model_bytes_directly", "1")
        so.add_session_config_entry(
            "session.use_ort_model_bytes_for_initializers", "1"
        )
        so.add_session_config_entry("session.disable_prepacking", "1")
        with open(p, "rb") as f:
            model_bytes = f.read()
        model_arg = model_bytes
    elif case == "onnx_external":
        p = ort_dir / "model.with_external.onnx"
        if not p.is_file():
            return {"case": case, "error": f"missing {p}; convert --external"}
        so.add_session_config_entry("session.disable_prepacking", "1")
        model_arg = str(p)
    else:
        return {"case": case, "error": f"unknown case {case}"}

    print(f"\n== case={case} ==")
    before = rss_mb()
    print(f"  before_session rss={before:.1f}")
    t0 = time.perf_counter()
    try:
        sess = ort.InferenceSession(model_arg, sess_options=so, providers=providers)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"case": case, "error": str(e)}
    load_s = time.perf_counter() - t0
    after = rss_mb()
    print(
        f"  after_session  rss={after:.1f} (+{after-before:.1f}) "
        f"load={load_s:.2f}s active={sess.get_providers()}"
    )

    rows = {
        "case": case,
        "session_rss": round(after, 1),
        "session_delta": round(after - before, 1),
        "load_s": round(load_s, 2),
    }
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
        rows[f"run{i+1}_rss"] = round(now, 1)
        rows[f"run{i+1}_rtf"] = round(dt / dur, 3)
    # keep model_bytes referenced until here
    _ = model_bytes
    return rows


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        result = run_case(sys.argv[2])
        Path("/tmp/ortfmt_worker.json").write_text(
            json.dumps(result, ensure_ascii=False) + "\n"
        )
        return

    # ort_bytes (for_initializers) segfaults on Jetson ORT1.23+CUDA; excluded by default.
    cases = os.environ.get(
        "BENCH_CASES", "onnx,ort_path,ort_bytes_direct,ort_mmap,onnx_external"
    ).split(",")
    summary = []
    for case in [c.strip() for c in cases if c.strip()]:
        print(f"\n######## spawn {case} ########")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(
            [sys.executable, __file__, "--worker", case], env=env, check=False
        )
        p = Path("/tmp/ortfmt_worker.json")
        if p.is_file():
            summary.append(json.loads(p.read_text()))

    out = Path("/tmp/ortfmt_mem.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {out}")
    print("\nSUMMARY run2_rss:")
    for r in summary:
        if "error" in r:
            print(f"  {r['case']}: ERROR {r['error'][:120]}")
        else:
            print(
                f"  {r['case']}: rss={r.get('run2_rss')} rtf={r.get('run2_rtf')}"
            )


if __name__ == "__main__":
    main()
