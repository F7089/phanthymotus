#!/usr/bin/env python3
"""A/B: Melo FP32 CUDA with/without session.disable_prepacking (fresh subprocesses).

Inside Jetson container:
  python3 /tmp/bench_tts_prepack_mem.py | tee /tmp/prepack_mem.txt
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


def heap_anon_mb() -> tuple[float, float]:
    """Best-effort [heap] + anonymous Rss from smaps (MiB)."""
    heap = anon = 0.0
    name = ""
    try:
        text = Path("/proc/self/smaps").read_text()
    except OSError:
        return -1.0, -1.0
    for line in text.splitlines():
        # Header: "start-end perms offset dev inode pathname"
        if "-" in line[:40] and " " in line and not line[0].isalpha():
            parts = line.split()
            name = parts[-1] if len(parts) >= 6 else ""
            continue
        if line.startswith("Rss:"):
            kb = int(line.split()[1])
            if name == "[heap]":
                heap += kb / 1024.0
            elif name in ("", "[anon]") or name.startswith("[anon:"):
                # anonymous mappings often have empty pathname
                if name == "[heap]":
                    continue
                # file-backed have a path starting with /
                if name.startswith("/"):
                    continue
                anon += kb / 1024.0
    return heap, anon


def cuda_used_mb() -> float | None:
    try:
        import ctypes

        lib = ctypes.CDLL("libcudart.so")
        free = ctypes.c_size_t()
        total = ctypes.c_size_t()
        # cudaMemGetInfo(free, total)
        rc = lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if rc != 0:
            return None
        return (total.value - free.value) / (1024.0 * 1024.0)
    except Exception:
        return None


def run_case(disable_prepack: bool) -> dict:
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

    tag = "prepack_off" if disable_prepack else "prepack_on"
    print(f"\n== {tag} ==")
    stages = {}
    stages["g2p"] = {
        "rss": round(rss_mb(), 1),
        "cuda_used": cuda_used_mb(),
    }
    print(f"  g2p          rss={stages['g2p']['rss']} cuda_used={stages['g2p']['cuda_used']}")

    cuda_opts = {
        "device_id": 0,
        "cudnn_conv_use_max_workspace": "1",
        "cudnn_conv_algo_search": "HEURISTIC",
    }
    providers = [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = True
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if disable_prepack:
        so.add_session_config_entry("session.disable_prepacking", "1")

    before = rss_mb()
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(model), sess_options=so, providers=providers)
    load_s = time.perf_counter() - t0
    after = rss_mb()
    heap, anon = heap_anon_mb()
    stages["session"] = {
        "rss": round(after, 1),
        "delta": round(after - before, 1),
        "load_s": round(load_s, 2),
        "heap": round(heap, 1),
        "anon": round(anon, 1),
        "cuda_used": cuda_used_mb(),
        "active": sess.get_providers(),
    }
    print(
        f"  after_session rss={after:.1f} (+{after-before:.1f}) "
        f"heap≈{heap:.0f} anon≈{anon:.0f} cuda_used={stages['session']['cuda_used']} "
        f"load={load_s:.2f}s"
    )

    for i in range(2):
        t1 = time.perf_counter()
        outs = sess.run(None, feed)
        dt = time.perf_counter() - t1
        audio = np.asarray(outs[0]).squeeze()
        dur = float(audio.size) / 44100.0
        now = rss_mb()
        heap, anon = heap_anon_mb()
        stages[f"run{i+1}"] = {
            "rss": round(now, 1),
            "delta": round(now - before, 1),
            "ort_s": round(dt, 3),
            "rtf": round(dt / dur, 3),
            "heap": round(heap, 1),
            "anon": round(anon, 1),
            "cuda_used": cuda_used_mb(),
        }
        print(
            f"  after_run{i+1}   rss={now:.1f} (+{now-before:.1f}) "
            f"heap≈{heap:.0f} anon≈{anon:.0f} "
            f"ort={dt:.3f}s rtf={dt/dur:.3f} cuda_used={stages[f'run{i+1}']['cuda_used']}"
        )

    return {"case": tag, "disable_prepacking": disable_prepack, "stages": stages}


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        disable = sys.argv[2] == "1"
        result = run_case(disable)
        Path("/tmp/prepack_worker.json").write_text(
            json.dumps(result, ensure_ascii=False) + "\n"
        )
        return

    summary = []
    for flag in ("0", "1"):
        print(f"\n######## spawn disable_prepacking={flag} ########")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(
            [sys.executable, __file__, "--worker", flag], env=env, check=False
        )
        p = Path("/tmp/prepack_worker.json")
        if p.is_file():
            summary.append(json.loads(p.read_text()))

    out = Path("/tmp/prepack_mem.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {out}")
    if len(summary) == 2:
        a = summary[0]["stages"].get("run2", {})
        b = summary[1]["stages"].get("run2", {})
        print(
            f"\nDELTA run2: rss {a.get('rss')} -> {b.get('rss')}  "
            f"rtf {a.get('rtf')} -> {b.get('rtf')}  "
            f"(prepack_on -> prepack_off)"
        )


if __name__ == "__main__":
    main()
