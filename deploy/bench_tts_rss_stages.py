#!/usr/bin/env python3
"""Stage RSS for Melo FP32 optimization direction (no finetune / no re-export).

Stages (fresh process):
  1 start
  2 after_numpy_ort_import
  3 after_g2p_ready          ← no ORT session yet (host base)
  4 after_session
  5 after_run1 / after_run2

Cases (env BENCH_CASES, default both):
  onnx           /models/.../model.onnx
  onnx_external  /tmp/melo_fp32_ort/model.with_external.onnx  (if present)

Host:
  docker cp deploy/bench_tts_rss_stages.py phanthymotus-tts-melo:/tmp/
  docker exec -u 0 phanthymotus-tts-melo python3 /tmp/bench_tts_rss_stages.py \\
    | tee ~/fanyi/wav_out/rss_stages.txt
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
    stages: dict = {"case": case}
    stages["start"] = round(rss_mb(), 1)
    print(f"\n== case={case} ==")
    print(f"  start                    rss={stages['start']}")

    g2p = Path("/models/melo-openepd-g2p-assets")
    onnx_dir = Path(
        "/models/vits-melo-longanlingxin-openepd-nobert-44100-fp32"
    )
    ext_dir = Path(os.environ.get("MELO_ORT_DIR", "/tmp/melo_fp32_ort"))
    text = os.environ.get(
        "BENCH_TEXT", "你好，这是榜单模拟测试。今天天气怎么样？"
    )

    import onnxruntime as ort

    stages["after_numpy_ort_import"] = round(rss_mb(), 1)
    print(f"  after_numpy_ort_import   rss={stages['after_numpy_ort_import']}")

    oedb = g2p / "openepd_eng_dict.oedb"
    pkl = g2p / "openepd_eng_dict.pickle"
    os.environ["MELO_OPENEPD_DICT"] = str(oedb if oedb.is_file() else pkl)
    ckpt = g2p / "vendor" / "melo_g2p" / "text" / "checkpoint20.npz"
    if ckpt.is_file():
        os.environ["MELO_G2P_OOV_CKPT"] = str(ckpt)
    os.environ.setdefault("MELO_SKIP_HF_TOKENIZER", "1")
    # Prefer image slim sources if present (git overlay).
    slim = Path("/work/melo_g2p_slim")
    text = g2p / "vendor" / "melo_g2p" / "text"
    if slim.is_dir() and text.is_dir():
        import shutil

        for name in ("english.py", "slim_g2p_oov.py", "openepd_compact.py"):
            src = slim / name
            if src.is_file():
                shutil.copy2(src, text / name)
    sys.path.insert(0, str(g2p / "vendor"))
    # Drop cached english if any
    for key in list(sys.modules):
        if key.startswith("melo_g2p"):
            sys.modules.pop(key, None)
    from melo_g2p.encode import encode_phones_tones
    print(f"  openepd={os.environ['MELO_OPENEPD_DICT']}")

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
    stages["after_g2p_ready"] = round(rss_mb(), 1)
    print(
        f"  after_g2p_ready          rss={stages['after_g2p_ready']}  "
        f"(NO session yet; host base)"
    )

    if case == "onnx":
        model = str(onnx_dir / "model.onnx")
    elif case == "onnx_external":
        model = str(ext_dir / "model.with_external.onnx")
        if not Path(model).is_file():
            return {
                "case": case,
                "error": f"missing {model}; run convert --external once",
                "stages": stages,
            }
    else:
        return {"case": case, "error": f"unknown case {case}", "stages": stages}

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
    sess = ort.InferenceSession(model, sess_options=so, providers=providers)
    load_s = time.perf_counter() - t0
    stages["after_session"] = round(rss_mb(), 1)
    stages["session_delta"] = round(
        stages["after_session"] - stages["after_g2p_ready"], 1
    )
    print(
        f"  after_session            rss={stages['after_session']} "
        f"(+{stages['session_delta']} vs g2p) load={load_s:.2f}s "
        f"providers={sess.get_providers()}"
    )

    for i in range(2):
        t1 = time.perf_counter()
        outs = sess.run(None, feed)
        dt = time.perf_counter() - t1
        audio = np.asarray(outs[0]).squeeze()
        dur = float(audio.size) / 44100.0
        key = f"after_run{i+1}"
        stages[key] = round(rss_mb(), 1)
        stages[f"run{i+1}_rtf"] = round(dt / dur, 3)
        print(
            f"  {key}              rss={stages[key]} "
            f"(+{stages[key]-stages['after_g2p_ready']:.1f} vs g2p) "
            f"ort={dt:.3f}s rtf={stages[f'run{i+1}_rtf']}"
        )
    stages["model"] = model
    return {"case": case, "ok": True, "stages": stages}


def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        result = run_case(sys.argv[2])
        Path("/tmp/rss_stages_worker.json").write_text(
            json.dumps(result, ensure_ascii=False) + "\n"
        )
        return

    cases = os.environ.get("BENCH_CASES", "onnx,onnx_external").split(",")
    summary = []
    for case in [c.strip() for c in cases if c.strip()]:
        print(f"\n######## spawn {case} ########")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        subprocess.run(
            [sys.executable, __file__, "--worker", case], env=env, check=False
        )
        p = Path("/tmp/rss_stages_worker.json")
        if p.is_file():
            summary.append(json.loads(p.read_text()))

    out = Path("/tmp/rss_stages.json")
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(f"\nwrote {out}")
    print("\nSUMMARY (optimization direction):")
    print("  host_base = after_g2p_ready (no ORT session)")
    print("  session_cost = after_session - after_g2p_ready")
    print("  steady = after_run2")
    for r in summary:
        st = r.get("stages") or {}
        if r.get("error"):
            print(f"  {r['case']}: ERROR {r['error'][:100]}")
            continue
        print(
            f"  {r['case']}: host_base={st.get('after_g2p_ready')}  "
            f"session={st.get('after_session')} "
            f"(+{st.get('session_delta')})  "
            f"run2={st.get('after_run2')} rtf={st.get('run2_rtf')}"
        )


if __name__ == "__main__":
    main()
