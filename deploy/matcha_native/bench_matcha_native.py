#!/usr/bin/env python3
"""Native Matcha baseline: official frontend + CUDA provider. No OpenEPD, no MCP.

Metrics (same spirit as Melo RSS stages):
  1) zh-en mix synthesizes
  2) sample_rate == 16000
  3) RTF
  4) host_base RSS   (sherpa imported, no OfflineTts yet)
  5) after_session RSS
  6) warmup / run2 RSS
  7) CUDA EP actually used

Compare to Melo external slim: RTF ≈ 0.082, run2 RSS ≈ 1177MB.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

# Official long zh-en sentence from k2-fsa generate_samples.py / docs.
OFFICIAL_MIX = (
    "我最近在学习machine learning，希望能够在未来的artificial intelligence领域有所建树。"
    "在这次vocation中，我们计划去Paris欣赏埃菲尔铁塔和卢浮宫的美景。"
    "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。"
    "开始数字测试。2025年12月4号，拨打110或者189202512043。123456块钱。"
    "在这个快速发展的时代，人工智能技术正在改变我们的生活方式。"
    "语音合成作为人工智能的重要应用之一，让机器能够用自然流畅的语音与人类进行交流。"
)

# Same sentence as Melo RSS stages, for RTF/RSS apples-to-apples.
MELO_PARITY = (
    "请你先打开 WiFi 连上办公室网络，确认信号稳定之后，再继续下载这次的系统更新包。"
)

WARMUP_ZH = "你好，这是测试。"
WARMUP_EN = "hello, this is a test."


def rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return -1.0


def gpu_mem_mb() -> float | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def write_wav(path: Path, samples, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        buf = bytearray(n * 2)
        for i, x in enumerate(samples):
            v = int(max(-1.0, min(1.0, float(x))) * 32767.0)
            buf[i * 2] = v & 0xFF
            buf[i * 2 + 1] = (v >> 8) & 0xFF
        w.writeframes(bytes(buf))


def parse_cuda_from_logs(text: str) -> dict:
    low = text.lower()
    hits = {
        "cuda_execution_provider": (
            "cudaexecutionprovider" in low
            or "cuda execution provider" in low
            or "provider: cuda" in low
            or "provider=cuda" in low
        ),
        "please_compile_gpu": "sherpa_onnx_enable_gpu" in low
        or ("please compile" in low and "gpu" in low),
        "fallback_cpu": "got provider: cpu" in low or "provider: cpu" in low,
    }
    return hits


def generate(tts, text: str):
    try:
        return tts.generate(text, sid=0, speed=1.0)
    except TypeError:
        import sherpa_onnx

        cfg = sherpa_onnx.GenerationConfig()
        cfg.sid = 0
        cfg.speed = 1.0
        return tts.generate(text, cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "MATCHA_MODEL_DIR",
            str(Path(__file__).resolve().parent / "models"),
        ),
    )
    parser.add_argument("--provider", default=os.environ.get("MATCHA_PROVIDER", "cuda"))
    parser.add_argument("--num-threads", type=int, default=int(os.environ.get("MATCHA_NUM_THREADS", "2")))
    parser.add_argument(
        "--out-dir",
        default=os.environ.get(
            "MATCHA_WAV_OUT",
            str(Path.home() / "fanyi" / "wav_out" / "matcha_native"),
        ),
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        default=os.environ.get("MATCHA_REQUIRE_CUDA", "1") == "1",
    )
    args = parser.parse_args()

    root = Path(args.model_dir).resolve()
    matcha = root / "matcha-icefall-zh-en"
    if not matcha.is_dir():
        # allow dest == the extracted folder itself
        if (root / "model-steps-3.onnx").is_file():
            matcha = root
            vocoder = root.parent / "vocos-16khz-univ.onnx"
        else:
            print(f"missing {matcha}; run download_models.py", file=sys.stderr)
            return 2
    else:
        vocoder = root / "vocos-16khz-univ.onnx"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stages: dict = {
        "model": str(matcha),
        "vocoder": str(vocoder),
        "provider_requested": args.provider,
        "num_threads": args.num_threads,
        "melo_reference": {"rtf": 0.082, "run2_rss_mb": 1177},
        "start": round(rss_mb(), 1),
        "gpu_mem_start_mb": gpu_mem_mb(),
    }
    print(f"start rss={stages['start']} gpu_mem={stages['gpu_mem_start_mb']}", flush=True)

    import sherpa_onnx

    stages["after_import"] = round(rss_mb(), 1)
    stages["host_base"] = stages["after_import"]
    stages["sherpa_onnx_version"] = getattr(sherpa_onnx, "__version__", "?")
    stages["sherpa_onnx_file"] = sherpa_onnx.__file__
    print(
        f"host_base (after import, no OfflineTts) rss={stages['host_base']} "
        f"ver={stages['sherpa_onnx_version']}",
        flush=True,
    )

    ver = str(stages["sherpa_onnx_version"])
    looks_cuda_wheel = "+cuda" in ver.lower()
    if args.provider == "cuda" and args.require_cuda and not looks_cuda_wheel:
        # still try: some source builds omit +cuda in __version__
        print(
            f"WARN: sherpa-onnx version={ver} has no +cuda; will still request provider=cuda",
            flush=True,
        )

    fst = ",".join(
        str(matcha / name)
        for name in ("phone-zh.fst", "date-zh.fst", "number-zh.fst")
    )
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(matcha / "model-steps-3.onnx"),
                vocoder=str(vocoder),
                lexicon=str(matcha / "lexicon.txt"),
                tokens=str(matcha / "tokens.txt"),
                data_dir=str(matcha / "espeak-ng-data"),
            ),
            num_threads=args.num_threads,
            provider=args.provider,
            debug=True,
        ),
        max_num_sentences=1,
        rule_fsts=fst,
    )
    if not tts_config.validate():
        print("OfflineTtsConfig.validate() failed", file=sys.stderr)
        return 2

    log_buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(log_buf), redirect_stderr(log_buf):
        tts = sherpa_onnx.OfflineTts(tts_config)
    load_s = time.perf_counter() - t0
    debug_log = log_buf.getvalue()
    (out_dir / "offline_tts_debug.log").write_text(debug_log, encoding="utf-8")
    log_hits = parse_cuda_from_logs(debug_log)

    stages["after_session"] = round(rss_mb(), 1)
    stages["session_delta"] = round(stages["after_session"] - stages["host_base"], 1)
    stages["session_load_s"] = round(load_s, 2)
    stages["gpu_mem_after_session_mb"] = gpu_mem_mb()
    stages["debug_log_hits"] = log_hits
    sr = int(getattr(tts, "sample_rate", 0) or 0)
    stages["sample_rate"] = sr
    print(
        f"after_session rss={stages['after_session']} "
        f"(+{stages['session_delta']} vs host_base) load={load_s:.2f}s "
        f"sample_rate={sr} gpu_mem={stages['gpu_mem_after_session_mb']}",
        flush=True,
    )
    print(f"debug_log_hits={log_hits}", flush=True)

    gpu0 = stages["gpu_mem_start_mb"]
    gpu1 = stages["gpu_mem_after_session_mb"]
    gpu_grew = gpu0 is not None and gpu1 is not None and (gpu1 - gpu0) >= 20
    cuda_ep = False
    if args.provider == "cuda":
        if log_hits.get("please_compile_gpu") or (
            log_hits.get("fallback_cpu") and not log_hits.get("cuda_execution_provider")
        ):
            cuda_ep = False
        elif log_hits.get("cuda_execution_provider") or gpu_grew or looks_cuda_wheel:
            cuda_ep = True
    stages["cuda_ep_enabled"] = bool(cuda_ep)
    stages["cuda_evidence"] = {
        "version_plus_cuda": looks_cuda_wheel,
        "debug_log": log_hits,
        "gpu_mem_grew": gpu_grew,
    }

    if args.provider == "cuda" and args.require_cuda and not stages["cuda_ep_enabled"]:
        print(
            "FATAL: provider=cuda requested but CUDA EP evidence missing. "
            "Refuse silent CPU fallback. See isolated Dockerfile.jetson.matcha. "
            f"log={out_dir / 'offline_tts_debug.log'}",
            file=sys.stderr,
        )
        (out_dir / "bench_matcha_native.json").write_text(
            json.dumps(stages, ensure_ascii=False, indent=2) + "\n"
        )
        return 3

    def run_one(tag: str, text: str, save: bool) -> dict:
        t1 = time.perf_counter()
        audio = generate(tts, text)
        dt = time.perf_counter() - t1
        samples = audio.samples
        sr_i = int(audio.sample_rate)
        n = len(samples)
        dur = n / float(sr_i) if sr_i else 0.0
        rtf = dt / dur if dur > 1e-6 else -1.0
        row = {
            "tag": tag,
            "rss_mb": round(rss_mb(), 1),
            "elapsed_s": round(dt, 3),
            "audio_s": round(dur, 3),
            "rtf": round(rtf, 3),
            "sample_rate": sr_i,
            "n_samples": n,
            "ok_mix": n > 0,
        }
        print(
            f"{tag} rss={row['rss_mb']} elapsed={dt:.3f}s audio={dur:.3f}s "
            f"rtf={rtf:.3f} sr={sr_i} n={n}",
            flush=True,
        )
        if save and n:
            write_wav(out_dir / f"{tag}.wav", samples, sr_i)
        return row

    stages["warmup_zh"] = run_one("warmup_zh", WARMUP_ZH, True)
    stages["warmup_en"] = run_one("warmup_en", WARMUP_EN, True)
    stages["run1_melo_parity"] = run_one("run1_melo_parity", MELO_PARITY, False)
    stages["run2_melo_parity"] = run_one("run2_melo_parity", MELO_PARITY, True)
    stages["official_mix"] = run_one("official_mix", OFFICIAL_MIX, True)

    stages["after_run2"] = stages["run2_melo_parity"]["rss_mb"]
    stages["run2_rtf"] = stages["run2_melo_parity"]["rtf"]
    stages["sample_rate_ok"] = stages["run2_melo_parity"]["sample_rate"] == 16000
    stages["mix_ok"] = bool(
        stages["official_mix"]["ok_mix"]
        and stages["warmup_en"]["ok_mix"]
        and stages["warmup_zh"]["ok_mix"]
    )

    summary = {
        "zh_en_mix_ok": stages["mix_ok"],
        "sample_rate": stages["run2_melo_parity"]["sample_rate"],
        "sample_rate_ok": stages["sample_rate_ok"],
        "rtf_run2_melo_parity": stages["run2_rtf"],
        "rtf_official_mix": stages["official_mix"]["rtf"],
        "host_base_rss_mb": stages["host_base"],
        "after_session_rss_mb": stages["after_session"],
        "warmup_zh_rss_mb": stages["warmup_zh"]["rss_mb"],
        "run2_rss_mb": stages["after_run2"],
        "cuda_ep_enabled": stages["cuda_ep_enabled"],
        "vs_melo_external": {
            "melo_rtf": 0.082,
            "melo_run2_rss_mb": 1177,
            "rtf_delta": round(stages["run2_rtf"] - 0.082, 3),
            "rss_delta_mb": round(stages["after_run2"] - 1177, 1)
            if stages["after_run2"] >= 0
            else None,
        },
        "wav_out": str(out_dir),
    }
    stages["summary"] = summary
    report = out_dir / "bench_matcha_native.json"
    report.write_text(json.dumps(stages, ensure_ascii=False, indent=2) + "\n")
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"wrote {report}", flush=True)
    return 0 if stages["mix_ok"] and stages["sample_rate_ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
