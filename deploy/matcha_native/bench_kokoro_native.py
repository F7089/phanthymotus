#!/usr/bin/env python3
"""Kokoro FP32 native baseline on the same CUDA sherpa-onnx image as Matcha.

debug=False (Matcha debug log was too noisy). Official lexicon + eSpeak, no OpenEPD.
Kokoro is typically 24kHz; report actual sample_rate.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_matcha_native import (
    MELO_PARITY,
    OFFICIAL_MIX,
    WARMUP_EN,
    WARMUP_ZH,
    generate,
    gpu_mem_mb,
    parse_cuda_from_logs,
    rss_mb,
    write_wav,
)


def lexicon_csv(model_dir: Path) -> str:
    parts = []
    for name in ("lexicon-us-en.txt", "lexicon-gb-en.txt", "lexicon-zh.txt"):
        p = model_dir / name
        if p.is_file():
            parts.append(str(p))
    return ",".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=os.environ.get(
            "KOKORO_MODEL_DIR",
            str(Path.home() / "fanyi" / "kokoro_native" / "kokoro-multi-lang-v1_1"),
        ),
    )
    parser.add_argument("--provider", default=os.environ.get("MATCHA_PROVIDER", "cuda"))
    parser.add_argument("--sid", type=int, default=int(os.environ.get("KOKORO_SID", "18")))
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument(
        "--out-dir",
        default=os.environ.get(
            "KOKORO_WAV_OUT",
            str(Path.home() / "fanyi" / "wav_out" / "kokoro_native"),
        ),
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        default=os.environ.get("MATCHA_REQUIRE_CUDA", "1") == "1",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    if (model_dir / "kokoro-multi-lang-v1_1" / "model.onnx").is_file():
        model_dir = model_dir / "kokoro-multi-lang-v1_1"
    if not (model_dir / "model.onnx").is_file():
        print(f"missing {model_dir}/model.onnx; run download_kokoro.py", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = {
        "model": str(model_dir),
        "sid": args.sid,
        "provider_requested": args.provider,
        "start": round(rss_mb(), 1),
        "gpu_mem_start_mb": gpu_mem_mb(),
        "matcha_reference": {"rtf": 0.03, "run2_rss_mb": 603},
        "melo_reference": {"rtf": 0.082, "run2_rss_mb": 1177},
    }
    print(f"start rss={stages['start']}", flush=True)

    import sherpa_onnx

    stages["host_base"] = round(rss_mb(), 1)
    stages["sherpa_onnx_version"] = getattr(sherpa_onnx, "__version__", "?")
    looks_cuda = "+cuda" in str(stages["sherpa_onnx_version"]).lower()
    print(f"host_base rss={stages['host_base']} ver={stages['sherpa_onnx_version']}", flush=True)

    fsts = [
        str(model_dir / n)
        for n in ("phone-zh.fst", "date-zh.fst", "number-zh.fst")
        if (model_dir / n).is_file()
    ]
    tts_config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=str(model_dir / "model.onnx"),
                voices=str(model_dir / "voices.bin"),
                tokens=str(model_dir / "tokens.txt"),
                data_dir=str(model_dir / "espeak-ng-data"),
                lexicon=lexicon_csv(model_dir),
            ),
            num_threads=args.num_threads,
            provider=args.provider,
            debug=False,
        ),
        max_num_sentences=1,
        rule_fsts=",".join(fsts),
    )
    if not tts_config.validate():
        print("config.validate() failed", file=sys.stderr)
        return 2

    log_buf = io.StringIO()
    t0 = time.perf_counter()
    with redirect_stdout(log_buf), redirect_stderr(log_buf):
        tts = sherpa_onnx.OfflineTts(tts_config)
    stages["after_session"] = round(rss_mb(), 1)
    stages["session_load_s"] = round(time.perf_counter() - t0, 2)
    stages["sample_rate"] = int(getattr(tts, "sample_rate", 0) or 0)
    hits = parse_cuda_from_logs(log_buf.getvalue())
    (out_dir / "offline_tts_debug.log").write_text(log_buf.getvalue(), encoding="utf-8")
    cuda_ep = bool(
        args.provider == "cuda"
        and (
            hits.get("cuda_execution_provider")
            or looks_cuda
            or (not hits.get("please_compile_gpu"))
        )
    )
    if hits.get("please_compile_gpu"):
        cuda_ep = False
    stages["cuda_ep_enabled"] = cuda_ep
    print(
        f"after_session rss={stages['after_session']} "
        f"load={stages['session_load_s']}s sr={stages['sample_rate']} cuda={cuda_ep}",
        flush=True,
    )
    if args.provider == "cuda" and args.require_cuda and not cuda_ep:
        print("FATAL: CUDA EP not evidenced", file=sys.stderr)
        return 3

    def run_one(tag: str, text: str, save: bool) -> dict:
        t1 = time.perf_counter()
        try:
            audio = tts.generate(text, sid=args.sid, speed=1.0)
        except TypeError:
            audio = generate(tts, text)
        dt = time.perf_counter() - t1
        n = len(audio.samples)
        sr_i = int(audio.sample_rate)
        dur = n / float(sr_i) if sr_i else 0.0
        rtf = dt / dur if dur > 1e-6 else -1.0
        row = {
            "tag": tag,
            "rss_mb": round(rss_mb(), 1),
            "elapsed_s": round(dt, 3),
            "audio_s": round(dur, 3),
            "rtf": round(rtf, 3),
            "sample_rate": sr_i,
            "ok_mix": n > 0,
        }
        print(
            f"{tag} rss={row['rss_mb']} elapsed={dt:.3f}s audio={dur:.3f}s "
            f"rtf={rtf:.3f} sr={sr_i}",
            flush=True,
        )
        if save and n:
            write_wav(out_dir / f"{tag}.wav", audio.samples, sr_i)
        return row

    stages["warmup_zh"] = run_one("warmup_zh", WARMUP_ZH, True)
    stages["warmup_en"] = run_one("warmup_en", WARMUP_EN, True)
    stages["run1_melo_parity"] = run_one("run1_melo_parity", MELO_PARITY, False)
    stages["run2_melo_parity"] = run_one("run2_melo_parity", MELO_PARITY, True)
    stages["official_mix"] = run_one("official_mix", OFFICIAL_MIX, True)

    summary = {
        "backend": "kokoro-multi-lang-v1_1-fp32",
        "sid": args.sid,
        "zh_en_mix_ok": bool(stages["official_mix"]["ok_mix"]),
        "sample_rate": stages["run2_melo_parity"]["sample_rate"],
        "rtf_run2_melo_parity": stages["run2_melo_parity"]["rtf"],
        "rtf_official_mix": stages["official_mix"]["rtf"],
        "host_base_rss_mb": stages["host_base"],
        "after_session_rss_mb": stages["after_session"],
        "run2_rss_mb": stages["run2_melo_parity"]["rss_mb"],
        "official_mix_rss_mb": stages["official_mix"]["rss_mb"],
        "cuda_ep_enabled": stages["cuda_ep_enabled"],
        "vs_matcha": {
            "matcha_rtf": 0.03,
            "matcha_run2_rss_mb": 603,
            "rtf_delta": round(stages["run2_melo_parity"]["rtf"] - 0.03, 3),
            "rss_delta_mb": round(stages["run2_melo_parity"]["rss_mb"] - 603, 1),
        },
        "wav_out": str(out_dir),
    }
    stages["summary"] = summary
    (out_dir / "bench_kokoro_native.json").write_text(
        json.dumps(stages, ensure_ascii=False, indent=2) + "\n"
    )
    print("\nSUMMARY", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["zh_en_mix_ok"] else 4


if __name__ == "__main__":
    sys.exit(main())
