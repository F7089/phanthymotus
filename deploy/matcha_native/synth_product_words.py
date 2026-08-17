#!/usr/bin/env python3
"""Synthesize Melo-hard product words with native Matcha (eSpeak/lexicon).

debug=False. Reuses the CUDA sherpa-onnx Matcha image.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_matcha_native import generate, rss_mb, write_wav

# Cases from the Melo lexicon thread (iphone/ipad/CUDA/API...).
CASES = [
    ("01_iphone_ipad_sync", "请把 iPhone 和 iPad 的设置同步一下。"),
    ("02_two_iphone_one_ipad", "我有两个 iphone 和一个 ipad。"),
    ("03_iphone15_8gb", "iPhone 15 有 8GB 内存。"),
    ("04_wifi", "请你先打开 WiFi 连上办公室网络，确认信号稳定之后，再继续下载这次的系统更新包。"),
    ("05_cuda", "CUDA 已经安装好了，可以用 GPU 加速。"),
    ("06_api_http", "这个 API 返回 HTTP 404，请检查一下。"),
    ("07_chatgpt_deepseek", "请用 ChatGPT 或者 deepseek 帮我看一下这段代码。"),
    ("08_tts_asr_gpu", "TTS 和 ASR 都要用 GPU。"),
    ("09_ceo_it", "CEO 明天会来看 IT 系统。"),
    ("10_watch_airpods", "Apple Watch 和 AirPods 也要配对。"),
]


def load_tts(model_root: Path, provider: str, num_threads: int):
    import sherpa_onnx

    matcha = model_root / "matcha-icefall-zh-en"
    vocoder = model_root / "vocos-16khz-univ.onnx"
    if not matcha.is_dir() and (model_root / "model-steps-3.onnx").is_file():
        matcha = model_root
        vocoder = model_root.parent / "vocos-16khz-univ.onnx"
    fst = ",".join(
        str(matcha / n) for n in ("phone-zh.fst", "date-zh.fst", "number-zh.fst")
    )
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(matcha / "model-steps-3.onnx"),
                vocoder=str(vocoder),
                lexicon=str(matcha / "lexicon.txt"),
                tokens=str(matcha / "tokens.txt"),
                data_dir=str(matcha / "espeak-ng-data"),
            ),
            num_threads=num_threads,
            provider=provider,
            debug=False,
        ),
        max_num_sentences=1,
        rule_fsts=fst,
    )
    if not cfg.validate():
        raise RuntimeError("OfflineTtsConfig.validate() failed")
    return sherpa_onnx.OfflineTts(cfg)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("MATCHA_MODEL_DIR", "/models/matcha_native"),
    )
    parser.add_argument("--provider", default=os.environ.get("MATCHA_PROVIDER", "cuda"))
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("MATCHA_WAV_OUT", "/wav_out/product_words"),
    )
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"rss_start={rss_mb():.1f}", flush=True)
    tts = load_tts(Path(args.model_dir), args.provider, args.num_threads)
    sr = int(getattr(tts, "sample_rate", 0) or 0)
    print(f"loaded sr={sr} rss={rss_mb():.1f} provider={args.provider}", flush=True)

    index = []
    for name, text in CASES:
        t0 = time.perf_counter()
        audio = generate(tts, text)
        dt = time.perf_counter() - t0
        n = len(audio.samples)
        sr_i = int(audio.sample_rate)
        dur = n / float(sr_i) if sr_i else 0.0
        wav = out_dir / f"{name}.wav"
        write_wav(wav, audio.samples, sr_i)
        line = f"{name} rtf={dt / dur if dur else -1:.3f} audio={dur:.2f}s sr={sr_i} text={text}"
        print(line, flush=True)
        index.append(f"<p><a href='{name}.wav'>{name}</a> {text}</p>")
    (out_dir / "index.html").write_text(
        "<meta charset='utf-8'><h3>Matcha product words</h3>\n" + "\n".join(index),
        encoding="utf-8",
    )
    print(f"wrote {out_dir} rss={rss_mb():.1f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
