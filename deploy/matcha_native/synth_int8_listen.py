#!/usr/bin/env python3
"""INT8 Matcha listen: acoustic + vocos both *int8*.onnx. Not for leaderboard."""
from __future__ import annotations

import argparse
import os
import subprocess
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

LONG = (
    "请把 iPhone 和 iPad 的设置同步一下。"
    "CUDA 已经安装好了，可以用 GPU 加速。"
    "CEO 明天会来看 IT 系统，这个 API 返回 HTTP 404。"
    "他去西藏学藏文，藏品放在银行，行长说行不行？"
)
WARMUP = "你好，这是 INT8 测试。"


def rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def gpu_mem_mb():
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


def generate(tts, text: str):
    try:
        return tts.generate(text, sid=0, speed=1.0)
    except TypeError:
        import sherpa_onnx

        cfg = sherpa_onnx.GenerationConfig()
        cfg.sid = 0
        cfg.speed = 1.0
        return tts.generate(text, cfg)


def resolve_paths(model_dir: Path, acoustic_name: str, vocoder_name: str):
    if (model_dir / "tokens.txt").is_file():
        pack = model_dir
        vocoder = model_dir / vocoder_name
        if not vocoder.is_file():
            vocoder = model_dir.parent / vocoder_name
    else:
        pack = model_dir / "matcha-icefall-zh-en"
        vocoder = model_dir / vocoder_name
        if not vocoder.is_file():
            vocoder = pack / vocoder_name
    return pack, pack / acoustic_name, vocoder


def load_tts(pack: Path, acoustic: Path, vocoder: Path, provider: str, num_threads: int):
    import sherpa_onnx

    fst = ",".join(
        str(pack / n) for n in ("phone-zh.fst", "date-zh.fst", "number-zh.fst")
    )
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(acoustic),
                vocoder=str(vocoder),
                lexicon=str(pack / "lexicon.txt"),
                tokens=str(pack / "tokens.txt"),
                data_dir=str(pack / "espeak-ng-data"),
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


def sample_peaks(stop: threading.Event, out: dict) -> None:
    gpu_peak = -1.0
    rss_peak = -1.0
    while not stop.wait(0.05):
        g = gpu_mem_mb()
        if g is not None:
            gpu_peak = max(gpu_peak, g)
        r = rss_mb()
        if r >= 0:
            rss_peak = max(rss_peak, r)
    out["gpu"] = gpu_peak
    out["rss"] = rss_peak


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("MATCHA_MODEL_DIR", "."))
    parser.add_argument("--acoustic", default="model-steps-6.int8.onnx")
    parser.add_argument("--vocoder", default="vocos-16khz-univ.int8.onnx")
    parser.add_argument("--provider", default=os.environ.get("MATCHA_PROVIDER", "cuda"))
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--out", default="")
    parser.add_argument("--http", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    pack, acoustic, vocoder = resolve_paths(model_dir, args.acoustic, args.vocoder)
    if not acoustic.is_file():
        raise SystemExit(f"missing INT8 acoustic: {acoustic}")
    if not vocoder.is_file():
        raise SystemExit(f"missing INT8 vocoder: {vocoder}")
    if "int8" not in acoustic.name.lower() or "int8" not in vocoder.name.lower():
        raise SystemExit("refuse: both acoustic and vocoder must be *int8*.onnx")

    print(f"acoustic={acoustic} size={acoustic.stat().st_size/1024/1024:.1f}MB", flush=True)
    print(f"vocoder={vocoder} size={vocoder.stat().st_size/1024/1024:.1f}MB", flush=True)
    print(f"provider_request={args.provider}", flush=True)

    print(f"rss_before={rss_mb():.1f}MB gpu_before={gpu_mem_mb()}", flush=True)

    try:
        tts = load_tts(pack, acoustic, vocoder, args.provider, args.num_threads)
        used_provider = args.provider
    except Exception as exc:
        print(f"WARN: load provider={args.provider} failed: {type(exc).__name__}: {exc}", flush=True)
        if args.provider.lower() == "cpu":
            raise
        print("fallback provider=cpu", flush=True)
        tts = load_tts(pack, acoustic, vocoder, "cpu", args.num_threads)
        used_provider = "cpu"

    print(f"rss_after_load={rss_mb():.1f}MB gpu_after_load={gpu_mem_mb()}", flush=True)
    generate(tts, WARMUP)

    peaks: dict = {}
    stop = threading.Event()
    th = threading.Thread(target=sample_peaks, args=(stop, peaks), daemon=True)
    th.start()
    t0 = time.perf_counter()
    audio = generate(tts, LONG)
    elapsed = time.perf_counter() - t0
    stop.set()
    th.join(timeout=1.0)

    sr = int(audio.sample_rate)
    dur = len(audio.samples) / float(sr)
    rtf = elapsed / dur if dur > 0 else -1.0
    out = Path(args.out) if args.out else Path("int8_long.wav")
    write_wav(out, audio.samples, sr)

    print(
        f"INT8_OK provider={used_provider} sr={sr} dur={dur:.3f}s "
        f"wall={elapsed:.3f}s RTF={rtf:.4f} "
        f"rss_peak={peaks.get('rss', rss_mb()):.1f}MB "
        f"gpu_peak_mb={peaks.get('gpu')} "
        f"out={out}",
        flush=True,
    )

    if args.http <= 0:
        return 0

    wav_bytes = out.read_bytes()
    latest = {"wav": wav_bytes, "rtf": rtf, "dur": dur}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *_a, **_k):
            return

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                body = (
                    "<!doctype html><meta charset=utf-8>"
                    f"<p>Matcha INT8 provider={used_provider} "
                    f"RTF={latest['rtf']:.4f} dur={latest['dur']:.2f}s</p>"
                    f"<p><a href=/wav>{quote(out.name)}</a></p>"
                    "<audio controls src=/wav></audio>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/wav":
                data = latest["wav"]
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)

    ip = os.popen("hostname -I").read().split()
    ip = ip[0] if ip else "0.0.0.0"
    httpd = HTTPServer(("0.0.0.0", args.http), H)
    print(f"HTTP http://{ip}:{args.http}/  Ctrl-C stop", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
