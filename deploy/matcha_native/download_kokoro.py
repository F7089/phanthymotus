#!/usr/bin/env python3
"""Download official sherpa-onnx kokoro-multi-lang-v1_1 (FP32, ZH+EN).

Same pipeline as Matcha native: sherpa-onnx + CUDA, no OpenEPD, no Melo.
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

GH = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
GHFAST = "https://ghfast.top/https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
NAME = "kokoro-multi-lang-v1_1.tar.bz2"
URLS = [f"{GHFAST}/{NAME}", f"{GH}/{NAME}"]
REQUIRED = (
    "model.onnx",
    "voices.bin",
    "tokens.txt",
    "espeak-ng-data/phontab",
    "lexicon-zh.txt",
    "lexicon-us-en.txt",
)


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.1f}MB"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = [-1]

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        done = block_num * block_size
        if total_size > 0:
            pct = min(int(done * 100 / total_size), 100)
            if pct >= last[0] + 10:
                last[0] = pct
                print(
                    f"  {dest.name}: {pct}% ({_mb(min(done, total_size))}/{_mb(total_size)})",
                    flush=True,
                )

    print(f"GET {url}", flush=True)
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.replace(dest)
    print(f"  saved {dest} ({_mb(dest.stat().st_size)})", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest",
        default=os.environ.get(
            "KOKORO_MODEL_DIR",
            str(Path.home() / "fanyi" / "kokoro_native"),
        ),
    )
    args = parser.parse_args()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / NAME
    marker = dest / "kokoro-multi-lang-v1_1" / "model.onnx"
    if not marker.is_file():
        if not (archive.is_file() and archive.stat().st_size > 1024):
            err = []
            for url in URLS:
                try:
                    download(url, archive)
                    break
                except Exception as e:
                    err.append(f"{url}: {e}")
            else:
                raise RuntimeError("download failed:\n" + "\n".join(err))
        print(f"extract {archive}", flush=True)
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(dest)
    model_dir = dest / "kokoro-multi-lang-v1_1"
    missing = [str(model_dir / r) for r in REQUIRED if not (model_dir / r).exists()]
    if missing:
        raise RuntimeError("missing:\n  " + "\n  ".join(missing))
    onnx = model_dir / "model.onnx"
    voices = model_dir / "voices.bin"
    print(f"model.onnx {onnx.stat().st_size / (1024**2):.1f}MB", flush=True)
    print(f"voices.bin {voices.stat().st_size / (1024**2):.1f}MB", flush=True)
    print("verify OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
