#!/usr/bin/env python3
"""Download official sherpa-onnx matcha-icefall-zh-en + vocos-16khz-univ.

Does not touch Melo / OpenEPD. Assets stay out of git.

Layout after success (official):
  <dest>/matcha-icefall-zh-en/model-steps-3.onnx
  <dest>/matcha-icefall-zh-en/lexicon.txt
  <dest>/matcha-icefall-zh-en/tokens.txt
  <dest>/matcha-icefall-zh-en/espeak-ng-data/
  <dest>/matcha-icefall-zh-en/phone-zh.fst
  <dest>/matcha-icefall-zh-en/date-zh.fst
  <dest>/matcha-icefall-zh-en/number-zh.fst
  <dest>/vocos-16khz-univ.onnx
"""
from __future__ import annotations

import argparse
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

COS = "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public"
GH_TTS = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
GH_VOC = "https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models"

FILES = {
    "matcha-icefall-zh-en.tar.bz2": [
        f"{COS}/matcha-icefall-zh-en.tar.bz2",
        f"{GH_TTS}/matcha-icefall-zh-en.tar.bz2",
    ],
    "vocos-16khz-univ.onnx": [
        f"{COS}/vocos-16khz-univ.onnx",
        f"{GH_VOC}/vocos-16khz-univ.onnx",
    ],
}

REQUIRED = (
    "model-steps-3.onnx",
    "lexicon.txt",
    "tokens.txt",
    "phone-zh.fst",
    "date-zh.fst",
    "number-zh.fst",
    "espeak-ng-data/phontab",
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
        elif done and done // (20 * 1024 * 1024) != last[0]:
            last[0] = done // (20 * 1024 * 1024)
            print(f"  {dest.name}: {_mb(done)}", flush=True)

    print(f"GET {url}", flush=True)
    urllib.request.urlretrieve(url, tmp, reporthook=hook)
    tmp.replace(dest)
    print(f"  saved {dest} ({_mb(dest.stat().st_size)})", flush=True)


def fetch_one(name: str, dest: Path) -> None:
    if dest.is_file() and dest.stat().st_size > 1024:
        print(f"skip existing {dest} ({_mb(dest.stat().st_size)})", flush=True)
        return
    errors = []
    for url in FILES[name]:
        try:
            download(url, dest)
            return
        except Exception as e:
            errors.append(f"{url}: {e}")
            print(f"  failed: {e}", flush=True)
    raise RuntimeError(f"could not download {name}:\n" + "\n".join(errors))


def extract_matcha(archive: Path, dest: Path) -> Path:
    out = dest / "matcha-icefall-zh-en"
    marker = out / "model-steps-3.onnx"
    if marker.is_file():
        print(f"skip extract, already have {marker}", flush=True)
        return out
    print(f"extract {archive} -> {dest}", flush=True)
    with tarfile.open(archive, "r:bz2") as tf:
        tf.extractall(dest)
    if not marker.is_file():
        raise RuntimeError(f"extract did not produce {marker}")
    return out


def verify(root: Path) -> None:
    model_dir = root / "matcha-icefall-zh-en"
    vocoder = root / "vocos-16khz-univ.onnx"
    missing = []
    for rel in REQUIRED:
        p = model_dir / rel
        if not p.exists():
            missing.append(str(p))
    if not vocoder.is_file():
        missing.append(str(vocoder))
    if missing:
        raise RuntimeError("missing files:\n  " + "\n  ".join(missing))

    acoustic = model_dir / "model-steps-3.onnx"
    a_mb = acoustic.stat().st_size / (1024 * 1024)
    v_mb = vocoder.stat().st_size / (1024 * 1024)
    print(f"acoustic {acoustic.name}: {a_mb:.1f}MB (expect ~72-76)", flush=True)
    print(f"vocoder  {vocoder.name}: {v_mb:.1f}MB (expect ~51-54)", flush=True)
    if not (70 <= a_mb <= 80):
        print(f"WARN: acoustic size {a_mb:.1f}MB outside 70-80MB", flush=True)
    if not (48 <= v_mb <= 58):
        print(f"WARN: vocoder size {v_mb:.1f}MB outside 48-58MB", flush=True)
    print("verify OK", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dest",
        default=os.environ.get(
            "MATCHA_MODEL_DIR",
            str(Path(__file__).resolve().parent / "models"),
        ),
    )
    args = parser.parse_args()
    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"dest={dest}", flush=True)

    tar_path = dest / "matcha-icefall-zh-en.tar.bz2"
    voc_path = dest / "vocos-16khz-univ.onnx"
    fetch_one("matcha-icefall-zh-en.tar.bz2", tar_path)
    fetch_one("vocos-16khz-univ.onnx", voc_path)
    extract_matcha(tar_path, dest)
    verify(dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
