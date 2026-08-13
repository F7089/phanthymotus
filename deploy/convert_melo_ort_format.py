#!/usr/bin/env python3
"""Convert Melo FP32 ONNX → ORT format (+ optional external-data ONNX).

Inside Jetson container (has onnxruntime + onnx from prior benches):

  python3 /tmp/convert_melo_ort_format.py

Writes next to the source model:
  model.ort
  model.with_external.onnx + model.with_external.onnx.data  (optional)
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--onnx",
        default="/models/vits-melo-longanlingxin-openepd-nobert-44100-fp32/model.onnx",
    )
    ap.add_argument(
        "--out-dir",
        default="/tmp/melo_fp32_ort",
        help="write model.ort here (/models may be read-only)",
    )
    ap.add_argument("--external", action="store_true", help="also emit external-data ONNX")
    args = ap.parse_args()

    src = Path(args.onnx)
    if not src.is_file():
        raise SystemExit(f"missing {src}")
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Work in out_dir: convert_onnx_models_to_ort writes alongside the .onnx
    work_onnx = out_dir / "model.onnx"
    if work_onnx.resolve() != src.resolve():
        print(f"copy {src} -> {work_onnx}")
        shutil.copy2(src, work_onnx)

    print("converting to ORT format (may take a few minutes)...")
    cmd = [
        sys.executable,
        "-m",
        "onnxruntime.tools.convert_onnx_models_to_ort",
        "--optimization_style",
        "Fixed",
        str(work_onnx),
    ]
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)

    # Tool may emit model.ort or model.ort with optimization style suffix
    candidates = sorted(out_dir.glob("model*.ort"))
    if not candidates:
        raise SystemExit(f"no .ort produced in {out_dir}")
    # Prefer plain model.ort
    ort_path = out_dir / "model.ort"
    if not ort_path.is_file():
        shutil.copy2(candidates[0], ort_path)
    print(f"ORT: {ort_path} ({ort_path.stat().st_size/1024/1024:.1f} MiB)")
    for c in candidates:
        print(f"  also: {c.name} ({c.stat().st_size/1024/1024:.1f} MiB)")

    if args.external:
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data

        ext_onnx = out_dir / "model.with_external.onnx"
        print(f"external-data ONNX -> {ext_onnx}")
        m = onnx.load(str(src))
        convert_model_to_external_data(
            m,
            all_tensors_to_one_file=True,
            location="model.with_external.onnx.data",
            size_threshold=1024,
            convert_attribute=False,
        )
        onnx.save(m, str(ext_onnx))
        data = out_dir / "model.with_external.onnx.data"
        print(
            f"  protobuf={ext_onnx.stat().st_size/1024:.1f} KiB "
            f"data={data.stat().st_size/1024/1024:.1f} MiB"
            if data.is_file()
            else f"  wrote {ext_onnx}"
        )

    print("done")


if __name__ == "__main__":
    main()
