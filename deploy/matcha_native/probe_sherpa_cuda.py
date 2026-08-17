#!/usr/bin/env python3
"""Probe whether THIS python's sherpa-onnx is a CUDA build.

Do not confuse with `import onnxruntime` CUDA EP (Melo uses that wheel).
PyPI `pip install sherpa-onnx` is CPU-only; GPU needs
SHERPA_ONNX_ENABLE_GPU=ON / version containing '+cuda'.

This script never pip-installs and never touches Melo packages.
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def rss_mb() -> float:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def _ldd(path: str) -> str:
    try:
        return subprocess.check_output(["ldd", path], text=True, stderr=subprocess.STDOUT)
    except Exception as e:
        return f"ldd_failed: {e}"


def _ort_python() -> dict:
    """System/site onnxruntime (Melo's wheel). Not sherpa's bundled ORT."""
    info = {"import_ok": False}
    try:
        import onnxruntime as ort

        info["import_ok"] = True
        info["version"] = getattr(ort, "__version__", "?")
        info["file"] = getattr(ort, "__file__", "?")
        info["available_providers"] = list(ort.get_available_providers())
        info["has_cuda_ep"] = "CUDAExecutionProvider" in info["available_providers"]
        info["note"] = (
            "This is process-level onnxruntime, NOT proof sherpa-onnx uses CUDA. "
            "Melo OpenEPD uses this wheel directly; pip sherpa-onnx bundles its own ORT."
        )
    except Exception as e:
        info["error"] = str(e)
    return info


def main() -> int:
    report: dict = {
        "python": sys.executable,
        "rss_mb_start": round(rss_mb(), 1),
        "cuda_build": False,
        "reason": [],
        "do_not_modify_melo": True,
    }

    try:
        import sherpa_onnx
    except Exception as e:
        report["sherpa_onnx_import"] = False
        report["error"] = str(e)
        report["reason"].append("sherpa_onnx not importable in this python")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2

    pkg = Path(sherpa_onnx.__file__).resolve().parent
    version = getattr(sherpa_onnx, "__version__", "?")
    report["sherpa_onnx_import"] = True
    report["sherpa_onnx_version"] = version
    report["sherpa_onnx_file"] = str(pkg)
    report["rss_mb_after_import"] = round(rss_mb(), 1)

    if "+cuda" in str(version).lower():
        report["cuda_build"] = True
        report["reason"].append(f"version has +cuda ({version})")
    else:
        report["reason"].append(
            f"version={version} has no +cuda suffix (typical of CPU pip wheel)"
        )

    so_files = sorted(
        glob.glob(str(pkg / "**" / "*sherpa_onnx*"), recursive=True)
        + glob.glob(str(pkg / "**" / "libonnxruntime*"), recursive=True)
        + glob.glob(str(pkg / "**" / "*providers_cuda*"), recursive=True)
    )
    # also sibling lib dirs used by some wheels
    for extra in (pkg / "lib", pkg.parent / "sherpa_onnx.libs"):
        if extra.is_dir():
            so_files.extend(str(p) for p in extra.rglob("*") if p.is_file())
    so_files = sorted(set(so_files))
    report["package_files_sample"] = so_files[:40]

    cuda_lib_hits = []
    for f in so_files:
        name = os.path.basename(f).lower()
        if "cuda" in name or "cudnn" in name:
            cuda_lib_hits.append(f)
        if name.endswith(".so") or ".so." in name:
            ldd = _ldd(f)
            if "libcudart" in ldd or "libonnxruntime_providers_cuda" in ldd:
                cuda_lib_hits.append(f"{f} :: libcudart/providers_cuda")
                report["cuda_build"] = True
    report["cuda_lib_hits"] = cuda_lib_hits[:20]
    if cuda_lib_hits and not report["cuda_build"]:
        report["cuda_build"] = True
        report["reason"].append("found CUDA libs next to sherpa_onnx")
    if not cuda_lib_hits:
        report["reason"].append("no CUDA .so / libcudart linkage found in sherpa_onnx package")

    report["site_onnxruntime"] = _ort_python()

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\nVERDICT:", file=sys.stderr)
    if report["cuda_build"]:
        print("  sherpa-onnx looks like a CUDA build. OK to bench with provider=cuda.", file=sys.stderr)
        return 0
    print(
        "  sherpa-onnx is NOT a CUDA build. Do NOT pip-reinstall inside the Melo image.\n"
        "  Build the isolated image: perception/Dockerfile.jetson.matcha",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
