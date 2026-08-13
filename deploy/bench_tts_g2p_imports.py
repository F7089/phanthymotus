#!/usr/bin/env python3
"""Split the +445MB openepd_g2p spike (GPT A→E). No ORT session / no finetune.

A  before import melo_g2p.encode
B  after  import melo_g2p.encode   (+ new sys.modules)
C  after  load OpenEPD pickle
D  after  first encode
E  after  second encode
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return -1.0


def mark(label: str, rows: list, extra: dict | None = None) -> float:
    r = round(rss_mb(), 1)
    row = {"step": label, "rss_mib": r}
    if rows:
        row["delta_mib"] = round(r - rows[-1]["rss_mib"], 1)
    else:
        row["delta_mib"] = 0.0
    if extra:
        row.update(extra)
    rows.append(row)
    d = row["delta_mib"]
    print(f"{label:28s}  rss={r:7.1f}  delta={d:+7.1f}", flush=True)
    return r


def main() -> None:
    g2p = Path("/models/melo-openepd-g2p-assets")
    os.environ["MELO_OPENEPD_DICT"] = str(g2p / "openepd_eng_dict.pickle")
    os.environ.setdefault("MELO_SKIP_HF_TOKENIZER", "1")
    sys.path.insert(0, str(g2p / "vendor"))

    rows: list = []
    print(f"pid={os.getpid()}", flush=True)

    # Warm the same cheap deps as host_imports (so A ≈ ~123MB)
    import numpy  # noqa: F401
    import jieba

    list(jieba.cut("你好"))
    import onnxruntime  # noqa: F401

    mark("A_before_melo_import", rows)

    before_mods = set(sys.modules)
    t0 = time.perf_counter()
    from melo_g2p.encode import encode_phones_tones

    import_s = time.perf_counter() - t0
    after_mods = set(sys.modules)
    new_mods = sorted(after_mods - before_mods)
    mark("B_after_melo_import", rows, {"import_s": round(import_s, 3), "new_modules": len(new_mods)})

    # Suspicious heavy packages
    suspects = [
        "torch",
        "torchaudio",
        "transformers",
        "librosa",
        "numba",
        "llvmlite",
        "sklearn",
        "nltk",
        "pandas",
        "tensorflow",
        "pypinyin",
        "cn2an",
        "g2p_en",
        "unidecode",
        "inflect",
        "gruut",
    ]
    print("\nsuspect sys.modules:")
    for name in suspects:
        print(f"  {name:16s} {name in sys.modules}")

    print(f"\nnew sys.modules ({len(new_mods)}):")
    for x in new_mods:
        print(f"  {x}")

    # Load OpenEPD pickle alone (may already be loaded by english import)
    t1 = time.perf_counter()
    import pickle

    with open(g2p / "openepd_eng_dict.pickle", "rb") as f:
        eng = pickle.load(f)
    mark(
        "C_after_openepd_pickle",
        rows,
        {
            "pickle_s": round(time.perf_counter() - t1, 3),
            "entries": len(eng) if hasattr(eng, "__len__") else None,
        },
    )

    with open(g2p / "config.json", encoding="utf-8") as f:
        meta = json.load(f)
    symbols = list(meta["symbols"])
    lang = str(meta.get("language") or "ZH_MIX_EN")
    add_blank = bool(meta.get("add_blank", True))

    t2 = time.perf_counter()
    encode_phones_tones("你好 hello", symbols, add_blank=add_blank, language=lang)
    mark("D_after_first_encode", rows, {"encode_s": round(time.perf_counter() - t2, 3)})

    t3 = time.perf_counter()
    encode_phones_tones(
        "今天天气怎么样？", symbols, add_blank=add_blank, language=lang
    )
    mark("E_after_second_encode", rows, {"encode_s": round(time.perf_counter() - t3, 3)})

    # maps: torch / blas / etc
    print("\n/proc/self/maps heavy hits:")
    try:
        hits = []
        for line in Path("/proc/self/maps").read_text().splitlines():
            if "/" not in line:
                continue
            path = line[line.find("/") :]
            low = path.lower()
            if any(
                k in low
                for k in (
                    "libtorch",
                    "torch/",
                    "libtensorflow",
                    "libopenblas",
                    "liblapack",
                    "libgfortran",
                    "numba",
                    "llvmlite",
                    "cudnn",
                    "cublas",
                    "onnxruntime",
                    "transformers",
                    "nltk",
                )
            ):
                hits.append(path)
        for h in sorted(set(hits))[:50]:
            print(f"  {h}")
        if not hits:
            print("  (none)")
    except Exception as e:
        print(f"  maps error: {e}")

    out = Path("/tmp/g2p_imports.json")
    payload = {"pid": os.getpid(), "stages": rows, "new_modules": new_mods}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("\nSUMMARY A→E:")
    for r in rows:
        print(
            f"  {r['step']:28s} rss={r['rss_mib']} delta={r.get('delta_mib', 0):+}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
