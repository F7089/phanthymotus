#!/usr/bin/env python3
"""Refine +445MB G2P spike (no double-load OpenEPD, split encode imports).

Prior result:
  A import encode          +0
  C pickle.load OpenEPD  +217   ← file ~292KB expands hugely
  D first encode         +449   ← but C already held a copy (inflated)

This run:
  A  baseline (numpy+jieba+ort)
  B  import melo_g2p.encode
  C1 import text.chinese
  C2 import text.english   (usually loads OpenEPD once)
  D  first encode
  E  second encode
  + suspect modules / maps after D
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


def mark(label: str, rows: list, extra: dict | None = None) -> None:
    r = round(rss_mb(), 1)
    row = {"step": label, "rss_mib": r}
    row["delta_mib"] = round(r - rows[-1]["rss_mib"], 1) if rows else 0.0
    if extra:
        row.update(extra)
    rows.append(row)
    print(
        f"{label:28s}  rss={r:7.1f}  delta={row['delta_mib']:+7.1f}",
        flush=True,
    )


def dump_suspects(tag: str) -> None:
    suspects = [
        "torch",
        "transformers",
        "nltk",
        "g2p_en",
        "pypinyin",
        "cn2an",
        "scipy",
        "sklearn",
        "numba",
        "librosa",
        "inflect",
        "unidecode",
        "jieba",
        "numpy",
        "onnxruntime",
    ]
    print(f"\nsuspect @ {tag}:")
    for name in suspects:
        print(f"  {name:16s} {name in sys.modules}")


def main() -> None:
    g2p = Path("/models/melo-openepd-g2p-assets")
    os.environ["MELO_OPENEPD_DICT"] = str(g2p / "openepd_eng_dict.pickle")
    os.environ.setdefault("MELO_SKIP_HF_TOKENIZER", "1")
    sys.path.insert(0, str(g2p / "vendor"))

    rows: list = []
    print(f"pid={os.getpid()}", flush=True)

    import numpy  # noqa: F401
    import jieba

    list(jieba.cut("你好"))
    import onnxruntime  # noqa: F401

    mark("A_baseline", rows)

    before = set(sys.modules)
    from melo_g2p.encode import encode_phones_tones

    new = sorted(set(sys.modules) - before)
    mark("B_import_encode", rows, {"new_modules": new})
    print(f"new modules after B: {new}")

    before = set(sys.modules)
    from melo_g2p.text import chinese  # noqa: F401

    new = sorted(set(sys.modules) - before)
    mark("C1_import_chinese", rows, {"new_modules": new})
    print(f"new modules after C1 ({len(new)}):")
    for x in new:
        print(f"  {x}")
    dump_suspects("C1")

    before = set(sys.modules)
    from melo_g2p.text import english  # noqa: F401

    new = sorted(set(sys.modules) - before)
    mark("C2_import_english", rows, {"new_modules": new})
    print(f"new modules after C2 ({len(new)}):")
    for x in new:
        print(f"  {x}")
    dump_suspects("C2")

    # OpenEPD file size vs in-memory if english exposed it
    pkl = g2p / "openepd_eng_dict.pickle"
    print(f"\nopenepd pickle file_bytes={pkl.stat().st_size}")

    with open(g2p / "config.json", encoding="utf-8") as f:
        meta = json.load(f)
    symbols = list(meta["symbols"])
    lang = str(meta.get("language") or "ZH_MIX_EN")
    add_blank = bool(meta.get("add_blank", True))

    t0 = time.perf_counter()
    encode_phones_tones("你好 hello", symbols, add_blank=add_blank, language=lang)
    mark("D_first_encode", rows, {"sec": round(time.perf_counter() - t0, 3)})
    dump_suspects("D")

    t1 = time.perf_counter()
    encode_phones_tones(
        "今天天气怎么样？", symbols, add_blank=add_blank, language=lang
    )
    mark("E_second_encode", rows, {"sec": round(time.perf_counter() - t1, 3)})

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
                    "libopenblas",
                    "liblapack",
                    "libgfortran",
                    "scipy",
                    "nltk",
                    "pypinyin",
                )
            ):
                hits.append(path)
        for h in sorted(set(hits))[:40]:
            print(f"  {h}")
        if not hits:
            print("  (none)")
    except Exception as e:
        print(f"  maps error: {e}")

    out = Path("/tmp/g2p_imports.json")
    out.write_text(
        json.dumps({"pid": os.getpid(), "stages": rows}, ensure_ascii=False, indent=2)
        + "\n"
    )
    print("\nSUMMARY:")
    for r in rows:
        print(
            f"  {r['step']:28s} rss={r['rss_mib']} delta={r.get('delta_mib', 0):+}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
