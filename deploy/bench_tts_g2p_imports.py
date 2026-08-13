#!/usr/bin/env python3
"""Subdivide the +445MB openepd_g2p_encode spike (still no ORT session)."""
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


def step(name: str, fn, rows: list) -> None:
    before = rss_mb()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    after = rss_mb()
    rows.append(
        {
            "step": name,
            "rss_mib": round(after, 1),
            "delta_mib": round(after - before, 1),
            "sec": round(dt, 3),
        }
    )
    print(
        f"{name:36s}  rss={after:7.1f}  delta={after-before:+7.1f}  {dt:.3f}s",
        flush=True,
    )


def main() -> None:
    g2p = Path("/models/melo-openepd-g2p-assets")
    os.environ["MELO_OPENEPD_DICT"] = str(g2p / "openepd_eng_dict.pickle")
    os.environ.setdefault("MELO_SKIP_HF_TOKENIZER", "1")
    sys.path.insert(0, str(g2p / "vendor"))
    rows: list = []

    print(f"pid={os.getpid()} start_rss={rss_mb():.1f}")
    # Match prior host_imports baseline-ish: numpy+jieba already warm in real svc,
    # but here start clean and only walk G2P internals.
    step("import_numpy", lambda: __import__("numpy"), rows)
    step("import_jieba_cut", lambda: list(__import__("jieba").cut("你好")), rows)

    step("import_melo_g2p_pkg", lambda: __import__("melo_g2p"), rows)

    def _cleaner():
        from melo_g2p.text import cleaner  # noqa: F401

    step("import_melo_g2p.text.cleaner", _cleaner, rows)

    def _chinese():
        from melo_g2p.text import chinese  # noqa: F401

    step("import_melo_g2p.text.chinese", _chinese, rows)

    def _chinese_mix():
        from melo_g2p.text import chinese_mix  # noqa: F401

    step("import_melo_g2p.text.chinese_mix", _chinese_mix, rows)

    def _english():
        from melo_g2p.text import english  # noqa: F401

    step("import_melo_g2p.text.english", _english, rows)

    def _load_openepd():
        import pickle

        p = g2p / "openepd_eng_dict.pickle"
        with open(p, "rb") as f:
            d = pickle.load(f)
        print(f"  openepd_entries≈{len(d) if hasattr(d, '__len__') else '?'}", flush=True)

    step("pickle_load_openepd", _load_openepd, rows)

    def _encode():
        from melo_g2p.encode import encode_phones_tones

        with open(g2p / "config.json", encoding="utf-8") as f:
            meta = json.load(f)
        encode_phones_tones(
            "你好 hello",
            list(meta["symbols"]),
            add_blank=bool(meta.get("add_blank", True)),
            language=str(meta.get("language") or "ZH_MIX_EN"),
        )

    step("encode_phones_tones_once", _encode, rows)

    # Who is mapped?
    try:
        maps = Path("/proc/self/maps").read_text().splitlines()
        heavy = []
        for line in maps:
            if "/" not in line:
                continue
            path = line[line.find("/") :]
            if any(
                k in path
                for k in (
                    "torch",
                    "transformers",
                    "nltk",
                    "pypinyin",
                    "onnxruntime",
                    "scipy",
                    "jieba",
                    "melo",
                    "cuda",
                    "cudnn",
                )
            ):
                heavy.append(path)
        uniq = sorted(set(heavy))[:40]
        print("\nmapped libs (sample):")
        for u in uniq:
            print(f"  {u}")
    except Exception as e:
        print(f"maps skip: {e}")

    out = Path("/tmp/g2p_imports.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print("-" * 64)
    print(f"final_rss={rss_mb():.1f}")
    print("TOP deltas:")
    for r in sorted(rows, key=lambda x: x["delta_mib"], reverse=True)[:8]:
        print(f"  {r['delta_mib']:+7.1f}  {r['step']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
