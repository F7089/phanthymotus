#!/usr/bin/env python3
"""Break down ~566MB host_base via sequential imports (no ORT session).

Run inside Jetson TTS container (or via the shell wrapper).
Each step prints RSS and delta. Does NOT create InferenceSession.
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


def step(name: str, fn, rows: list) -> None:
    before = rss_mb()
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    after = rss_mb()
    row = {
        "step": name,
        "rss_mib": round(after, 1),
        "delta_mib": round(after - before, 1),
        "sec": round(dt, 3),
    }
    rows.append(row)
    print(
        f"{name:28s}  rss={after:7.1f}  delta={after-before:+7.1f}  {dt:.3f}s",
        flush=True,
    )


def main() -> None:
    g2p = Path("/models/melo-openepd-g2p-assets")
    rows: list = []
    print(f"pid={os.getpid()}")
    print(f"{'step':28s}  {'rss':>10}  {'delta':>8}  time")
    print("-" * 60)
    rows.append({"step": "start", "rss_mib": round(rss_mb(), 1), "delta_mib": 0.0})
    print(f"{'start':28s}  rss={rss_mb():7.1f}  delta={0:+7.1f}  0.000s", flush=True)

    step("import_numpy", lambda: __import__("numpy"), rows)
    step("import_scipy", lambda: __import__("scipy"), rows)

    def _jieba():
        import jieba  # noqa: F401

        # same as Melo path: build/load dict once
        list(jieba.cut("你好"))

    step("import_jieba_warmup", _jieba, rows)

    step("import_onnxruntime", lambda: __import__("onnxruntime"), rows)

    def _ort_providers():
        import onnxruntime as ort

        _ = ort.get_available_providers()

    step("ort_get_available_providers", _ort_providers, rows)

    def _ros():
        # Optional: may be absent in some shells
        try:
            import rclpy  # noqa: F401

            return
        except Exception as e:
            print(f"  (rclpy skip: {e})", flush=True)

    step("import_rclpy", _ros, rows)

    def _openepd_g2p():
        os.environ["MELO_OPENEPD_DICT"] = str(g2p / "openepd_eng_dict.pickle")
        os.environ.setdefault("MELO_SKIP_HF_TOKENIZER", "1")
        sys.path.insert(0, str(g2p / "vendor"))
        from melo_g2p.encode import encode_phones_tones  # noqa: F401

        with open(g2p / "config.json", encoding="utf-8") as f:
            meta = json.load(f)
        encode_phones_tones(
            "你好 hello",
            list(meta["symbols"]),
            add_blank=bool(meta.get("add_blank", True)),
            language=str(meta.get("language") or "ZH_MIX_EN"),
        )

    step("openepd_g2p_encode", _openepd_g2p, rows)

    # Business MCP stack (as imported by perception main) — best effort
    def _mcp_stack():
        work = Path("/work")
        if str(work) not in sys.path:
            sys.path.insert(0, str(work))
        # Avoid starting servers; import only.
        try:
            import plugins.tts as tts_mod  # noqa: F401

            _ = tts_mod
        except Exception as e:
            print(f"  (plugins.tts skip: {e})", flush=True)
            try:
                import yaml  # noqa: F401
            except Exception:
                pass

    step("import_plugins_tts", _mcp_stack, rows)

    out = Path("/tmp/host_imports.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print("-" * 60)
    print(f"final_rss={rss_mb():.1f}  (no ORT session)")
    print(f"wrote {out}")
    print("\nTOP deltas:")
    for r in sorted(
        [x for x in rows if x["step"] != "start"],
        key=lambda x: x.get("delta_mib", 0),
        reverse=True,
    )[:8]:
        print(f"  {r['delta_mib']:+7.1f}  {r['step']}")


if __name__ == "__main__":
    main()
