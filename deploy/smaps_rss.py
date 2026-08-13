#!/usr/bin/env python3
"""Parse /proc/<pid>/smaps by mapping header lines (not blank-line blocks)."""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+)\s+(\S+)\s+")


def main() -> None:
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    path = Path(f"/proc/{pid}/smaps")
    if not path.is_file():
        raise SystemExit(f"missing {path}")

    text = path.read_text()
    cur = None
    rows = []

    def flush():
        nonlocal cur
        if not cur:
            return
        rows.append(cur)
        cur = None

    for line in text.splitlines():
        m = HEADER.match(line)
        if m:
            flush()
            parts = line.split()
            pathname = "[anon]"
            if len(parts) >= 6:
                pathname = parts[5]
                # smaps can have spaces in path rarely; join rest
                if len(parts) > 6:
                    pathname = " ".join(parts[5:])
            cur = {
                "addr": f"{m.group(1)}-{m.group(2)}",
                "perm": m.group(3),
                "path": pathname,
                "rss_kb": 0,
                "pss_kb": 0,
                "anon_kb": 0,
                "swap_kb": 0,
            }
            continue
        if not cur:
            continue
        if line.startswith("Rss:"):
            cur["rss_kb"] = int(line.split()[1])
        elif line.startswith("Pss:"):
            cur["pss_kb"] = int(line.split()[1])
        elif line.startswith("Anonymous:"):
            cur["anon_kb"] = int(line.split()[1])
        elif line.startswith("Swap:"):
            cur["swap_kb"] = int(line.split()[1])
    flush()

    # keep meaningful resident rows
    kept = [r for r in rows if r["rss_kb"] >= 1024]
    print(f"mappings={len(rows)} with_rss>=1MiB={len(kept)}")
    print("=== top mappings by Rss ===")
    for r in sorted(kept, key=lambda x: -x["rss_kb"])[:40]:
        print(
            f"{r['rss_kb']/1024:8.1f} MiB Rss | "
            f"{r['pss_kb']/1024:8.1f} MiB Pss | "
            f"{r['anon_kb']/1024:8.1f} MiB Anon | "
            f"{r['perm']:5s} | {r['path']}"
        )

    by = defaultdict(lambda: {"rss": 0, "anon": 0})
    for r in kept:
        p = r["path"]
        if p.startswith("/"):
            tag = p.rsplit("/", 1)[-1]
        else:
            tag = p
        by[tag]["rss"] += r["rss_kb"]
        by[tag]["anon"] += r["anon_kb"]

    print("\n=== aggregated by name (Rss) ===")
    for tag, v in sorted(by.items(), key=lambda kv: -kv[1]["rss"])[:30]:
        print(
            f"{v['rss']/1024:8.1f} MiB Rss | {v['anon']/1024:8.1f} MiB Anon | {tag}"
        )

    print("\n=== totals from parsed rows ===")
    print(f"sum Rss:  {sum(r['rss_kb'] for r in rows)/1024:.1f} MiB")
    print(f"sum Anon: {sum(r['anon_kb'] for r in rows)/1024:.1f} MiB")

    rollup = Path(f"/proc/{pid}/smaps_rollup")
    if rollup.is_file():
        print("\n=== smaps_rollup ===")
        for l in rollup.read_text().splitlines():
            if l.startswith(("Rss:", "Pss:", "Anonymous:", "Shared_Clean:", "Private_Dirty:")):
                print(l)


if __name__ == "__main__":
    main()
