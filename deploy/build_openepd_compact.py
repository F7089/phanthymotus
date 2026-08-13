#!/usr/bin/env python3
"""Build mmap OpenEPD lexicon on the JuiceFS / data host (NOT Jetson).

  python3 deploy/build_openepd_compact.py

Reads:
  /mnt/data/fanyi/phanthymotus_tts/melo-openepd-g2p-assets/openepd_eng_dict.pickle
  (or extracts from melo-openepd-g2p-assets.tar.bz2)
Writes:
  /mnt/data/fanyi/phanthymotus_tts/openepd_eng_dict.oedb
HTTP:
  http://172.28.4.81:34567/fanyi/phanthymotus_tts/openepd_eng_dict.oedb
"""
from __future__ import annotations

import os
import sys
import tarfile
import tempfile
from pathlib import Path

# Allow running from repo or /tmp copy
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "melo_g2p_slim"))
sys.path.insert(0, str(_HERE.parent / "perception" / "melo_g2p_slim"))

from openepd_compact import build_compact  # noqa: E402


def _find_pickle(juice: Path) -> Path:
    direct = [
        juice / "openepd_eng_dict.pickle",
        juice / "melo-openepd-g2p-assets" / "openepd_eng_dict.pickle",
        Path("/mnt/data/fanyi/melo_training/repos/MeloTTS/melo/text/openepd_eng_dict.pickle"),
    ]
    for p in direct:
        if p.is_file():
            return p
    tar = juice / "melo-openepd-g2p-assets.tar.bz2"
    if not tar.is_file():
        raise SystemExit(f"missing pickle and tar under {juice}")
    tmp = Path(tempfile.mkdtemp(prefix="oedb_"))
    with tarfile.open(tar, "r:bz2") as tf:
        member = None
        for m in tf.getmembers():
            if m.name.endswith("openepd_eng_dict.pickle") and m.isfile():
                member = m
                break
        if member is None:
            raise SystemExit("pickle not inside tar")
        tf.extract(member, tmp)
    return tmp / member.name


def main() -> None:
    juice = Path(os.environ.get("MELO_JUICE_ROOT", "/mnt/data/fanyi/phanthymotus_tts"))
    out = Path(
        os.environ.get(
            "MELO_OPENEPD_OEDB",
            str(juice / "openepd_eng_dict.oedb"),
        )
    )
    src = _find_pickle(juice)
    print(f"in={src}")
    meta = build_compact(src, out)
    print(meta)
    # also place next to assets dir for local mounts
    side = juice / "melo-openepd-g2p-assets" / "openepd_eng_dict.oedb"
    if (juice / "melo-openepd-g2p-assets").is_dir():
        import shutil

        shutil.copy2(out, side)
        print(f"also={side}")
    print(f"HTTP: http://172.28.4.81:34567/fanyi/phanthymotus_tts/{out.name}")


if __name__ == "__main__":
    main()
