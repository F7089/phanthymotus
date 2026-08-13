#!/usr/bin/env python3
"""Compare slim.predict vs original g2p_en G2p.predict on OpenEPD OOVs.

Success criterion (GPT): phoneme sequence match rate ~100% on OOV set.

Inside container (needs g2p_en installed for reference only):
  python3 /tmp/bench_tts_oov_parity.py | tee /tmp/oov_parity.txt
"""
from __future__ import annotations

import os
import random
import re
import sys
from pathlib import Path


def main() -> None:
    g2p_assets = Path("/models/melo-openepd-g2p-assets")
    text_dir = g2p_assets / "vendor" / "melo_g2p" / "text"
    sys.path.insert(0, str(text_dir))

    import pickle

    from slim_g2p_oov import SlimG2pOov, predict_oov

    # Reference: official g2p_en predict only (still loads nltk once here)
    import g2p_en
    from g2p_en.g2p import G2p

    def _resolve_ckpt() -> str:
        cand = [
            os.environ.get("MELO_G2P_OOV_CKPT", ""),
            str(text_dir / "checkpoint20.npz"),
            str(Path(g2p_en.__file__).resolve().parent / "checkpoint20.npz"),
        ]
        for p in cand:
            if p and os.path.isfile(p):
                return p
        raise FileNotFoundError(
            "checkpoint20.npz missing; run: bash deploy/patch_melo_slim_g2p.sh"
        )

    class _Ref(G2p):
        def __init__(self):
            # Bypass cmudict/homograph init cost path partially — still imports nltk
            # at module level. We only call .predict().
            super().__init__()

    print("loading OpenEPD for OOV sampling...")
    with open(g2p_assets / "openepd_eng_dict.pickle", "rb") as f:
        eng = pickle.load(f)
    keys = set(eng.keys()) if hasattr(eng, "keys") else set()
    print(f"openepd_entries={len(keys)}")

    # Candidate OOV-ish words (not in OpenEPD). Mix categories.
    seed_words = [
        # coined / tech
        "activationist",
        "tokenisation",
        "quantisation",
        "dockerized",
        "microfrontend",
        "vectorizer",
        "backpropagator",
        "onnxruntime",
        "jetson",
        "phanthymotus",
        "openepd",
        "melotts",
        "longanlingxin",
        "cuda",
        "tensorrt",
        "whisperx",
        "langchain",
        "pytorch",
        "huggingface",
        "autograd",
        # brands / products
        "nvidia",
        "qualcomm",
        "bytedance",
        "tiktok",
        "chatgpt",
        "deepseek",
        "anthropic",
        "cursor",
        "notion",
        "figma",
        # names / places-ish
        "kyubyong",
        "szechuan",
        "shenzhen",
        "hangzhou",
        "yokohama",
        "reykjavik",
        # odd spellings / hyphen-derived stems
        "colourised",
        "favouritable",
        "reimplemented",
        "overparameterized",
        "underfitting",
        "semisupervised",
        "multilingualism",
        "speechification",
        "vocoderless",
        "grapheme",
        "phonemicize",
        "heteronymic",
        "disambiguator",
        "unpickleable",
        # plurals / variants
        "activationists",
        "tokenizers",
        "quantizers",
        "dockerizing",
        "microfrontends",
    ]
    # Generate more letter-soup OOVs for volume
    rng = random.Random(0)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    while len(seed_words) < 250:
        n = rng.randint(6, 12)
        w = "".join(rng.choice(alphabet) for _ in range(n))
        if w.upper() not in keys and w not in seed_words:
            seed_words.append(w)

    oovs = []
    for w in seed_words:
        if w.upper() not in keys:
            oovs.append(w)
    oovs = oovs[:250]
    print(f"test_oovs={len(oovs)}")

    ckpt = _resolve_ckpt()
    os.environ["MELO_G2P_OOV_CKPT"] = ckpt
    print(f"ckpt={ckpt}")
    print("init slim + reference G2p (ref loads nltk once)...")
    slim = SlimG2pOov(ckpt=ckpt)
    ref = _Ref()

    mismatch = []
    for i, w in enumerate(oovs):
        a = slim.predict(w)
        b = ref.predict(w)
        if a != b:
            mismatch.append((w, a, b))
        if (i + 1) % 50 == 0:
            print(f"  checked {i+1}/{len(oovs)} mismatches={len(mismatch)}")

    n = len(oovs)
    ok = n - len(mismatch)
    rate = 100.0 * ok / n if n else 0.0
    print("\nRESULT:")
    print(f"  matched={ok}/{n}  ({rate:.2f}%)")
    print(f"  mismatched={len(mismatch)}")
    for w, a, b in mismatch[:20]:
        print(f"  FAIL {w!r}")
        print(f"    slim={a}")
        print(f"    ref ={b}")
    if rate < 99.5:
        raise SystemExit(1)
    print("PASS: slim predict aligns with g2p_en.predict (>=99.5%)")


if __name__ == "__main__":
    main()
