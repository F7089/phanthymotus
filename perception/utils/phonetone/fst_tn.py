"""WeText / Kaldifst TN helpers vendored from PhoneTone zh_en_frontend."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path

PUNCTUATION_MAP = {
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "；": ";",
    "：": ":",
    "、": ",",
    "《": '"',
    "》": '"',
    "【": "[",
    "】": "]",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
}
TONE3_RE = re.compile(r"[a-züv]+[1-5]")


def _unidecode(char: str) -> str:
    try:
        from unidecode import unidecode

        return unidecode(char)
    except Exception:
        return char if char.isascii() else ""


class FstNormalizer:
    _lock = threading.RLock()

    def __init__(self, root: Path):
        import kaldifst

        manifest = json.loads((root / "tn_manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported TN manifest")
        paths = []
        for name in ("zh_tn_tagger.fst", "zh_tn_verbalizer.fst"):
            path, meta = root / name, manifest["fst"][name]
            if path.stat().st_size != meta["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != meta["sha256"]:
                raise ValueError(f"TN checksum mismatch: {name}")
            paths.append(str(path))
        self.tagger, self.verbalizer = map(kaldifst.TextNormalizer, paths)

    def __call__(self, text: str) -> str:
        from wetext import token_parser

        tagged = self.tagger(text)
        with self._lock:
            old = token_parser.escape_value
            try:
                token_parser.escape_value = lambda value: value
                reordered = token_parser.TokenParser("zh", "tn").reorder(tagged)
            finally:
                token_parser.escape_value = old
        return self.verbalizer(reordered)


def apply_overlay(text: str, tokens: list[str], overlay: dict) -> list[str]:
    if len(tokens) != len(text):
        raise ValueError("Chinese pinyin alignment failed")
    phrases = sorted(overlay, key=len, reverse=True)
    for phrase in phrases:
        expected = overlay[phrase]
        if len(expected) != len(phrase) or not all(TONE3_RE.fullmatch(value) for value in expected):
            raise ValueError(f"invalid pinyin overlay: {phrase}")
    out, index = list(tokens), 0
    while index < len(text):
        phrase = next((value for value in phrases if text.startswith(value, index)), None)
        if phrase:
            out[index : index + len(phrase)] = overlay[phrase]
            index += len(phrase)
        else:
            index += 1
    return out


def transliterate_non_cjk(text: str) -> str:
    return "".join(
        PUNCTUATION_MAP.get(
            char,
            char if "\u4e00" <= char <= "\u9fff" or char.isascii() else _unidecode(char),
        )
        for char in text
    )


def release_root(explicit: str | None = None) -> Path:
    path = explicit or os.environ.get("MATCHA_FRONTEND_RELEASE", "")
    if not path:
        raise RuntimeError("MATCHA_FRONTEND_RELEASE is not set")
    return Path(path)
