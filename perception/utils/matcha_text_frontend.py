#!/usr/bin/env python3
"""Matcha text frontend used before sherpa lexicon/espeak.

Generic rules, not a word list:
  1. Isolated Latin letter in mixed text → uppercase (质量 m → 质量 M).
  2. Short Latin runs touching formula symbols / 等于 → spaced uppercase
     (F=ma / F等于ma → F = M A / F 等于 M A). English words like open stay.
  3. Optional WeText fst: numbers, dates, ALLCAPS (CTO → C T O).

Do not also run sherpa phone/date/number FSTs when WeText is on.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)

_OPS = r"=+\*/^×÷＝"
_OP = re.compile(rf"[{re.escape(_OPS)}]|等于")
_CHUNK = re.compile(
    rf"([A-Za-z]{{1,8}})?(\s*)({_OP.pattern})(\s*)([A-Za-z]{{1,8}})?"
)
_ISOLATED = re.compile(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")


def _should_split(run: str) -> bool:
    if not run:
        return False
    if len(run) == 1:
        return True
    if run.isupper() and 2 <= len(run) <= 6:
        return True
    if run.islower() and len(run) <= 2:
        return True
    return False


def _letters(run: str) -> str:
    if _should_split(run):
        return " ".join(run.upper())
    return run


def expand_latin_letters(text: str) -> str:
    def around_op(m: re.Match[str]) -> str:
        left, sp1, op, sp2, right = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        if not left and not right:
            return m.group(0)
        parts = []
        if left:
            parts.append(_letters(left))
        parts.append(op if op == "等于" else f" {op} ")
        if right:
            parts.append(_letters(right))
        out = " ".join(p.strip() for p in parts if p.strip())
        if not left:
            out = (sp1 or " ") + out
        if not right:
            out = out + (sp2 or " ")
        return f" {out} "

    text = _CHUNK.sub(around_op, text)
    text = _ISOLATED.sub(lambda m: f" {m.group(1).upper()} ", text)
    return re.sub(r" {2,}", " ", text).strip()


class WeTextNormalizer:
    def __init__(self, tn_dir: str | Path):
        import kaldifst

        tn_dir = Path(tn_dir)
        tagger_p = tn_dir / "zh_tn_tagger.fst"
        verb_p = tn_dir / "zh_tn_verbalizer.fst"
        if not tagger_p.is_file() or not verb_p.is_file():
            raise FileNotFoundError(f"missing WeText fst under {tn_dir}")
        self._tagger = kaldifst.TextNormalizer(str(tagger_p))
        self._verbalizer = kaldifst.TextNormalizer(str(verb_p))

    def __call__(self, text: str) -> str:
        tagged = self._tagger(text)
        try:
            from wetext import token_parser

            original_escape = token_parser.escape_value
            try:
                token_parser.escape_value = lambda value: value
                reordered = token_parser.TokenParser("zh", "tn").reorder(tagged)
            finally:
                token_parser.escape_value = original_escape
        except Exception as e:
            log.warning("WeText TokenParser skipped: %s", e)
            reordered = tagged
        return self._verbalizer(reordered)


def resolve_wetext_dir(path: str | Path | None = None) -> Path | None:
    candidates = []
    if path:
        candidates.append(Path(path))
    env = os.environ.get("WETEXT_DIR", "").strip()
    if env:
        candidates.append(Path(env))
    for c in candidates:
        if (c / "zh_tn_tagger.fst").is_file() and (c / "zh_tn_verbalizer.fst").is_file():
            return c
    return None


class MatchaTextFrontend:
    """Letter rules always; WeText if fst dir is present."""

    def __init__(self, wetext_dir: str | Path | None = None):
        resolved = resolve_wetext_dir(wetext_dir)
        self.wetext_dir = resolved
        self._wetext = WeTextNormalizer(resolved) if resolved else None

    @property
    def has_wetext(self) -> bool:
        return self._wetext is not None

    def normalize(self, text: str, *, trace: bool = False) -> str:
        if not text or not text.strip():
            return text
        letters = expand_latin_letters(text)
        if trace:
            print("letters:", letters)
        if self._wetext is None:
            if trace:
                print("wetext: (off)")
                print("frontend:", letters)
            return letters
        tn = self._wetext(letters)
        if trace:
            print("wetext:", tn)
        frontend = expand_latin_letters(tn)
        if trace:
            print("frontend:", frontend)
        return frontend


if __name__ == "__main__":
    samples = [
        ("牛顿第二定律是 F=ma ，质量 m 乘以加速度 a 等于力 F。", "F = M A", "质量 M"),
        ("please open the meeting.", "open", None),
        ("open=true", "open = true", None),
        ("公司的 CTO 和 CEO 明天开会，AI 和 GPT 也会讨论。", "CTO", None),
    ]
    for src, must, must2 in samples:
        out = expand_latin_letters(src)
        assert must in out, (src, out)
        if must2:
            assert must2 in out, (src, out)
        print("OK", out)
    print("letter rules ok; WeText loads only when fst dir is set")
