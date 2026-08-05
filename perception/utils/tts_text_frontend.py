"""TTS text frontend: acronym expand + lead text_process + WeText.

Pipeline:
  - ALLCAPS Latin tokens (len>=2) → letter-split; other English kept for lexicon
  - wetext + optional lead text_process
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|Wi-?Fi|wi-?fi")


def expand_en_acronyms(text: str) -> str:
    """ALLCAPS Latin tokens (len>=2) → letter-split; other English kept as words."""
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        tok = m.group(0)
        if tok.lower().replace("-", "") == "wifi":
            return "Wi Fi"
        if tok.isupper() and tok.isalpha() and len(tok) >= 2:
            return " ".join(list(tok))
        return tok

    return _TOKEN_RE.sub(_repl, text)


def apply_lead_text_process(text: str, language: str = "zh") -> str:
    """Numbers/units/NSW via lead's text_process (best-effort)."""
    try:
        from utils.text_process.cleaner import text_preprocess

        out = text_preprocess(text, language)
        return out if out is not None else text
    except Exception as e:
        log.warning("[tts] lead text_process skipped: %s", e)
        return text


def apply_wetext(text: str) -> str:
    try:
        from wetext import Normalizer

        tn = getattr(apply_wetext, "_tn", None)
        if tn is None:
            tn = Normalizer(lang="auto", operator="tn")
            apply_wetext._tn = tn  # type: ignore[attr-defined]
        return tn.normalize(text)
    except Exception as e:
        log.warning("[tts] wetext skipped: %s", e)
        return text


def normalize_for_tts(
    text: str,
    *,
    expand_acronyms: bool = True,
    use_text_process: bool = True,
    use_wetext: bool = True,
    language: str = "zh",
) -> str:
    """Full frontend pipeline before VITS synthesize."""
    if not text or not text.strip():
        return text
    original = text
    if expand_acronyms:
        text = expand_en_acronyms(text)
    if use_text_process:
        text = apply_lead_text_process(text, language=language)
    if use_wetext:
        text = apply_wetext(text)
    if text != original:
        log.info(
            "[tts] text frontend (%s->%s chars): %r -> %r",
            len(original),
            len(text),
            original[:64],
            text[:64],
        )
    return text
