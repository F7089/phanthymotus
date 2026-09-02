"""PhoneTone V1 frontend (jieba + ToneSandhi + 94 phone)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import jieba
import jieba.posseg as psg
from pypinyin import Style, lazy_pinyin, load_phrases_dict

from .fst_tn import FstNormalizer, apply_overlay, release_root, transliterate_non_cjk
from .heteronym import custom_dict, jieba_phrases
from .symbols import (
    language_id_map,
    language_tone_start_map,
    num_languages,
    num_tones,
    punctuation,
    symbols,
)
from .tone_sandhi import ToneSandhi

_SYMBOL_TO_ID = {symbol: index for index, symbol in enumerate(symbols)}
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z]+(?:['-][A-Za-z]+)*|[!?…,.'-]")
_ARPA_RE = re.compile(r"^([A-Z]+)([012])?$")
_INITIALS = (
    "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s", "y", "w",
)
_SANDHI_PROTECTED = {"得了满分"}


@dataclass(frozen=True)
class PhoneToneResult:
    normalized_text: str
    phones: tuple[str, ...]
    phone_ids: tuple[int, ...]
    tone_ids: tuple[int, ...]
    language_ids: tuple[int, ...]

    def __post_init__(self):
        if not (len(self.phone_ids) == len(self.tone_ids) == len(self.language_ids)):
            raise ValueError("phone/tone/language sequences must be equal length")


def _release() -> Path:
    return release_root()


@lru_cache(maxsize=1)
def _assets():
    root = _release()
    mapping = {}
    for line in (root / "opencpop-strict.txt").read_text(encoding="utf-8").splitlines():
        pinyin, phones = line.split("\t", 1)
        mapping[pinyin] = phones.split()
    custom_en = json.loads((root / "custom_en_pronunciations.json").read_text(encoding="utf-8"))
    cmu = {}
    with (root / "cmudict.rep").open(encoding="utf-8") as source:
        for line in source:
            if line.startswith("##") or "  " not in line:
                continue
            word, pronunciation = line.rstrip().split("  ", 1)
            cmu.setdefault(re.sub(r"\(\d+\)$", "", word), pronunciation.replace(" - ", " ").split())
    load_phrases_dict(custom_dict, style="tone2")
    for phrase in jieba_phrases:
        jieba.add_word(phrase)
    return mapping, custom_en, cmu, FstNormalizer(root / "tn_cache"), ToneSandhi()


def _zh_phones(text: str, gold_lexical_pinyin: Sequence[str] | None = None):
    mapping, _, _, _, sandhi = _assets()
    phones, tones = [], []
    lexical = lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    lexical = apply_overlay(text, lexical, _lexical_overlay())
    if gold_lexical_pinyin is not None:
        if len(gold_lexical_pinyin) != len(text):
            raise ValueError("Gold lexical pinyin must align one-to-one with Chinese text")
        lexical = list(gold_lexical_pinyin)
    protected = {
        index + offset
        for phrase in _SANDHI_PROTECTED
        for index in range(len(text))
        if text.startswith(phrase, index)
        for offset in range(len(phrase))
    }
    cursor = 0
    for word, pos in sandhi.pre_merge_for_modify(psg.lcut(text)):
        start = cursor
        full = lexical[start : start + len(word)]
        cursor += len(word)
        word_raw = lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)
        explicit = [
            raw != fixed or start + index in protected
            for index, (raw, fixed) in enumerate(zip(word_raw, full))
        ]
        finals = lazy_pinyin(word, style=Style.FINALS_TONE3, neutral_tone_with_five=True)
        finals = [final[:-1] + tone3[-1] for final, tone3 in zip(finals, full)]
        finals = sandhi.modified_tone(word, pos, finals)
        finals = [
            final[:-1] + full[index][-1] if explicit[index] else final
            for index, final in enumerate(finals)
        ]
        for tone3, modified_final in zip(full, finals):
            if not tone3 or not modified_final or modified_final[-1] not in "12345":
                raise ValueError(f"invalid Chinese TONE3: {word!r} -> {tone3!r}")
            tone = int(modified_final[-1])
            syllable = tone3[:-1]
            initial = next((value for value in _INITIALS if syllable.startswith(value)), "")
            raw_final = syllable[len(initial) :]
            syllable = initial + {"uei": "ui", "iou": "iu", "uen": "un"}.get(raw_final, raw_final)
            if not initial:
                syllable = {"ing": "ying", "i": "yi", "in": "yin", "u": "wu"}.get(syllable, syllable)
                if syllable and syllable[0] in "viu" and syllable not in mapping:
                    syllable = {"v": "yu", "i": "y", "u": "w"}[syllable[0]] + syllable[1:]
            if syllable not in mapping:
                raise ValueError(f"Chinese pinyin OOV: {syllable!r} ({word!r})")
            unit = mapping[syllable]
            phones.extend(unit)
            tones.extend([tone] * len(unit))
    return phones, tones


@lru_cache(maxsize=1)
def _lexical_overlay() -> dict[str, list[str]]:
    from pypinyin.contrib.tone_convert import to_tone3

    return {
        phrase: [to_tone3(item[0], neutral_tone_with_five=True) for item in values]
        for phrase, values in custom_dict.items()
    }


def _arpa_to_phone(value: str):
    match = _ARPA_RE.fullmatch(value)
    if not match:
        raise ValueError(f"invalid ARPAbet phone: {value!r}")
    phone, stress = match.groups()
    symbol = "V" if phone == "V" else phone.lower()
    if symbol not in _SYMBOL_TO_ID:
        raise ValueError(f"English phone OOV: {symbol!r}")
    return symbol, int(stress) + 1 if stress is not None else 3


@lru_cache(maxsize=1)
def _g2p():
    import nltk
    from g2p_en import G2p

    local_nltk = str(_release() / "nltk_data")
    if local_nltk not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk)
    return G2p()


def _en_phones(word: str):
    _, custom_en, cmu, _, _ = _assets()
    pronunciation = custom_en.get(word.upper()) or cmu.get(word.upper())
    if pronunciation is None:
        pronunciation = [value for value in _g2p()(word) if value != " "]
    converted = [_arpa_to_phone(value) for value in pronunciation if _ARPA_RE.fullmatch(value)]
    if not converted:
        raise ValueError(f"English G2P produced no phones: {word!r}")
    return [x[0] for x in converted], [x[1] for x in converted]


def prepare_phonetone(text: str, gold_lexical_pinyin: Sequence[str] | None = None) -> PhoneToneResult:
    _, _, _, normalizer, _ = _assets()
    normalized = transliterate_non_cjk(normalizer(text)).replace("嗯", "恩").replace("呣", "母")
    if gold_lexical_pinyin is not None and len(gold_lexical_pinyin) != len(normalized):
        raise ValueError(
            f"Gold/text alignment mismatch: {len(gold_lexical_pinyin)} != {len(normalized)}"
        )
    phones, raw_tones, langs = ["_"], [0], ["ZH"]
    for match in _TOKEN_RE.finditer(normalized):
        token = match.group()
        if _CJK_RE.fullmatch(token):
            token_gold = gold_lexical_pinyin[match.start() : match.end()] if gold_lexical_pinyin else None
            token_phones, token_tones = _zh_phones(token, token_gold)
            language = "ZH"
        elif token in punctuation:
            token_phones, token_tones, language = [token], [0], "ZH"
        else:
            token_phones, token_tones = _en_phones(token)
            language = "EN"
        phones.extend(token_phones)
        raw_tones.extend(token_tones)
        langs.extend([language] * len(token_phones))
    phones.append("_")
    raw_tones.append(0)
    langs.append("ZH")
    phone_ids = tuple(_SYMBOL_TO_ID[phone] for phone in phones)
    tone_ids = tuple(tone + language_tone_start_map[lang] for tone, lang in zip(raw_tones, langs))
    language_ids = tuple(language_id_map[lang] for lang in langs)
    if any(tone >= num_tones for tone in tone_ids) or any(lang >= num_languages for lang in language_ids):
        raise ValueError("PhoneTone ID exceeds frozen VITS-compatible range")
    return PhoneToneResult(normalized, tuple(phones), phone_ids, tone_ids, language_ids)
