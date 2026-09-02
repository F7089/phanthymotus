"""PhoneTone V1 runtime for gentleman Matcha ranking."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .frontend import PhoneToneResult, prepare_phonetone

MAX_TOKENS = int(os.environ.get("TTS_TRT_MAX_TOKENS", "256"))
MIN_TOKENS = int(os.environ.get("TTS_TRT_MIN_TOKENS", "8"))


def is_phonetone_dir(model_dir: str) -> bool:
    return Path(model_dir, "frontend_release", "opencpop-strict.txt").is_file()


def configure_release(model_dir: str) -> str:
    root = str(Path(model_dir) / "frontend_release")
    os.environ["MATCHA_FRONTEND_RELEASE"] = root
    os.environ.setdefault("MATCHA_SAMPLE_RATE", "16000")
    if not Path(root, "opencpop-strict.txt").is_file():
        raise FileNotFoundError("PhoneTone frontend_release missing under %s" % root)
    return root


def intersperse(values, item=0):
    out = [item] * (len(values) * 2 + 1)
    out[1::2] = list(values)
    return out


def encode_for_matcha(text: str, temperature: float = 0.667, length_scale: float = 1.0) -> dict:
    result = prepare_phonetone((text or "").strip())
    phone_ids = intersperse(result.phone_ids, 0)
    tone_ids = intersperse(result.tone_ids, 0)
    language_ids = intersperse(result.language_ids, 0)
    if len(phone_ids) > MAX_TOKENS:
        phone_ids = phone_ids[:MAX_TOKENS]
        tone_ids = tone_ids[:MAX_TOKENS]
        language_ids = language_ids[:MAX_TOKENS]
    real_len = len(phone_ids)
    if real_len < MIN_TOKENS:
        pad = MIN_TOKENS - real_len
        phone_ids = phone_ids + [0] * pad
        tone_ids = tone_ids + [0] * pad
        language_ids = language_ids + [0] * pad
    return {
        "normalized_text": result.normalized_text,
        "phones": result.phones,
        "x": np.asarray(phone_ids, np.int64)[None, :],
        "x_lengths": np.asarray([real_len], np.int64),
        "tones": np.asarray(tone_ids, np.int64)[None, :],
        "languages": np.asarray(language_ids, np.int64)[None, :],
        "scales": np.asarray([temperature, length_scale], np.float32),
        "real_len": real_len,
    }


class PhoneToneFrontend:
    def __init__(self, model_dir: str):
        self.release = configure_release(model_dir)

    @property
    def has_wetext(self) -> bool:
        return Path(self.release, "tn_cache", "zh_tn_tagger.fst").is_file()

    def normalize(self, text: str) -> str:
        return prepare_phonetone((text or "").strip()).normalized_text
