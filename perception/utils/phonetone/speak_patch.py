"""Pre-TN speak patches so ranking matches intended Chinese readings.

Runs before PhoneTone FST. Keep rules narrow; prefer explicit patterns over
guessing every slash/minus in the wild.
"""

from __future__ import annotations

import re

# Ranking / docs often use U+2011 NB hyphen etc. FST date/SN rules only match ASCII "-".
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]")

# SI-ish units glued to a number. Avoid bare "A" in model ids (A-10, A1024).
_UNIT_AFTER_DIGIT = (
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*mA\b"), r"\1毫安"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*kA\b"), r"\1千安"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*kV\b"), r"\1千伏"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*kW\b"), r"\1千瓦"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*Hz\b"), r"\1赫兹"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*W\b"), r"\1瓦"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*V\b"), r"\1伏"),
    # Current: only when 电流/安培 nearby, or explicit "数字A" after 电流 already handled.
    (re.compile(r"(电流|安培)\s*(\d+(?:\.\d+)?)\s*A\b"), r"\1\2安"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*A\b(?!\s*[-/])"), r"\1安"),
)

# GPU / product model numbers: force hyphen so TN reads digits one-by-one.
_MODEL_DIGIT = (
    re.compile(r"(?i)\b(RTX|GTX|RX)\s*[-]?\s*(\d{3,5})\b"),
    r"\1-\2",
)

# Acronym slash: MCP/DDS → MCP DDS (pause), never "除".
_ACRONYM_SLASH = re.compile(r"\b([A-Z]{2,})(?:/([A-Z]{2,}))+\b")

# Formula equals without relying on TN digit gate.
_EQ = re.compile(r"([A-Za-z0-9\)）])\s*=\s*([A-Za-z0-9\(（])")

# After 等于, short latin product (F=ma) → spaced letters.
_EQ_LATIN = re.compile(r"等于([a-z]{1,6})\b")

# MR-0719 style: keep letters split so CMU never says "mister".
_MR_CODE = re.compile(r"\bMR\s*[-–—]?\s*(\d+)\b", re.I)

# "订单 ID：8492610" is cardinal without a serial cue; TN needs 订单号/编号.
_ORDER_ID = re.compile(r"订单\s*ID\s*[:：]?\s*", re.I)

# Single-letter algebra minus: a-b → a减b.
# Do not use \\b: Python \\w treats CJK as word chars, so a-b等于… would miss.
# ASCII lookaround keeps SN-73049 / example-site / A-10 untouched.
_LETTER_MINUS = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])\s*-\s*([A-Za-z])(?![A-Za-z0-9])")

# Tail / explicit digit-serial cues already handled by TN for 尾号; reinforce 编号.
_SERIAL_SPACED = re.compile(
    r"(尾号|编号|型号|工号|卡号|券码|验证码|订单号|物流单号)\s*([0-9]{2,})",
)


def _expand_acronym_slash(match: re.Match) -> str:
    return " ".join(match.group(0).split("/"))


def patch_speak_text(text: str) -> str:
    """Rewrite text before FST TN / G2P."""
    if not text:
        return text
    out = _DASH_RE.sub("-", text)

    out = _ORDER_ID.sub("订单号", out)
    out = _MODEL_DIGIT[0].sub(_MODEL_DIGIT[1], out)
    out = _ACRONYM_SLASH.sub(_expand_acronym_slash, out)
    out = _MR_CODE.sub(lambda m: f"M R-{m.group(1)}", out)
    out = _EQ.sub(r"\1等于\2", out)
    out = _EQ_LATIN.sub(lambda m: "等于" + " ".join(m.group(1).upper()), out)
    # After EQ expands ma; single-letter 减 only (not SN-73049 / example-site).
    out = _LETTER_MINUS.sub(r"\1减\2", out)

    for pattern, repl in _UNIT_AFTER_DIGIT:
        out = pattern.sub(repl, out)

    # Ensure serial cue + digits stay adjacent for TN digit verbalizer.
    out = _SERIAL_SPACED.sub(r"\1\2", out)
    return out
