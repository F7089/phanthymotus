"""Pre-TN speak patches so ranking matches intended Chinese readings.

Runs before PhoneTone FST. Keep rules narrow; prefer explicit patterns over
guessing every slash/minus in the wild.
"""

from __future__ import annotations

import re

# Ranking / docs often use U+2011 NB hyphen etc. FST date/SN rules only match ASCII "-".
_DASH_RE = re.compile(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d]")

# Hard replacements first — never rely on TN for these slash pairs.
# 企业级 + noun: Matcha lengthens the blank after 级. 的 changes that phone context.
# 企业级AI stays a language switch if 的 is inserted in front of AI, so read AI in Chinese.
_HARD_REPLACEMENTS = (
    (re.compile(r"(?i)MCP\s*[/／∕⁄]\s*DDS"), "MCP DDS"),
    (re.compile(r"(?i)TCP\s*[/／∕⁄]\s*IP"), "TCP IP"),
    # NO.0917 → number 0917 (TN then reads digits one-by-one).
    (re.compile(r"(?i)\bNO\s*\.\s*(?=\d)"), "number "),
    (re.compile(r"企业级的?\s*AI(?![A-Za-z])"), "企业级的人工智能"),
    (re.compile(r"企业级(?!的)(?=[\u4e00-\u9fff])"), "企业级的"),
)

# SI-ish units glued to a number. Avoid bare "A" in model ids (A-10, A1024).
_UNIT_AFTER_DIGIT = (
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*mA\b"), r"\1毫安"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*kA\b"), r"\1千安"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*kV\b"), r"\1千伏"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*kW\b"), r"\1千瓦"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*Hz\b"), r"\1赫兹"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*W\b"), r"\1瓦"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*V\b"), r"\1伏"),
    (re.compile(r"(电流|安培)\s*(\d+(?:\.\d+)?)\s*A\b"), r"\1\2安"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*A\b(?!\s*[-/])"), r"\1安"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*GB\b"), r"\1吉字节"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*MB\b"), r"\1兆字节"),
    (re.compile(r"(?i)(?<![A-Za-z])(\d+(?:\.\d+)?)\s*KB\b"), r"\1千字节"),
)

# GPU / product model numbers: force hyphen so TN reads digits one-by-one.
_MODEL_DIGIT = (
    re.compile(r"(?i)\b(RTX|GTX|RX)\s*[-]?\s*(\d{3,5})\b"),
    r"\1-\2",
)

# Acronym slash: MCP/DDS → MCP DDS. UPPERCASE only — re.I would eat URL paths
# like example.com/help → "com help" and drop the path slash reading.
_SLASH_CHARS = r"/／∕⁄"
_ACRONYM_SLASH = re.compile(
    rf"(?<![A-Za-z0-9])([A-Z]{{2,}}(?:\s*[{_SLASH_CHARS}]\s*[A-Z]{{2,}})+)(?![A-Za-z0-9])"
)

# Formula equals without relying on TN digit gate.
_EQ = re.compile(r"([A-Za-z0-9\)）])\s*=\s*([A-Za-z0-9\(（])")

# After 等于, short latin product (F=ma) → spaced letters.
_EQ_LATIN = re.compile(r"等于([a-z]{1,6})\b")

# MR-0719 style: keep letters split so CMU never says "mister".
_MR_CODE = re.compile(r"\bMR\s*[-–—]?\s*(\d+)\b", re.I)

# "订单 ID：8492610" is cardinal without a serial cue; TN needs 订单号/编号.
_ORDER_ID = re.compile(r"订单\s*ID\s*[:：]?\s*", re.I)

# Product/model hyphen with a letter on the right (GPT-4o, Qwen3-TTS).
# Pure digit tails (RTX-4090, SN-73049) stay hyphenated for TN serial reading.
_PRODUCT_HYPHEN = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{2,}\d*)-([0-9]*[A-Za-z][A-Za-z0-9]*)(?![A-Za-z0-9])"
)
# 2.5-TTS / 4-mini: digit-hyphen-letters is a version, not 减.
_VERSION_LETTER_HYPHEN = re.compile(
    r"(\d)-([A-Za-z]{2,})(?![A-Za-z0-9])"
)
# Qwen3 / 4o / USB3. Scientific 1e-4 is expanded before this so it stays intact.
_LETTER_DIGIT = re.compile(r"([A-Za-z])(\d)")
_DIGIT_LETTER = re.compile(r"(\d)([A-Za-z])")
_SCI_NUM = re.compile(
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![A-Za-z0-9])"
)
# Mixtral-8x7B / 640x480 ASCII multiply. Unicode × is already handled by FST.
_DIGIT_X_DIGIT = re.compile(r"(?<=\d)[xX](?=\d)")
# 2x-3: keep 减 before letter/digit spacing (otherwise FST drops the minus).
_ALGEBRA_VAR_MINUS = re.compile(r"(?<=\d)([A-Za-z])-(\d)")
_URL_HOLD_RE = re.compile(r"\[\[\[U([A-Z]+)\]\]\]")

# Single-letter algebra minus: a-b → a减b.
# Do not use \\b: Python \\w treats CJK as word chars, so a-b等于… would miss.
_LETTER_MINUS = re.compile(r"(?<![A-Za-z0-9])([A-Za-z])\s*-\s*([A-Za-z])(?![A-Za-z0-9])")

# Hex / key material A7-F2-9D: TN would say 减; force spaced tokens instead.
# Require ≥1 A-F so date fragments like 12-05 / 26-12-05 are not rewritten.
_HEX_BYTES = re.compile(
    r"(?<![A-Za-z0-9])([0-9A-Fa-f]{2}(?:-[0-9A-Fa-f]{2})+)(?![A-Za-z0-9])"
)
_HEX_LETTER = re.compile(r"[A-Fa-f]")

# Tail / explicit digit-serial cues already handled by TN for 尾号; reinforce 编号.
_SERIAL_SPACED = re.compile(
    r"(尾号|编号|型号|工号|卡号|券码|验证码|订单号|物流单号)\s*([0-9]{2,})",
)

# "最后四位是 5690" is cardinal without a serial cue; space digits so TN
# reads 五 六 九 零 (no redundant 尾号/编号).
_LAST_N_DIGITS = re.compile(
    r"((?:最后|末尾|后)\s*[一二三四五六七八九十两\d]+\s*位)\s*[是为:：]?\s*([0-9]{2,})"
)

# Acronym+≥+number+unit is tagged as math "…二" plus measure "八分贝".
# Expand the operator first so 28 stays one cardinal (二十八). Only when a
# digit follows, so arrows like -> are left alone.
_CMP_BEFORE_NUM = (
    (re.compile(r">=\s*(?=\d)"), "大于等于"),
    (re.compile(r"≤\s*(?=\d)|<=\s*(?=\d)"), "小于等于"),
    (re.compile(r"≥\s*(?=\d)"), "大于等于"),
    (re.compile(r"(?<![<>=])>\s*(?=\d)"), "大于"),
    (re.compile(r"(?<![<>=])<\s*(?=\d)"), "小于"),
)

# Hold URLs so later slash/minus rules cannot rewrite path segments.
_URL_HOLD = re.compile(r"(?i)\bhttps?://[^\s，。；！？,]+|\bwww\.[^\s，。；！？,]+")
_URL_PARSE = re.compile(r"(?i)^(https?)://([^/\s]+)(/.*)?$")


def _expand_acronym_slash(match: re.Match) -> str:
    parts = re.split(rf"\s*[{_SLASH_CHARS}]\s*", match.group(0))
    return " ".join(p for p in parts if p)


def _expand_hex_bytes(match: re.Match) -> str:
    value = match.group(0)
    if not _HEX_LETTER.search(value):
        return value
    return " ".join(value.split("-"))


def _space_last_n_digits(match: re.Match) -> str:
    cue, digits = match.group(1), match.group(2)
    return f"{cue}是 {' '.join(digits)}"


def _alpha_id(n: int) -> str:
    n += 1
    chars: list[str] = []
    while n:
        n, remainder = divmod(n - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _alpha_id_value(tag: str) -> int:
    value = 0
    for char in tag:
        value = value * 26 + (ord(char) - 64)
    return value - 1


def _expand_sci(match: re.Match) -> str:
    base, exp = match.group(1), match.group(2)
    if exp.startswith("-"):
        sign, digits = "负", exp[1:]
    elif exp.startswith("+"):
        sign, digits = "", exp[1:]
    else:
        sign, digits = "", exp
    return f"{base}乘十的{sign}{digits}次方"


def _verbalize_url(url: str) -> str:
    """Force protocol/path slashes to 斜杠 so TN cannot drop them."""
    match = _URL_PARSE.match(url)
    if match is None:
        return url.replace("/", " 斜杠 ").replace(".", " 点 ")
    scheme, host, path = match.group(1), match.group(2), match.group(3) or ""
    scheme_cn = " ".join(scheme.upper()) + " 冒号 斜杠 斜杠"
    host_cn = re.sub(r"(?i)\bwww\b", "W W W", host).replace(".", " 点 ")
    path_parts = [part for part in path.split("/") if part]
    path_cn = "".join(f" 斜杠 {part}" for part in path_parts)
    return re.sub(r"\s+", " ", f"{scheme_cn} {host_cn}{path_cn}").strip()


def patch_speak_text(text: str) -> str:
    """Rewrite text before FST TN / G2P."""
    if not text:
        return text
    out = _DASH_RE.sub("-", text)

    held: list[str] = []

    def _hold_url(match: re.Match) -> str:
        held.append(match.group(0))
        return f"[[[U{_alpha_id(len(held) - 1)}]]]"

    out = _URL_HOLD.sub(_hold_url, out)

    for pattern, repl in _HARD_REPLACEMENTS:
        out = pattern.sub(repl, out)

    out = _ORDER_ID.sub("订单号", out)
    out = _LAST_N_DIGITS.sub(_space_last_n_digits, out)
    out = _MODEL_DIGIT[0].sub(_MODEL_DIGIT[1], out)
    out = _HEX_BYTES.sub(_expand_hex_bytes, out)
    out = _PRODUCT_HYPHEN.sub(r"\1 \2", out)
    out = _VERSION_LETTER_HYPHEN.sub(r"\1 \2", out)
    out = _SCI_NUM.sub(_expand_sci, out)
    out = _ALGEBRA_VAR_MINUS.sub(r"\1减\2", out)
    out = _DIGIT_X_DIGIT.sub("乘", out)
    out = _LETTER_DIGIT.sub(r"\1 \2", out)
    out = _DIGIT_LETTER.sub(r"\1 \2", out)
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

    for pattern, repl in _CMP_BEFORE_NUM:
        out = pattern.sub(repl, out)

    def _restore_url(match: re.Match) -> str:
        index = _alpha_id_value(match.group(1))
        if 0 <= index < len(held):
            return _verbalize_url(held[index])
        return match.group(0)

    out = _URL_HOLD_RE.sub(_restore_url, out)
    return out
