"""TTS text frontend: acronym expand + lead text_process + WeText."""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Whole English words / brands that must NOT be letter-split.
_KEEP_AS_WORD = frozenset(
    w.lower()
    for w in (
        "bluetooth",
        "online",
        "offline",
        "loading",
        "connected",
        "disconnected",
        "successful",
        "failed",
        "warning",
        "error",
        "update",
        "download",
        "upload",
        "settings",
        "cancel",
        "confirm",
        "restart",
        "timeout",
        "retry",
        "ready",
        "busy",
        "idle",
        "running",
        "stopped",
        "paused",
        "resume",
        "cache",
        "memory",
        "network",
        "wireless",
        "password",
        "username",
        "account",
        "message",
        "notification",
        "permission",
        "authentication",
        "authorization",
        "firewall",
        "proxy",
        "latency",
        "throughput",
        "bandwidth",
        "speaker",
        "microphone",
        "camera",
        "sensor",
        "battery",
        "temperature",
        "jetson",
        "orin",
        "docker",
        "python",
        "ubuntu",
        "linux",
        "windows",
        "android",
        "apple",
        "google",
        "microsoft",
        "amazon",
        "nvidia",
        "intel",
        "openai",
        "tesla",
        "github",
        "huggingface",
        "melotts",
        "fastdds",
        "tensorrt",
        "kubernetes",
        "wifi",
    )
)

# Always letter-split these (even if they look like words).
_FORCE_SPELL = frozenset(
    w.upper()
    for w in (
        "AI",
        "CPU",
        "GPU",
        "NPU",
        "CUDA",
        "USB",
        "HDMI",
        "HTTP",
        "HTTPS",
        "API",
        "SDK",
        "ONNX",
        "TTS",
        "ASR",
        "OCR",
        "TCP",
        "UDP",
        "SSH",
        "DNS",
        "VPN",
        "SSD",
        "RAM",
        "ROM",
        "LED",
        "LCD",
        "OLED",
        "GPS",
        "NFC",
        "BLE",
        "MQTT",
        "ROS",
        "DDS",
        "MCP",
        "JSON",
        "YAML",
        "XML",
        "HTML",
        "CSS",
        "SQL",
        "LLM",
        "VITS",
        "RTF",
        "TTFT",
        "PCM",
        "WAV",
        "MP3",
        "AAC",
        "OPUS",
        "PDF",
        "URL",
        "UUID",
        "IP",
        "ID",
        "OS",
        "UI",
        "UX",
    )
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|Wi-?Fi|wi-?fi")


def expand_en_acronyms(text: str) -> str:
    """Spell out ALLCAPS abbreviations; keep normal English words intact."""
    if not text:
        return text

    def _repl(m: re.Match) -> str:
        tok = m.group(0)
        raw = tok
        # Normalize Wi-Fi variants
        if tok.lower().replace("-", "") == "wifi":
            return "Wi Fi"
        upper = tok.upper()
        lower = tok.lower()
        if upper in _FORCE_SPELL or (
            tok.isupper()
            and 2 <= len(tok) <= 6
            and tok.isalpha()
            and lower not in _KEEP_AS_WORD
        ):
            return " ".join(list(upper))
        if lower in _KEEP_AS_WORD:
            return raw
        return raw

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

        # Lazy singleton via function attribute
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
            original[:48],
            text[:48],
        )
    return text
