"""TTS text frontend: compound split + acronym expand + lead text_process + WeText.

Pipeline:
  - non-word compounds → split (deepseek → deep seek)
  - real English words kept intact
  - ALLCAPS Latin tokens (len>=2) → letter-split
  - 藏族/藏文/藏塔 → 臧… (heteronym without FST)
  - wetext + optional lead text_process
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

# Longer keys first via sort at apply time.
_SPLIT_COMPOUNDS: list[tuple[str, str]] = [
    ("Apple Pay", "Apple Pay"),
    ("apple pay", "Apple Pay"),
    ("photoshop", "photo shop"),
    ("Photoshop", "photo shop"),
    ("PHOTOSHOP", "photo shop"),
    ("deepseek", "deep seek"),
    ("DeepSeek", "deep seek"),
    ("Deepseek", "deep seek"),
    ("DEEPSEEK", "deep seek"),
    ("roadmap", "road map"),
    ("Roadmap", "road map"),
    ("ROADMAP", "road map"),
    ("iPhone", "i phone"),
    ("iphone", "i phone"),
    ("IPHONE", "i phone"),
    ("iPad", "i pad"),
    ("ipad", "i pad"),
    ("IPAD", "i pad"),
    ("iCloud", "i cloud"),
    ("icloud", "i cloud"),
    ("newapp", "new app"),
    ("藏族", "臧族"),
    ("藏文", "臧文"),
    ("藏塔", "臧塔"),
]

_KEEP_AS_WORD = frozenset(
    w.lower()
    for w in (
        "offer",
        "python",
        "java",
        "nike",
        "twitter",
        "facebook",
        "apple",
        "pay",
        "product",
        "review",
        "thanks",
        "smooth",
        "team",
        "leader",
        "deadline",
        "project",
        "mac",
        "lucy",
        "hi",
        "ppt",
        "photo",
        "shop",
        "deep",
        "seek",
        "road",
        "map",
        "phone",
        "pad",
        "cloud",
        "new",
        "app",
        "common",
        "questions",
        "technical",
        "issues",
        "photoshop",
        "deepseek",
        "roadmap",
        "iphone",
        "ipad",
        "bluetooth",
        "online",
        "offline",
        "wifi",
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
        "ubuntu",
        "linux",
        "windows",
        "android",
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
    )
)

_FORCE_SPELL = frozenset(
    w.upper()
    for w in (
        "AI",
        "IT",
        "FAQ",
        "CEO",
        "CTO",
        "CFO",
        "IPO",
        "OA",
        "POS",
        "ATM",
        "KTV",
        "IELTS",
        "GPS",
        "PPT",
        "UX",
        "UI",
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
        "KPI",
        "PR",
        "HR",
        "MVP",
        "NDA",
    )
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*|Wi-?Fi|wi-?fi")


def apply_compound_split(text: str) -> str:
    if not text:
        return text
    for src, dst in sorted(_SPLIT_COMPOUNDS, key=lambda x: len(x[0]), reverse=True):
        if src in text:
            text = text.replace(src, dst)
    return text


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
    text = apply_compound_split(text)
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
