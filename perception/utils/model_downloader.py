"""
utils/model_downloader.py — Auto-download sherpa-onnx models from COS if missing.
"""

from __future__ import annotations

import logging
import os
import tarfile
import tempfile
import zipfile
from urllib.request import urlretrieve

log = logging.getLogger(__name__)

COS_BASE = "https://agi-phanthy-dev-1252788780.cos.ap-beijing.myqcloud.com/public"
JUICEFS_BASE = "http://172.28.4.81:34567/fanyi/phanthymotus_tts"


def _progress_hook(name: str):
    """Create a reporthook for urlretrieve that logs download progress."""
    last_pct = [0]
    def hook(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(int(block_num * block_size * 100 / total_size), 100)
            if pct >= last_pct[0] + 10:
                last_pct[0] = pct
                mb_done = block_num * block_size / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                log.info(f"[model_downloader] {name}: {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)")
    return hook

MODELS = {
    "asr": {
        "url": f"{COS_BASE}/sherpa-onnx-streaming-paraformer-bilingual-zh-en.zip",
        "check_file": "tokens.txt",
    },
    "asr_en": {
        "url": f"{COS_BASE}/sherpa-onnx-streaming-zipformer-en-2023-06-26.zip",
        "check_file": "tokens.txt",
    },
    "asr_sensevoice": {
        "url": f"{COS_BASE}/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.zip",
        "check_file": "tokens.txt",
    },
    "tts": {
        "url": f"{JUICEFS_BASE}/matcha-kai-16k-e500.tar.bz2",
        "check_file": "model-steps-3.onnx",
    },
    "tts_matcha_kai": {
        "url": f"{JUICEFS_BASE}/matcha-kai-16k-e500.tar.bz2",
        "check_file": "model-steps-3.onnx",
    },
    "tts_matcha_lufeng": {
        "url": f"{JUICEFS_BASE}/matcha-lufeng-16k-e500.tar.bz2",
        "check_file": "model-steps-3.onnx",
    },
    "tts_matcha_gentleman": {
        "url": f"{JUICEFS_BASE}/matcha-gentleman-phonetone-16k.tar.bz2",
        "check_file": "model-steps-3.onnx",
    },
    # WeText TN graphs (tagger+verbalizer). JuiceFS only — never git.
    "tts_wetext": {
        "url": f"{JUICEFS_BASE}/wetext.tar.bz2",
        "check_file": "zh_tn_tagger.fst",
    },
    "tts_vocoder": {
        "url": f"{JUICEFS_BASE}/vocos-16khz-univ.onnx",
        "check_file": "vocos-16khz-univ.onnx",
        "single_file": True,
    },
    # 8k Melo: longanlingxin pack (model.onnx + lexicon + dict)
    "tts_melo_8k": {
        "url": f"{JUICEFS_BASE}/vits-melo-longanlingxin-8k.tar.bz2",
        "check_file": "model.onnx",
    },
    # Shared OpenEPD + Melo G2P large assets (pickle / tokens / vendor skeleton).
    # Slim english.py lives in the git image (/work/melo_g2p_slim), NOT in this tar.
    "tts_melo_openepd_g2p": {
        "url": f"{JUICEFS_BASE}/melo-openepd-g2p-assets.tar.bz2",
        "check_file": "openepd_eng_dict.pickle",
    },
    # OOV neural weights (numpy). JuiceFS only — never git.
    "tts_melo_g2p_oov_ckpt": {
        "url": f"{JUICEFS_BASE}/checkpoint20.npz",
        "check_file": "checkpoint20.npz",
        "single_file": True,
    },
    # Compact OpenEPD lexicon (mmap .oedb). JuiceFS only — never git.
    "tts_melo_openepd_compact": {
        "url": f"{JUICEFS_BASE}/openepd_eng_dict.oedb",
        "check_file": "openepd_eng_dict.oedb",
        "single_file": True,
    },
    # Voice ONNX-only packs (model.onnx + tiny model_meta.json)
    # Prefer FP32 on Jetson CUDA EP: QUInt8 dynamic quant causes ~660 Memcpy
    # fallbacks and ~27x slower ORT than FP32 (see Jetson int8 vs fp32 bench).
    "tts_melo_openepd_fp32": {
        "url": f"{JUICEFS_BASE}/vits-melo-longanlingxin-openepd-nobert-44100-fp32.tar.bz2",
        "check_file": "model.onnx",
    },
    "tts_melo_openepd_fp32_lufeng": {
        "url": f"{JUICEFS_BASE}/vits-melo-lufeng-openepd-nobert-44100-fp32.tar.bz2",
        "check_file": "model.onnx",
    },
    "tts_melo_openepd_fp32_kai": {
        "url": f"{JUICEFS_BASE}/vits-melo-kai-openepd-nobert-44100-fp32.tar.bz2",
        "check_file": "model.onnx",
    },
    "tts_melo_openepd_fp16": {
        "url": f"{JUICEFS_BASE}/vits-melo-longanlingxin-openepd-nobert-44100-fp16.tar.bz2",
        "check_file": "model.onnx",
    },
    "tts_melo_openepd_int8": {
        "url": f"{JUICEFS_BASE}/vits-melo-longanlingxin-openepd-nobert-44100-int8.tar.bz2",
        "check_file": "model.onnx",
    },
    "tts_melo_openepd_int8_lufeng": {
        "url": f"{JUICEFS_BASE}/vits-melo-lufeng-openepd-nobert-44100-int8.tar.bz2",
        "check_file": "model.onnx",
    },
    "tts_melo_openepd_int8_kai": {
        "url": f"{JUICEFS_BASE}/vits-melo-kai-openepd-nobert-44100-int8.tar.bz2",
        "check_file": "model.onnx",
    },
    # Piper dual-G2P B2 (model.onnx + model.onnx.json + frontend + lexicon + vendor/g2p)
    "tts_piper_b2": {
        "url": f"{JUICEFS_BASE}/piper-longanlingxin-b2.tar.bz2",
        "check_file": "model.onnx",
    },
    "tts_melo": {
        "url": f"{JUICEFS_BASE}/vits-melo-tts-zh_en.tar.bz2",
        "check_file": "model.onnx",
    },
    "tts_melo_int8": {
        "url": f"{JUICEFS_BASE}/vits-melo-tts-zh_en_int8.tar.bz2",
        "check_file": "model.onnx",
    },
    "tts_zh_finetuned": {
        "url": f"{JUICEFS_BASE}/zh_finetuned.tar.bz2",
        "check_file": "model.onnx",
    },
    "tts_kokoro_int8": {
        "url": f"{JUICEFS_BASE}/kokoro-int8-multi-lang-v1_1.tar.bz2",
        "check_file": "voices.bin",
    },
    "kws": {
        "url": f"{COS_BASE}/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2",
        "check_file": "tokens.txt",
    },
    "vad": {
        "url": f"{COS_BASE}/silero_vad.onnx",
        "check_file": "silero_vad.onnx",
        "single_file": True,
    },
}


def ensure_model(name: str, model_dir: str) -> None:
    """Ensure model files exist in model_dir. Download if missing."""
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model name: {name}")

    check_path = os.path.join(model_dir, info["check_file"])
    if os.path.exists(check_path):
        log.info(f"[model_downloader] {name}: already exists at {model_dir}")
        return

    url = info["url"]
    os.makedirs(model_dir, exist_ok=True)
    log.info(f"[model_downloader] {name}: downloading from {url} ...")

    if info.get("single_file"):
        dest = os.path.join(model_dir, info["check_file"])
        urlretrieve(url, dest, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: done.")
        return

    suffix = ".zip" if url.endswith(".zip") else ".tar.bz2"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urlretrieve(url, tmp_path, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: extracting to {model_dir} ...")
        if suffix == ".zip":
            _extract_zip(tmp_path, model_dir)
        else:
            _extract_tar(tmp_path, model_dir)
        log.info(f"[model_downloader] {name}: done.")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not os.path.exists(check_path):
        raise RuntimeError(
            f"[model_downloader] {name}: download completed but {info['check_file']} "
            f"not found in {model_dir}"
        )


def _extract_zip(zip_path: str, model_dir: str) -> None:
    """Extract zip, stripping common top-level directory prefix."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = [n for n in zf.namelist()
                 if not n.endswith('/') and not n.startswith('__MACOSX')]
        if not names:
            raise RuntimeError(f"Empty archive: {zip_path}")

        prefix = _common_prefix_from_names(names)
        for name in names:
            stripped = name[len(prefix):] if prefix else name
            if not stripped:
                continue
            dest = os.path.join(model_dir, stripped)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(name) as src, open(dest, 'wb') as dst:
                dst.write(src.read())


def _extract_tar(tar_path: str, model_dir: str) -> None:
    """Extract tar.bz2, stripping common top-level directory prefix."""
    with tarfile.open(tar_path, "r:bz2") as tf:
        members = tf.getmembers()
        if not members:
            raise RuntimeError(f"Empty archive: {tar_path}")

        names = [m.name for m in members if not m.isdir()]
        prefix = _common_prefix_from_names(names)
        for m in members:
            if m.isdir():
                continue
            if prefix:
                m.name = m.name[len(prefix):]
            if not m.name:
                continue
            m.name = m.name.lstrip("/")
            tf.extract(m, model_dir)


def _common_prefix_from_names(names: list[str]) -> str:
    """Find common top-level directory prefix from file name list."""
    dirs_with_slash = [n.split("/", 1) for n in names if "/" in n]
    if not dirs_with_slash:
        return ""
    first_parts = set(parts[0] for parts in dirs_with_slash)
    if len(first_parts) == 1:
        return first_parts.pop() + "/"
    return ""
