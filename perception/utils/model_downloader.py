"""
utils/model_downloader.py — Fetch Gentleman Matcha pack from JuiceFS if missing.
"""

from __future__ import annotations

import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from urllib.request import urlretrieve

log = logging.getLogger(__name__)

JUICEFS_BASE = "http://172.28.4.81:34567/fanyi/phanthymotus_tts"
# Same JuiceFS data disk the tar lives on. Prefer this over HTTP when present.
JUICEFS_LOCAL = os.environ.get("TTS_JUICEFS_DIR", "/mnt/data/fanyi/phanthymotus_tts")


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
    "tts_matcha_gentleman": {
        "url": f"{JUICEFS_BASE}/matcha-gentleman-phonetone-16k.tar.bz2",
        "check_file": "model-steps-3.onnx",
        # Ranking uses 3-step. Do not extract the 10-step graph into page cache.
        "skip_files": ("model-steps-10.onnx",),
    },
}


def drop_file_pages(path: str, *, log_result: bool = True) -> int:
    """Evict file pages from the page cache (POSIX_FADV_DONTNEED).

    Ranking cgroup max_usage includes cache. Freshly written files are dirty;
    fsync first or DONTNEED is a no-op on Linux. After ORT copies weights,
    on-disk onnx/tar/wheel do not need to stay resident.
    """
    if not path or not os.path.exists(path):
        return 0
    files: list[str] = []
    if os.path.isfile(path):
        files = [path]
    else:
        for root, _, names in os.walk(path):
            for name in names:
                files.append(os.path.join(root, name))
    n = 0
    for file_path in files:
        fd = -1
        try:
            try:
                fd = os.open(file_path, os.O_RDWR)
            except OSError:
                fd = os.open(file_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            except OSError:
                pass
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            n += 1
        except OSError:
            continue
        finally:
            if fd >= 0:
                os.close(fd)
    if n and log_result:
        log.info("[model_downloader] dropped page cache for %s files under %s", n, path)
    return n


def _unlink_skipped(model_dir: str, skip_files) -> None:
    for name in skip_files or ():
        victim = os.path.join(model_dir, name)
        if os.path.isfile(victim):
            try:
                drop_file_pages(victim)
                os.unlink(victim)
                log.info("[model_downloader] removed unused %s", victim)
            except OSError as e:
                log.warning("[model_downloader] could not remove %s: %s", victim, e)


def _local_juicefs_src(url: str) -> str | None:
    name = os.path.basename(url.split("?", 1)[0])
    path = os.path.join(JUICEFS_LOCAL, name)
    return path if os.path.isfile(path) else None


def ensure_model(name: str, model_dir: str) -> None:
    """Ensure model files exist in model_dir. Load from data disk, else HTTP."""
    info = MODELS.get(name)
    if not info:
        raise ValueError(f"Unknown model name: {name}")

    check_path = os.path.join(model_dir, info["check_file"])
    skip_files = info.get("skip_files") or ()
    if os.path.exists(check_path):
        log.info(f"[model_downloader] {name}: already exists at {model_dir}")
        _unlink_skipped(model_dir, skip_files)
        drop_file_pages(model_dir)
        return

    url = info["url"]
    os.makedirs(model_dir, exist_ok=True)
    local = _local_juicefs_src(url)

    if info.get("single_file"):
        dest = os.path.join(model_dir, info["check_file"])
        if local:
            log.info(f"[model_downloader] {name}: copy from data disk {local}")
            shutil.copy2(local, dest)
        else:
            log.info(f"[model_downloader] {name}: downloading from {url} ...")
            urlretrieve(url, dest, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: done.")
        return

    suffix = ".zip" if url.endswith(".zip") else ".tar.bz2"
    if local:
        log.info(f"[model_downloader] {name}: extract from data disk {local}")
        if suffix == ".zip":
            _extract_zip(local, model_dir, skip_files)
        else:
            _extract_tar(local, model_dir, skip_files)
        _unlink_skipped(model_dir, skip_files)
        drop_file_pages(model_dir)
        log.info(f"[model_downloader] {name}: done.")
        if not os.path.exists(check_path):
            raise RuntimeError(
                f"[model_downloader] {name}: local extract completed but "
                f"{info['check_file']} not found in {model_dir}"
            )
        return

    log.info(f"[model_downloader] {name}: downloading from {url} ...")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urlretrieve(url, tmp_path, reporthook=_progress_hook(name))
        log.info(f"[model_downloader] {name}: extracting to {model_dir} ...")
        if suffix == ".zip":
            _extract_zip(tmp_path, model_dir, skip_files)
        else:
            _extract_tar(tmp_path, model_dir, skip_files)
        drop_file_pages(tmp_path)
        log.info(f"[model_downloader] {name}: done.")
    finally:
        if os.path.exists(tmp_path):
            drop_file_pages(tmp_path)
            os.unlink(tmp_path)
    _unlink_skipped(model_dir, skip_files)
    drop_file_pages(model_dir)

    if not os.path.exists(check_path):
        raise RuntimeError(
            f"[model_downloader] {name}: download completed but {info['check_file']} "
            f"not found in {model_dir}"
        )


def _extract_zip(zip_path: str, model_dir: str, skip_files=()) -> None:
    """Extract zip, stripping common top-level directory prefix."""
    skip = set(skip_files or ())
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
            if os.path.basename(stripped) in skip:
                continue
            dest = os.path.join(model_dir, stripped)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(name) as src, open(dest, 'wb') as dst:
                dst.write(src.read())
                dst.flush()
                os.fsync(dst.fileno())
            drop_file_pages(dest, log_result=False)


def _extract_tar(tar_path: str, model_dir: str, skip_files=()) -> None:
    """Extract tar.bz2, stripping common top-level directory prefix."""
    skip = set(skip_files or ())
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
            if os.path.basename(m.name) in skip:
                continue
            tf.extract(m, model_dir)
            drop_file_pages(os.path.join(model_dir, m.name), log_result=False)


def _common_prefix_from_names(names: list[str]) -> str:
    """Find common top-level directory prefix from file name list."""
    dirs_with_slash = [n.split("/", 1) for n in names if "/" in n]
    if not dirs_with_slash:
        return ""
    first_parts = set(parts[0] for parts in dirs_with_slash)
    if len(first_parts) == 1:
        return first_parts.pop() + "/"
    return ""
