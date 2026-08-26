"""Make sherpa skip Vocoder::Create without rebuilding sherpa.

sherpa 1.13.6 sets need_vocoder=false when the acoustic ONNX first output is
named audio_output. generate() then returns the flat mel tensor as samples.
We rename the output once and cache the aliased file under TTS_VOCOS_TRT_CACHE.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def alias_acoustic_audio_output(src: str, cache_dir: str) -> str:
    src_p = Path(src)
    if not src_p.is_file():
        raise FileNotFoundError(src)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / (src_p.stem + ".audio_output.onnx")
    src_stat = src_p.stat()
    if dst.is_file() and dst.stat().st_mtime >= src_stat.st_mtime and dst.stat().st_size > 0:
        log.info("[tts] acoustic alias reuse %s", dst)
        return str(dst)

    import onnx

    model = onnx.load(str(src_p))
    if not model.graph.output:
        raise RuntimeError("acoustic ONNX has no outputs")
    old = model.graph.output[0].name
    if old == "audio_output":
        log.info("[tts] acoustic already named audio_output: %s", src)
        return str(src_p)
    model.graph.output[0].name = "audio_output"
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = "audio_output"
    tmp = dst.with_suffix(".tmp")
    onnx.save(model, str(tmp))
    os.replace(str(tmp), str(dst))
    log.info("[tts] acoustic alias %s -> audio_output (%s)", old, dst)
    return str(dst)
