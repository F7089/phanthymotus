"""Matcha ORT helpers. No TensorRT."""
from __future__ import annotations

import os

import numpy as np

MAX_MEL = int(os.environ.get("TTS_TRT_MAX_MEL", "2000"))


def crop_mel(mel, n_tokens: int, mel_lengths=None):
    """Crop padded decoder frames. Prefer model mel_lengths over token*24."""
    m = np.asarray(mel, dtype=np.float32)
    if m.ndim == 3:
        m = m[0]
    caps = [int(m.shape[1])]
    if mel_lengths is not None:
        ml = int(np.asarray(mel_lengths).reshape(-1)[0])
        if ml > 0:
            caps.append(ml)
    if n_tokens:
        caps.append(max(1, min(MAX_MEL, int(n_tokens) * 24)))
    end = max(1, min(caps))
    return m[:, :end]
