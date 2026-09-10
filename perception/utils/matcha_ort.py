"""Matcha / Vocos ORT helpers. No TensorRT."""
from __future__ import annotations

import os

import numpy as np

MAX_MEL = int(os.environ.get("TTS_TRT_MAX_MEL", "2000"))
VOCOS_N_FFT = 1024
VOCOS_HOP = 256
VOCOS_WIN = 1024


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


def _hann_periodic(n: int) -> np.ndarray:
    # Match torch.hann_window(n, periodic=True).
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / n)


def vocos_istft(mag, x, y) -> np.ndarray:
    """CPU iSTFT for Gentleman Vocos ONNX mag/x/y.

    Must match vocos.spectral_ops.ISTFT padding='same' (n_fft=1024, hop=256):
    irfft → * window → overlap-add → trim pad → / window_envelope.
    Do NOT apply scipy/sherpa spectrum gain (win.sum()); that over-amplifies
    torch-exported mag/x/y by ~512x and clips into metallic distortion.
    """
    mag = np.asarray(mag, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    while mag.ndim > 2:
        mag, x, y = mag[0], x[0], y[0]
    spec = np.asarray(mag * (x + 1j * y), dtype=np.complex128)
    n_bins, nseg = spec.shape
    if n_bins != VOCOS_N_FFT // 2 + 1:
        raise RuntimeError("vocos istft expected %s bins, got %s" % (VOCOS_N_FFT // 2 + 1, n_bins))
    win = _hann_periodic(VOCOS_WIN)
    pad = (VOCOS_WIN - VOCOS_HOP) // 2  # 384 for 1024/256 "same"
    frames = np.fft.irfft(spec, n=VOCOS_N_FFT, axis=0).real[:VOCOS_WIN, :]
    frames = frames * win[:, None]
    out_len = VOCOS_WIN + (nseg - 1) * VOCOS_HOP
    acc = np.zeros(out_len, dtype=np.float64)
    w2 = np.zeros(out_len, dtype=np.float64)
    for t in range(nseg):
        off = t * VOCOS_HOP
        acc[off : off + VOCOS_WIN] += frames[:, t]
        w2[off : off + VOCOS_WIN] += win * win
    acc = acc[pad : out_len - pad]
    w2 = w2[pad : out_len - pad]
    acc /= np.where(w2 > 1e-11, w2, 1.0)
    return acc.astype(np.float32)
