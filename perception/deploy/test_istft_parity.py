#!/usr/bin/env python3
"""Parity: numpy iSTFT vs scipy.signal.istft (Vocos 16 kHz settings).

Run on a machine with scipy (orin-03 container has it). Ranking does not import scipy.

  python3 perception/deploy/test_istft_parity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.vocos_trt import N_FFT, HOP, WIN, SAMPLE_RATE, istft_numpy, istft_scipy  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(0)
    n_bins = N_FFT // 2 + 1
    print("n_fft=%s hop=%s win=%s sr=%s" % (N_FFT, HOP, WIN, SAMPLE_RATE))
    worst = 0.0
    for n_frames in (16, 48, 80, 200):
        mag = rng.random((n_bins, n_frames), dtype=np.float64) * 0.1
        x = rng.standard_normal((n_bins, n_frames))
        y = rng.standard_normal((n_bins, n_frames))
        a = istft_scipy(mag, x, y)
        b = istft_numpy(mag, x, y)
        n = min(a.size, b.size)
        mae = float(np.mean(np.abs(a[:n] - b[:n])))
        rmse = float(np.sqrt(np.mean((a[:n] - b[:n]) ** 2)))
        mx = float(np.max(np.abs(a[:n] - b[:n])))
        worst = max(worst, mx)
        print(
            "frames=%s scipy_len=%s numpy_len=%s mae=%.4g rmse=%.4g max=%.4g"
            % (n_frames, a.size, b.size, mae, rmse, mx)
        )
        if a.size != b.size:
            print("FAIL length mismatch", file=sys.stderr)
            return 1
    if worst > 1e-5:
        print("FAIL max err %s" % worst, file=sys.stderr)
        return 1
    print("PASS max_err=%.4g" % worst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
