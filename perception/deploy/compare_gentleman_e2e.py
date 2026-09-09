#!/usr/bin/env python3
"""Compare dual-session vs merged Gentleman e2e on one fixed utterance.

Does not load both graphs in one process. Run twice:

  python3 compare_gentleman_e2e.py --mode dual --dump /tmp/gentleman_dual.npz
  TTS_GENTLEMAN_E2E_ONNX=... python3 compare_gentleman_e2e.py --mode e2e \\
      --dump /tmp/gentleman_e2e.npz --ref /tmp/gentleman_dual.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import wave

import numpy as np

sys.path[:0] = ["/work"]

TEXT = os.environ.get(
    "HEAP_TEXT", "各位评委大家好，我是 Gentleman 语音合成系统。"
)
MODEL_DIR = os.environ.get(
    "GENTLEMAN_MODEL_DIR", "/models/matcha-gentleman-phonetone-16k"
)


def _pcm16_to_f32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _write_wav(path: str, pcm: bytes, sr: int = 16000) -> None:
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dual", "e2e"), required=True)
    parser.add_argument("--dump")
    parser.add_argument("--ref")
    parser.add_argument("--wav")
    args = parser.parse_args()

    from plugins.tts import MatchaPhoneToneOrtAdapter
    from utils.phonetone import encode_for_matcha, configure_release
    from utils.matcha_trt import crop_mel
    from plugins.tts import _ort_outputs

    configure_release(MODEL_DIR)
    packed = encode_for_matcha(TEXT)
    print(
        "ENCODE text=%r x=%s real_len=%s"
        % (TEXT, tuple(packed["x"].shape), packed["real_len"]),
        flush=True,
    )

    t0 = time.perf_counter()
    adapter = MatchaPhoneToneOrtAdapter(model_dir=MODEL_DIR, hw_provider="cuda")
    print(
        "LOAD_S %.2f e2e=%s mode=%s"
        % (time.perf_counter() - t0, getattr(adapter, "_e2e", None), args.mode),
        flush=True,
    )
    if args.mode == "e2e" and not getattr(adapter, "_e2e", False):
        raise SystemExit("e2e mode requested but adapter fell back to dual session")
    if args.mode == "dual" and getattr(adapter, "_e2e", False):
        raise SystemExit("dual mode requested but adapter selected e2e session")

    t1 = time.perf_counter()
    pcm = adapter._synthesize_segment(TEXT)
    rtf_s = time.perf_counter() - t1
    samples = _pcm16_to_f32(pcm)
    finite = bool(np.isfinite(samples).all())
    print(
        "WAV n=%s seconds=%.3f synth_s=%.3f rtf=%.3f nan_inf=%s min=%.4f max=%.4f"
        % (
            samples.size,
            samples.size / 16000.0,
            rtf_s,
            rtf_s / max(samples.size / 16000.0, 1e-6),
            not finite,
            float(samples.min()) if samples.size else 0.0,
            float(samples.max()) if samples.size else 0.0,
        ),
        flush=True,
    )

    mel_len = None
    if args.mode == "dual":
        feeds = {
            name: packed[name]
            for name in ("x", "x_lengths", "tones", "languages", "scales")
        }
        ac_out = _ort_outputs(adapter._sess, feeds)
        mel = ac_out.get("mel")
        cropped = crop_mel(mel, packed["real_len"], ac_out.get("mel_lengths"))
        mel_len = int(cropped.shape[-1])
        print(
            "MEL shape=%s cropped_t=%s mel_lengths=%s"
            % (
                None if mel is None else tuple(np.asarray(mel).shape),
                mel_len,
                None
                if ac_out.get("mel_lengths") is None
                else np.asarray(ac_out["mel_lengths"]).tolist(),
            ),
            flush=True,
        )

    if args.dump:
        payload = dict(x=packed["x"], wav=samples, n=np.int64(samples.size))
        if mel_len is not None:
            payload["mel_len"] = np.int64(mel_len)
        np.savez(args.dump, **payload)
        print("DUMP %s" % args.dump, flush=True)
    if args.wav:
        _write_wav(args.wav, pcm)
        print("WAV_FILE %s" % args.wav, flush=True)

    if args.ref:
        ref = np.load(args.ref)
        ref_wav = ref["wav"]
        n = min(samples.size, ref_wav.size)
        diff = np.abs(samples[:n] - ref_wav[:n])
        print(
            "VS_DUAL dual_n=%s e2e_n=%s x_equal=%s max_abs=%.6f mean_abs=%.6f"
            % (
                int(ref["n"]),
                samples.size,
                np.array_equal(packed["x"], ref["x"]),
                float(diff.max()) if n else 0.0,
                float(diff.mean()) if n else 0.0,
            ),
            flush=True,
        )
        if samples.size != int(ref["n"]):
            print("WAV_LEN_MISMATCH", flush=True)
            return 1
        if not finite:
            print("E2E_NONFINITE", flush=True)
            return 1
    print("COMPARE_OK", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
