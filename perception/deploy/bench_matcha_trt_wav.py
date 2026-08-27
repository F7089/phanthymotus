#!/usr/bin/env python3
"""Matcha acoustic TRT + Vocos TRT -> 16k wav. No sherpa / no ORT CUDA.

Reuses engines in /opt/matcha_trt_cache and /opt/vocos_trt_cache.
Lexicon+tokens only (Chinese chars). Writes wav and cgroup stage dumps.

  python3 /deploy/bench_matcha_trt_wav.py \\
    --acoustic /opt/matcha_trt_cache/foo.engine \\
    --vocos /opt/vocos_trt_cache/vocos-....engine \\
    --model-dir /models/matcha-kai-16k-e500 \\
    --out /opt/matcha_trt_cache/hello.wav
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, "/deploy")
sys.path.insert(0, "/work")
from bench_matcha_trt_mem import (  # noqa: E402
    _Cudart,
    _numpy_dtype,
    deserialize,
    dump,
    find_cgroup,
)
from utils.vocos_trt import _istft  # noqa: E402

MAX_MEL = int(os.environ.get("TTS_TRT_MAX_MEL", "2000"))


def _load_tokens(path):
    tok2id = {}
    for line in open(path, encoding="utf-8"):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) == 1:
            tok2id[" "] = int(parts[0])
            continue
        a, b = parts[0], parts[-1]
        if b.lstrip("-").isdigit() and not a.lstrip("-").isdigit():
            tok2id[a] = int(b)
        elif a.lstrip("-").isdigit():
            tok2id[b] = int(a)
        else:
            tok2id[a] = int(b)
    return tok2id


def _load_lexicon(path):
    lex = {}
    for line in open(path, encoding="utf-8"):
        parts = line.strip().split()
        if len(parts) >= 2:
            lex[parts[0]] = parts[1:]
    return lex


def text_to_ids(text, lex, tok2id):
    phones = []
    skipped = []
    for ch in text:
        if ch.isspace() or ch in "，。！？、；：,.!?;:\"'""''":
            continue
        if ch in lex:
            phones.extend(lex[ch])
        elif ch in tok2id:
            phones.append(ch)
        else:
            skipped.append(ch)
    ids = []
    unk = tok2id.get("<unk>", tok2id.get("UNK"))
    missing = []
    for p in phones:
        if p in tok2id:
            ids.append(tok2id[p])
        elif unk is not None:
            ids.append(unk)
            missing.append(p)
        else:
            missing.append(p)
    for name in ("<sos>", "<bos>", "sos", "<s>"):
        if name in tok2id:
            ids = [tok2id[name]] + ids
            break
    for name in ("<eos>", "</s>", "eos"):
        if name in tok2id:
            ids = ids + [tok2id[name]]
            break
    return ids, phones, skipped, missing


def infer(engine, ctx, cuda, feeds, keep):
    nbind = int(engine.num_bindings)
    for i in range(nbind):
        if not engine.binding_is_input(i):
            continue
        name = engine.get_binding_name(i)
        if name not in feeds:
            raise RuntimeError("missing input %s" % name)
        arr = np.ascontiguousarray(feeds[name])
        feeds[name] = arr
        ctx.set_binding_shape(i, tuple(arr.shape))
    ptrs = []
    host_out = []
    for i in range(nbind):
        name = engine.get_binding_name(i)
        shape = tuple(int(d) for d in ctx.get_binding_shape(i))
        if any(d < 1 for d in shape):
            if (not engine.binding_is_input(i)) and len(shape) == 3:
                shape = (1, 80, MAX_MEL)
            elif name == "mels" and len(shape) == 3:
                shape = tuple(feeds["mels"].shape)
                ctx.set_binding_shape(i, shape)
            else:
                shape = tuple(max(d, 1) for d in shape)
        dtype = _numpy_dtype(engine.get_binding_dtype(i))
        nbytes = int(np.prod(shape)) * dtype.itemsize
        dptr = cuda.malloc(nbytes)
        keep.append(dptr)
        ptrs.append(dptr)
        if engine.binding_is_input(i):
            cuda.h2d(dptr, np.ascontiguousarray(feeds[name], dtype=dtype))
        else:
            host_out.append((name, np.empty(shape, dtype=dtype), dptr))
    if not ctx.execute_v2(ptrs):
        raise RuntimeError("execute_v2 failed")
    outs = {}
    for name, buf, dptr in host_out:
        cuda.d2h(dptr, buf)
        outs[name] = buf
    return outs


def crop_mel(mel):
    m = np.asarray(mel, dtype=np.float32)
    if m.ndim == 3:
        m = m[0]
    e = (m.astype(np.float64) ** 2).mean(axis=0)
    mx = float(e.max()) if e.size else 0.0
    if mx <= 0:
        return m
    hits = np.where(e > mx * 0.02)[0]
    if hits.size == 0:
        return m
    end = min(m.shape[1], int(hits[-1]) + 8)
    start = max(0, int(hits[0]))
    return m[:, start:end]


def write_wav(path, pcm, sr=16000):
    pcm = np.clip(np.asarray(pcm, dtype=np.float32).reshape(-1), -1.0, 1.0)
    i16 = (pcm * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(i16.tobytes())
    print(
        "wrote_wav",
        path,
        "samples",
        i16.size,
        "sec",
        round(i16.size / float(sr), 3),
        "bytes",
        os.path.getsize(path),
        flush=True,
    )


def cg():
    u, p = find_cgroup()
    return u, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acoustic", required=True)
    ap.add_argument("--vocos", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="你好，我是陆风。")
    args = ap.parse_args()
    keep = []
    dump("A_base")
    import tensorrt as trt  # noqa: F401

    print("imported tensorrt", trt.__version__, flush=True)
    dump("B_import_trt")
    ac_eng, ac_ctx = deserialize(args.acoustic, keep)
    dump("C_acoustic")
    vx_eng, vx_ctx = deserialize(args.vocos, keep)
    dump("D_vocos")
    cuda = _Cudart()
    keep.append(cuda)
    dump("E_cuda_ctx")

    tok2id = _load_tokens(os.path.join(args.model_dir, "tokens.txt"))
    lex = _load_lexicon(os.path.join(args.model_dir, "lexicon.txt"))
    ids, phones, skipped, missing = text_to_ids(args.text, lex, tok2id)
    print("text", args.text, flush=True)
    print("n_tokens", len(ids), "ids", ids[:40], flush=True)
    print("phones", " ".join(phones[:40]), flush=True)
    print("skipped", skipped, "missing_phone", missing[:20], flush=True)
    if len(ids) < 2:
        raise SystemExit("too few tokens")
    dump("F_frontend")

    feeds = {
        "x": np.asarray(ids, np.int32)[None, :],
        "x_length": np.asarray([len(ids)], np.int32),
        "noise_scale": np.asarray([0.667], np.float32),
    }
    t0 = time.time()
    ac_out = infer(ac_eng, ac_ctx, cuda, feeds, keep)
    mel = ac_out.get("mel")
    if mel is None:
        raise RuntimeError("no mel in %s" % list(ac_out))
    cropped = crop_mel(mel)
    print(
        "mel_raw",
        tuple(mel.shape),
        "cropped",
        tuple(cropped.shape),
        "acoustic_s",
        round(time.time() - t0, 3),
        flush=True,
    )
    dump("G_acoustic_infer")

    t1 = time.time()
    vx_in = {"mels": np.ascontiguousarray(cropped[None, ...], dtype=np.float32)}
    vx_out = infer(vx_eng, vx_ctx, cuda, vx_in, keep)
    mag = vx_out.get("mag")
    x = vx_out.get("x")
    y = vx_out.get("y")
    if mag is None or x is None or y is None:
        raise RuntimeError("vocos outs %s" % list(vx_out))
    pcm = _istft(np.asarray(mag[0]), np.asarray(x[0]), np.asarray(y[0]))
    print("vocos_s", round(time.time() - t1, 3), "pcm", pcm.shape, flush=True)
    write_wav(args.out, pcm)
    dump("H_first_wav")

    t2 = time.time()
    ac_out2 = infer(ac_eng, ac_ctx, cuda, feeds, keep)
    cropped2 = crop_mel(ac_out2["mel"])
    vx_out2 = infer(
        vx_eng,
        vx_ctx,
        cuda,
        {"mels": np.ascontiguousarray(cropped2[None, ...], dtype=np.float32)},
        keep,
    )
    pcm2 = _istft(vx_out2["mag"][0], vx_out2["x"][0], vx_out2["y"][0])
    out2 = args.out.replace(".wav", "_2.wav")
    write_wav(out2, pcm2)
    print("second_wav_s", round(time.time() - t2, 3), flush=True)
    dump("I_second_wav")

    u, p = cg()
    print(
        "CKPT_DONE wav=%s cgroup_usage=%s cgroup_max=%s"
        % (
            args.out,
            ("%.1f" % u) if u is not None else "NA",
            ("%.1f" % p) if p is not None else "NA",
        ),
        flush=True,
    )
    time.sleep(6)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        time.sleep(3)
        raise
