#!/usr/bin/env python3
"""Split Matcha load: base / acoustic-only / acoustic+vocos / +warmup.

Does not start ROS. Run as a throwaway container entrypoint so cgroup is
this process only. Do not docker exec into the live TTS server.

  python3 /deploy/bench_matcha_load_ckpt.py --mode base
  python3 /deploy/bench_matcha_load_ckpt.py --mode acoustic
  python3 /deploy/bench_matcha_load_ckpt.py --mode full
  python3 /deploy/bench_matcha_load_ckpt.py --mode full --warmup
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


def _mib(kb):
    return kb / 1024.0


def disable_sherpa_trt():
    try:
        import sherpa_onnx
    except Exception:
        return
    libdir = Path(sherpa_onnx.__file__).resolve().parent / "lib"
    for p in libdir.glob("libonnxruntime_providers_tensorrt*"):
        if str(p).endswith(".disabled"):
            continue
        dest = p.with_name(p.name + ".disabled")
        try:
            p.rename(dest)
            print("disabled", dest.name, flush=True)
        except OSError as e:
            print("disable_trt_skip", p.name, e, flush=True)


def find_cgroup():
    usage = peak = None
    cg = ""
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.strip().split(":")
            if len(parts) >= 3 and "memory" in parts[1]:
                cg = parts[2]
                break
            if line.startswith("0::"):
                cg = line.strip().split(":", 2)[-1]
    except Exception:
        cg = ""
    cid = os.environ.get("HOSTNAME", "")
    bases = []
    if cg:
        bases.append(Path("/sys/fs/cgroup") / cg.lstrip("/"))
        bases.append(Path("/sys/fs/cgroup/memory") / cg.lstrip("/"))
    bases.extend(
        [
            Path("/sys/fs/cgroup/memory/docker"),
            Path("/sys/fs/cgroup/memory"),
        ]
    )
    for base in bases:
        if not base.exists():
            continue
        for cur, pk in (
            (base / "memory.usage_in_bytes", base / "memory.max_usage_in_bytes"),
            (base / "memory.current", base / "memory.peak"),
        ):
            if cur.is_file():
                try:
                    usage = int(cur.read_text().split()[0]) / (1024.0 * 1024.0)
                except Exception:
                    usage = None
                if pk.is_file():
                    try:
                        peak = int(pk.read_text().split()[0]) / (1024.0 * 1024.0)
                    except Exception:
                        peak = None
                return usage, peak
        if base.name == "docker" and cid:
            for child in base.glob("*"):
                v1 = child / "memory.usage_in_bytes"
                if v1.is_file():
                    try:
                        usage = int(v1.read_text().split()[0]) / (1024.0 * 1024.0)
                        p1 = child / "memory.max_usage_in_bytes"
                        peak = (
                            int(p1.read_text().split()[0]) / (1024.0 * 1024.0)
                            if p1.is_file()
                            else None
                        )
                    except Exception:
                        pass
                    return usage, peak
    return usage, peak


def smaps_buckets(pid=None):
    pid = pid or os.getpid()
    rss = defaultdict(int)
    pss = defaultdict(int)
    cur = "unknown"
    path = Path("/proc/%d/smaps" % pid)
    for line in path.read_text(errors="replace").splitlines():
        if " kB" not in line[:30] and line[:1] in "0123456789abcdef":
            name = line.split()[-1] if len(line.split()) >= 6 else "[anon]"
            if name.startswith("/"):
                name = name.rsplit("/", 1)[-1]
            elif not (
                name.startswith("[")
                or name.startswith("anon")
                or name.startswith("dmabuf")
            ):
                name = "[anon]"
            cur = name
            continue
        if line.startswith("Rss:"):
            rss[cur] += int(line.split()[1])
        elif line.startswith("Pss:"):
            pss[cur] += int(line.split()[1])
    buckets = defaultdict(lambda: [0, 0])
    for name, kb in rss.items():
        if name == "[heap]":
            k = "heap"
        elif name.startswith("dmabuf"):
            k = "dmabuf"
        elif name in ("[anon]",):
            k = "anon"
        elif "cudnn" in name.lower():
            k = "cudnn"
        elif "cublas" in name.lower():
            k = "cublas"
        else:
            k = "other"
        buckets[k][0] += kb
        buckets[k][1] += pss[name]
    roll_rss = roll_pss = None
    roll = Path("/proc/%d/smaps_rollup" % pid)
    if roll.is_file():
        for line in roll.read_text().splitlines():
            if line.startswith("Rss:"):
                roll_rss = _mib(int(line.split()[1]))
            elif line.startswith("Pss:"):
                roll_pss = _mib(int(line.split()[1]))
    return buckets, roll_rss, roll_pss, sum(rss.values()), sum(pss.values())


def dump(tag):
    time.sleep(1.0)
    buckets, roll_rss, roll_pss, tot_rss, tot_pss = smaps_buckets()
    cg_u, cg_p = find_cgroup()
    print("=== CKPT %s ===" % tag, flush=True)
    print(
        "CKPT tag=%s cgroup_usage_mib=%s cgroup_max_mib=%s "
        "smaps_rss_mib=%.1f smaps_pss_mib=%.1f roll_rss=%s roll_pss=%s"
        % (
            tag,
            ("%.1f" % cg_u) if cg_u is not None else "NA",
            ("%.1f" % cg_p) if cg_p is not None else "NA",
            _mib(tot_rss),
            _mib(tot_pss),
            ("%.1f" % roll_rss) if roll_rss is not None else "NA",
            ("%.1f" % roll_pss) if roll_pss is not None else "NA",
        ),
        flush=True,
    )
    for k in ("heap", "dmabuf", "anon", "cudnn", "cublas", "other"):
        a, b = buckets.get(k, [0, 0])
        print(
            "CKPT %s %-8s rss=%.1f pss=%.1f" % (tag, k, _mib(a), _mib(b)),
            flush=True,
        )


def acoustic_only_onnx(src, cache_dir):
    sys.path.insert(0, "/work")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cached = cache / (Path(src).stem + ".audio_output.onnx")
    if cached.is_file() and cached.stat().st_size > 0:
        print("acoustic_alias reuse", cached, flush=True)
        return str(cached)
    from utils.matcha_skip_vocoder import alias_acoustic_audio_output

    return alias_acoustic_audio_output(src, str(cache))


def load_tts(mode):
    import sherpa_onnx

    model_dir = os.environ.get("TTS_MODEL_DIR", "/models/matcha-kai-16k-e500")
    cfg = os.environ.get("TTS_SHERPA_ORT_CONFIG", "/deploy/ort_cuda_jp5.config")
    ep = "cuda:%s" % cfg if os.path.isfile(cfg) else "cuda"
    acoustic = os.path.join(model_dir, "model-steps-3.onnx")
    vocoder = os.path.join(model_dir, "vocos-16khz-univ.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    lexicon = os.path.join(model_dir, "lexicon.txt")
    data_dir = os.path.join(model_dir, "espeak-ng-data")
    if not os.path.isdir(data_dir):
        data_dir = ""
    if not os.path.isfile(acoustic):
        raise SystemExit("missing acoustic %s" % acoustic)
    if mode == "acoustic":
        cache = os.environ.get("TTS_VOCOS_TRT_CACHE", "/opt/vocos_trt_cache")
        acoustic = acoustic_only_onnx(acoustic, cache)
        vocoder = ""
        print("mode=acoustic vocoder=SKIP alias=%s provider=%s" % (acoustic, ep), flush=True)
    else:
        if not os.path.isfile(vocoder):
            raise SystemExit("missing vocos %s" % vocoder)
        print("mode=full vocoder=%s provider=%s" % (vocoder, ep), flush=True)
    tts_kw = dict(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=acoustic,
                vocoder=vocoder,
                lexicon=lexicon if os.path.isfile(lexicon) else "",
                tokens=tokens,
                data_dir=data_dir,
                noise_scale=0.667,
            ),
            num_threads=2,
            provider=ep,
        ),
        rule_fsts="",
    )
    try:
        tts_config = sherpa_onnx.OfflineTtsConfig(**tts_kw, max_num_sentences=-1)
    except TypeError:
        tts_config = sherpa_onnx.OfflineTtsConfig(**tts_kw)
    return sherpa_onnx.OfflineTts(tts_config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("base", "acoustic", "full"), required=True)
    ap.add_argument("--warmup", action="store_true")
    args = ap.parse_args()

    disable_sherpa_trt()
    dump("import_before_sherpa")
    import sherpa_onnx  # noqa: F401

    dump("A_base")
    if args.mode == "base":
        print("CKPT_DONE mode=base", flush=True)
        time.sleep(5)
        return

    tts = load_tts(args.mode)
    tag = "B_acoustic" if args.mode == "acoustic" else "C_full"
    dump(tag)
    if args.warmup:
        audio = tts.generate("你好，我是陆风。", sid=0, speed=1.0)
        n = 0
        if audio is not None:
            samples = getattr(audio, "samples", None)
            if samples is not None:
                n = len(samples)
        print("warmup_samples", n, flush=True)
        dump("D_warmup")
    print("CKPT_DONE mode=%s" % args.mode, flush=True)
    time.sleep(8)


if __name__ == "__main__":
    main()
