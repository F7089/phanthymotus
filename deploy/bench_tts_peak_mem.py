#!/usr/bin/env python3
"""Measure peak memory for Jetson TTS container: startup + inference.

Jetson uses unified memory — there is no discrete VRAM like discrete GPUs.
This script samples:
  1) tegrastats RAM (system-wide, most reliable on Orin)
  2) nvidia-smi memory.used (host and/or inside container, if available)
  3) docker stats container MemUsage

WAV outputs go to ~/fanyi/wav_out by default.

Examples:
  python3 deploy/bench_tts_peak_mem.py
  python3 deploy/bench_tts_peak_mem.py --restart --runs 5
  python3 deploy/bench_tts_peak_mem.py --container phanthymotus-tts-melo \\
      --image phanthymotus-perception-tts:927c9c6-jp61
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_OUT = Path.home() / "fanyi" / "wav_out"
DEFAULT_TEXT = "请打开 WiFi 后继续下载更新。"
WARMUP_ZH = "你好，这是测试。"
WARMUP_EN = "hello, this is a test."


@dataclass
class Sample:
    t: float
    tegra_ram_mb: float | None = None
    nvsmi_mb: float | None = None
    docker_mb: float | None = None


@dataclass
class Sampler:
    container: str
    interval_s: float = 0.2
    samples: list[Sample] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _th: threading.Thread | None = None
    _tegra_proc: subprocess.Popen | None = None
    _tegra_last: float | None = None
    _tegra_lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> None:
        self._stop.clear()
        if shutil.which("tegrastats"):
            self._tegra_proc = subprocess.Popen(
                ["tegrastats", "--interval", str(int(self.interval_s * 1000))],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            threading.Thread(target=self._tegra_reader, daemon=True).start()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def _tegra_reader(self) -> None:
        assert self._tegra_proc and self._tegra_proc.stdout
        # RAM 4321/15678MB ...
        pat = re.compile(r"RAM\s+(\d+)/(\d+)MB", re.I)
        for line in self._tegra_proc.stdout:
            if self._stop.is_set():
                break
            m = pat.search(line)
            if m:
                with self._tegra_lock:
                    self._tegra_last = float(m.group(1))

    def _nvsmi(self) -> float | None:
        for cmd in (
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            [
                "docker",
                "exec",
                self.container,
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
        ):
            try:
                out = subprocess.check_output(
                    cmd, text=True, stderr=subprocess.DEVNULL, timeout=2
                ).strip()
                if out:
                    return float(out.splitlines()[0].strip())
            except Exception:
                continue
        return None

    def _docker_mem(self) -> float | None:
        try:
            out = subprocess.check_output(
                [
                    "docker",
                    "stats",
                    self.container,
                    "--no-stream",
                    "--format",
                    "{{.MemUsage}}",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            ).strip()
            # e.g. "1.234GiB / 62.7GiB" or "850MiB / ..."
            used = out.split("/")[0].strip()
            return _parse_docker_size_to_mb(used)
        except Exception:
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._tegra_lock:
                tegra = self._tegra_last
            self.samples.append(
                Sample(
                    t=time.time(),
                    tegra_ram_mb=tegra,
                    nvsmi_mb=self._nvsmi(),
                    docker_mb=self._docker_mem(),
                )
            )
            self._stop.wait(self.interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=3)
        if self._tegra_proc:
            try:
                self._tegra_proc.terminate()
                self._tegra_proc.wait(timeout=2)
            except Exception:
                try:
                    self._tegra_proc.kill()
                except Exception:
                    pass

    def peak_since(self, t0: float) -> dict[str, float | None]:
        window = [s for s in self.samples if s.t >= t0]
        if not window:
            window = list(self.samples)

        def _peak(attr: str) -> float | None:
            vals = [getattr(s, attr) for s in window if getattr(s, attr) is not None]
            return max(vals) if vals else None

        def _base(attr: str) -> float | None:
            for s in window:
                v = getattr(s, attr)
                if v is not None:
                    return v
            return None

        out = {}
        for key, attr in (
            ("tegra_ram", "tegra_ram_mb"),
            ("nvsmi", "nvsmi_mb"),
            ("docker", "docker_mb"),
        ):
            b, p = _base(attr), _peak(attr)
            out[f"{key}_base_mb"] = b
            out[f"{key}_peak_mb"] = p
            out[f"{key}_delta_mb"] = (p - b) if (p is not None and b is not None) else None
        return out


def _parse_docker_size_to_mb(s: str) -> float:
    s = s.strip()
    m = re.match(r"([0-9.]+)\s*([KMGTP]?i?B)", s, re.I)
    if not m:
        raise ValueError(s)
    val = float(m.group(1))
    unit = m.group(2).lower()
    mult = {
        "b": 1 / (1024 * 1024),
        "kb": 1 / 1024,
        "kib": 1 / 1024,
        "mb": 1.0,
        "mib": 1.0,
        "gb": 1024.0,
        "gib": 1024.0,
        "tb": 1024.0 * 1024,
        "tib": 1024.0 * 1024,
    }.get(unit, 1.0)
    return val * mult


def wait_mcp(url: str, timeout_s: float = 180.0) -> None:
    # light probe: POST empty-ish should not hang forever once server is up
    t0 = time.time()
    last_err = ""
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps({"text": "ping"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read()
            data = json.loads(raw.decode("utf-8"))
            if data.get("ok") or "wav_b64" in data or "info" in data:
                return
        except Exception as e:
            last_err = str(e)
        time.sleep(2)
    raise TimeoutError(f"MCP not ready within {timeout_s}s: {last_err}")


def synthesize(url: str, text: str, timeout_s: float = 120.0) -> tuple[bytes, float]:
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    elapsed = time.perf_counter() - t0
    data = json.loads(raw.decode("utf-8"))
    if not data.get("ok"):
        raise RuntimeError(f"tts/test failed: {data}")
    return base64.b64decode(data["wav_b64"]), elapsed


def _repo_root() -> Path:
    # deploy/bench_tts_peak_mem.py -> repo root
    return Path(__file__).resolve().parents[1]


def cgroup_memory_bytes(container: str) -> dict[str, int | None]:
    """Read cgroup v2 memory.current / memory.peak for the container."""
    out: dict[str, int | None] = {"current": None, "peak": None}
    try:
        pid = subprocess.check_output(
            ["docker", "inspect", "-f", "{{.State.Pid}}", container],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if not pid or pid == "0":
            return out
        cg = ""
        with open(f"/proc/{pid}/cgroup", encoding="utf-8") as f:
            for line in f:
                # 0::/system.slice/docker-xxx.scope
                if line.startswith("0::"):
                    cg = line.strip().split(":", 2)[-1]
                    break
                parts = line.strip().split(":")
                if len(parts) >= 3 and "memory" in parts[1]:
                    cg = parts[2]
        if not cg:
            return out
        base = Path("/sys/fs/cgroup") / cg.lstrip("/")
        for key, name in (("current", "memory.current"), ("peak", "memory.peak")):
            p = base / name
            if p.is_file():
                out[key] = int(p.read_text().strip())
    except Exception:
        return out
    return out


def restart_container(
    image: str,
    name: str,
    mcp_port: int,
    ws_port: int,
    mount_fp32_config: bool = True,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Recreate container. Image may still bake INT8 config — mount host FP32 files."""
    subprocess.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--runtime",
        "nvidia",
        "--network",
        "host",
        "--privileged",
        "-e",
        "NVIDIA_VISIBLE_DEVICES=all",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "-e",
        f"MCP_PORT={mcp_port}",
        "-e",
        f"WS_PORT={ws_port}",
    ]
    for k, v in (extra_env or {}).items():
        if v is None or v == "":
            continue
        cmd += ["-e", f"{k}={v}"]
    if Path("/models").is_dir():
        cmd += ["-v", "/models:/models"]

    root = _repo_root()
    cfg = root / "perception" / "config.yaml"
    dl = root / "perception" / "utils" / "model_downloader.py"
    tts = root / "perception" / "plugins" / "tts.py"
    if mount_fp32_config and cfg.is_file() and dl.is_file():
        cmd += ["-v", f"{cfg}:/work/config.yaml:ro"]
        cmd += ["-v", f"{dl}:/work/utils/model_downloader.py:ro"]
        if tts.is_file():
            cmd += ["-v", f"{tts}:/work/plugins/tts.py:ro"]
        print(f"[restart] mounting host FP32 config from {cfg}")
    else:
        print(
            "[restart] WARN: host config not mounted; image-baked config may still be INT8"
        )

    cmd.append(image)
    print("[restart]", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--container", default="phanthymotus-tts-melo")
    ap.add_argument("--image", default="phanthymotus-perception-tts:927c9c6-jp61")
    ap.add_argument("--mcp-port", type=int, default=15730)
    ap.add_argument("--ws-port", type=int, default=15731)
    ap.add_argument(
        "--restart",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cold start: rm+run container before measure (default: true; use --no-restart to skip)",
    )
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--label", default="", help="tag written into report (sweep id)")
    ap.add_argument(
        "--ort-env",
        action="append",
        default=[],
        help="extra -e KEY=VAL for container (repeatable), e.g. TTS_ORT_CUDNN_MAX_WORKSPACE=0",
    )
    args = ap.parse_args()

    extra_env: dict[str, str] = {}
    for item in args.ort_env:
        if "=" not in item:
            raise SystemExit(f"bad --ort-env {item!r}, want KEY=VAL")
        k, v = item.split("=", 1)
        extra_env[k] = v

    url = f"http://127.0.0.1:{args.mcp_port}/tts/test"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.label or "peakmem"
    report = args.out_dir / f"peak_mem_report_{tag}.json"

    print("NOTE: Jetson unified memory; prefer cgroup memory.peak + tegrastats.")
    print(f"container={args.container} url={url} label={tag}")
    print(f"out_dir={args.out_dir} ort_env={extra_env}")

    sampler = Sampler(container=args.container, interval_s=args.interval)
    sampler.start()
    time.sleep(0.5)

    phases: dict[str, dict] = {}
    cgroup_after_ready: dict[str, int | None] = {}
    cgroup_after_infer: dict[str, int | None] = {}

    try:
        if args.restart:
            t_boot = time.time()
            restart_container(
                args.image,
                args.container,
                args.mcp_port,
                args.ws_port,
                extra_env=extra_env,
            )
            print("waiting for MCP / first model load ...")
            wait_mcp(url, timeout_s=300)
            time.sleep(1.0)
            phases["startup_to_ready"] = sampler.peak_since(t_boot)
            cgroup_after_ready = cgroup_memory_bytes(args.container)
            print("[startup_to_ready]", phases["startup_to_ready"])
            print("[cgroup_after_ready]", cgroup_after_ready)
        else:
            print("using existing container (no --restart)")
            wait_mcp(url, timeout_s=60)
            cgroup_after_ready = cgroup_memory_bytes(args.container)

        # Always short bilingual warmup; --warmup repeats the pair if >1.
        for i in range(max(1, args.warmup)):
            print(f"[warmup {i}/{max(1, args.warmup)}] zh={WARMUP_ZH!r}")
            synthesize(url, WARMUP_ZH)
            print(f"[warmup {i}/{max(1, args.warmup)}] en={WARMUP_EN!r}")
            synthesize(url, WARMUP_EN)

        # Reset peak if kernel supports it (best-effort; may need root)
        try:
            pid = subprocess.check_output(
                ["docker", "inspect", "-f", "{{.State.Pid}}", args.container],
                text=True,
            ).strip()
            with open(f"/proc/{pid}/cgroup", encoding="utf-8") as f:
                cg = ""
                for line in f:
                    if line.startswith("0::"):
                        cg = line.strip().split(":", 2)[-1]
                        break
            peak_path = Path("/sys/fs/cgroup") / cg.lstrip("/") / "memory.peak"
            if peak_path.is_file() and os.access(peak_path, os.W_OK):
                peak_path.write_text("0\n")
                print(f"[cgroup] reset memory.peak via {peak_path}")
        except Exception as e:
            print(f"[cgroup] peak reset skipped: {e}")

        t_infer = time.time()
        rtf_rows = []
        for i in range(args.runs):
            wav, wall = synthesize(url, args.text)
            path = args.out_dir / f"peakmem_{tag}_run{i+1}.wav"
            path.write_bytes(wav)
            dur = 0.0
            if wav[:4] == b"RIFF":
                import wave
                import io

                with wave.open(io.BytesIO(wav), "rb") as w:
                    dur = w.getnframes() / float(w.getframerate())
            rtf = wall / dur if dur > 0 else float("inf")
            rtf_rows.append({"wall_s": wall, "audio_s": dur, "rtf": rtf, "wav": str(path)})
            print(f"[run {i+1}] wall={wall:.3f}s audio={dur:.3f}s rtf={rtf:.3f} -> {path}")

        phases["inference"] = sampler.peak_since(t_infer)
        cgroup_after_infer = cgroup_memory_bytes(args.container)
        print("[inference]", phases["inference"])
        print("[cgroup_after_infer]", cgroup_after_infer)

        def _mib(b: int | None) -> float | None:
            return None if b is None else round(b / 1024 / 1024, 1)

        summary = {
            "label": tag,
            "ort_env": extra_env,
            "container": args.container,
            "image": args.image if args.restart else "(existing)",
            "text": args.text,
            "phases": phases,
            "cgroup_after_ready_mib": {
                "current": _mib(cgroup_after_ready.get("current")),
                "peak": _mib(cgroup_after_ready.get("peak")),
            },
            "cgroup_after_infer_mib": {
                "current": _mib(cgroup_after_infer.get("current")),
                "peak": _mib(cgroup_after_infer.get("peak")),
            },
            "runs": rtf_rows,
            "avg_rtf": (
                sum(r["rtf"] for r in rtf_rows) / len(rtf_rows) if rtf_rows else None
            ),
            "note": (
                "Prefer cgroup memory.peak (kernel max) over docker stats; "
                "gpu_mem_limit only caps CUDA EP arena, not whole process."
            ),
        }
        report.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        print("---")
        print(f"label={tag} avg_rtf={summary['avg_rtf']}")
        print(
            "cgroup_ready_peak_mib=",
            summary["cgroup_after_ready_mib"]["peak"],
            "cgroup_infer_peak_mib=",
            summary["cgroup_after_infer_mib"]["peak"],
            "cgroup_infer_current_mib=",
            summary["cgroup_after_infer_mib"]["current"],
        )
        print(f"report: {report}")
    finally:
        sampler.stop()


if __name__ == "__main__":
    main()
