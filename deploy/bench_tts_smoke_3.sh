#!/usr/bin/env bash
# 3 short smoke lines against running MCP TTS (no deepseek-like product names).
# Warmup: 你好，这是测试。 / hello, this is a test.
set -euo pipefail

URL="${TTS_URL:-http://127.0.0.1:15730/tts/test}"
OUT="${OUT_DIR:-$HOME/fanyi/wav_out/smoke3}"
mkdir -p "$OUT"

WARMUP_ZH="你好，这是测试。"
WARMUP_EN="hello, this is a test."

# ~20 chars each; CN + CN/EN mix (no product-case leakage)
CASES=(
  "明天早上九点在公司门口见面。"
  "请打开 WiFi 后继续下载更新。"
  "地铁去 Airport 大概要多久？"
)

post() {
  local text="$1" wav="$2"
  python3 - "$URL" "$text" "$wav" <<'PY'
import json, sys, urllib.request, time, wave, io
url, text, wav = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({"text": text}).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=120) as r:
    data = r.read()
wall = time.perf_counter() - t0
open(wav, "wb").write(data)
dur = 0.0
if data[:4] == b"RIFF":
    with wave.open(io.BytesIO(data), "rb") as w:
        dur = w.getnframes() / float(w.getframerate())
rtf = wall / dur if dur > 0 else float("inf")
print(f"text={text!r}\n  wall={wall:.3f}s audio={dur:.3f}s rtf={rtf:.3f} -> {wav}")
PY
}

echo "== warmup =="
post "$WARMUP_ZH" "$OUT/warmup_zh.wav"
post "$WARMUP_EN" "$OUT/warmup_en.wav"

echo "== measure 3 cases =="
i=1
for t in "${CASES[@]}"; do
  post "$t" "$OUT/case${i}.wav"
  i=$((i + 1))
done
echo "done: $OUT"
