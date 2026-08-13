#!/usr/bin/env bash
# 3 short smoke lines against running MCP TTS (no deepseek-like product names).
# Warmup: 你好，这是测试。 / hello, this is a test.
set -euo pipefail

URL="${TTS_URL:-http://127.0.0.1:15730/tts/test}"
OUT="${OUT_DIR:-$HOME/fanyi/wav_out/smoke3}"
mkdir -p "$OUT"

WARMUP_ZH="你好，这是测试。"
WARMUP_EN="hello, this is a test."

# ~40 chars each; CN + CN/EN mix (no product-case leakage)
CASES=(
  "明天早上九点钟我们在公司南门的门口集合见面，请记得带上今天会议需要的材料。"
  "请你先打开 WiFi 连上办公室网络，确认信号稳定之后，再继续下载这次的系统更新包。"
  "请问从这里坐地铁去 Airport 大概要多久，中间需要换乘吗，有没有一条更快的路线？"
)

post() {
  local text="$1" wav="$2"
  python3 - "$URL" "$text" "$wav" <<'PY'
import base64, json, sys, urllib.request, time, wave, io
url, text, wav = sys.argv[1], sys.argv[2], sys.argv[3]
body = json.dumps({"text": text}).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=120) as r:
    raw = r.read()
wall = time.perf_counter() - t0
# /tts/test returns {"ok":true,"wav_b64":"..."} not raw WAV
payload = json.loads(raw.decode("utf-8"))
if not payload.get("ok") or "wav_b64" not in payload:
    raise SystemExit("bad response: " + repr(payload)[:200])
data = base64.b64decode(payload["wav_b64"])
open(wav, "wb").write(data)
dur = 0.0
if data[:4] == b"RIFF":
    with wave.open(io.BytesIO(data), "rb") as w:
        dur = w.getnframes() / float(w.getframerate())
rtf = wall / dur if dur > 0 else float("inf")
print(f"text={text!r}\n  wall={wall:.3f}s audio={dur:.3f}s rtf={rtf:.3f} bytes={len(data)} -> {wav}")
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
