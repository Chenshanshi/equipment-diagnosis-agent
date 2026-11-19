"""把各段解说音频按起始时刻铺成一条音轨，合到视频里（视频画面不重新编码）。

用法：python add_narration.py 视频.mp4 音频目录 CUES 输出.mp4
  音频目录里是 A.wav、B.wav…（44.1 kHz、16 位、单声道）；
  CUES 是各段解说的起始秒数，形如 {"A": 0.3, "B": 8.9, ...}，可以直接给 JSON 字符串，也可以给存着它的文件路径。
依赖：numpy、imageio-ffmpeg（用它自带的 ffmpeg）
"""
import json
import subprocess
import sys
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
video, clip_dir, out = sys.argv[1], Path(sys.argv[2]), sys.argv[4]
cues = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8-sig") if Path(sys.argv[3]).is_file() else sys.argv[3])
SR = 44100
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

# 视频时长
probe = subprocess.run([ffmpeg, "-hide_banner", "-i", video], capture_output=True, text=True, encoding="utf-8", errors="replace")
h, m, s = [x for x in probe.stderr.split("Duration:")[1].split(",")[0].strip().split(":")]
duration = int(h) * 3600 + int(m) * 60 + float(s)

track = np.zeros(int(duration * SR) + SR, dtype=np.float32)
end_prev = 0.0
for key in sorted(cues, key=lambda k: cues[k]):
    with wave.open(str(clip_dir / f"{key}.wav"), "rb") as wf:
        assert (wf.getframerate(), wf.getsampwidth(), wf.getnchannels()) == (SR, 2, 1), "音频需为 44.1kHz/16位/单声道"
        x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    start = int(cues[key] * SR)
    track[start:start + len(x)] += x[: max(0, len(track) - start)]
    t_end = cues[key] + len(x) / SR
    print(f"{key}: {cues[key]:6.2f}s → {t_end:6.2f}s" + ("  !! 和上一段重叠" if cues[key] < end_prev else ""))
    end_prev = t_end
track = track[: int(duration * SR)]
peak = float(np.abs(track).max()) or 1.0
track = np.clip(track / peak * 0.89, -1, 1)                      # 统一响度，留一点余量

tmp = Path(out).with_suffix(".narration.wav")
with wave.open(str(tmp), "wb") as wf:
    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes((track * 32767).astype(np.int16).tobytes())

r = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-i", str(tmp),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-ac", "2",
                    "-movflags", "+faststart", "-shortest", out], capture_output=True, text=True, encoding="utf-8", errors="replace")
tmp.unlink(missing_ok=True)
if r.returncode != 0:
    sys.exit("ffmpeg 失败：" + r.stderr[-800:])
info = subprocess.run([ffmpeg, "-hide_banner", "-i", out], capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
print("视频时长 %.1f 秒；解说在 %.1f 秒说完" % (duration, end_prev))
print("\n".join(l.strip() for l in info.splitlines() if "Stream #" in l or "Duration" in l))
