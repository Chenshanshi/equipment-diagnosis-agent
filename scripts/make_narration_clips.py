"""用微软在线语音（edge-tts）把解说词合成为各段音频，并转成 add_narration.py 需要的 WAV 格式。

注意：这一步会把解说词文本发送到微软的在线语音服务。
用法：python make_narration_clips.py 解说词.json 输出目录 语音名 [各段起始时间.json 视频秒数]
  语音名例如 zh-CN-XiaoxiaoNeural（女声）、zh-CN-YunxiNeural（男声）。
  给了"各段起始时间"和视频秒数时，会检查每段能否在下一段开始前说完，说不完就自动加快一点语速重新合成。
依赖：edge-tts、imageio-ffmpeg
"""
import asyncio
import json
import subprocess
import sys
import wave
from pathlib import Path

import edge_tts
import imageio_ffmpeg

sys.stdout.reconfigure(encoding="utf-8")
script = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
voice = sys.argv[3]
cues = json.loads(Path(sys.argv[4]).read_text(encoding="utf-8-sig")) if len(sys.argv) > 5 else None
video_len = float(sys.argv[5]) if len(sys.argv) > 5 else None
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


def window(key: str) -> float | None:
    """这段解说最多能说多久（到下一段开始前留 0.35 秒）。"""
    if not cues:
        return None
    later = sorted(v for v in list(cues.values()) + [video_len] if v > cues[key])
    return later[0] - cues[key] - 0.35


async def synth(key: str, text: str) -> float:
    limit = window(key)
    for rate in ("+0%", "+8%", "+16%", "+25%"):
        mp3, wav = out / f"{key}.mp3", out / f"{key}.wav"
        await edge_tts.Communicate(text, voice, rate=rate).save(str(mp3))
        # 转成 44.1 kHz / 16 位 / 单声道，并去掉开头和结尾的静音
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(mp3), "-af",
                        "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                        "silenceremove=start_periods=1:start_threshold=-45dB,areverse",
                        "-ar", "44100", "-ac", "1", "-sample_fmt", "s16", str(wav)], check=True)
        mp3.unlink()
        with wave.open(str(wav), "rb") as wf:
            dur = wf.getnframes() / wf.getframerate()
        if limit is None or dur <= limit:
            print(f"{key}: {dur:5.2f}s  语速 {rate}" + (f"  （上限 {limit:.2f}s）" if limit else ""))
            return dur
    print(f"{key}: {dur:5.2f}s  !! 语速加到 +25% 仍超过上限 {limit:.2f}s")
    return dur


async def main() -> None:
    durs = {k: round(await synth(k, t), 2) for k, t in script.items()}
    (out / "durations.json").write_text(json.dumps(durs), encoding="utf-8")
    print("合计 %.1f 秒" % sum(durs.values()))


asyncio.run(main())
