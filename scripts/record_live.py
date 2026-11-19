"""录制智能体真实运行过程（一镜到底，不剪辑）。

流程：打字提问 → 大模型规划并调用 MCP 工具 → 给出结论 → 回看检测卡片和报告卡片
      → 打开诊断报告并下翻 → 回到结论。
前提：app.py 已在运行（默认 http://127.0.0.1:8770，别的端口用环境变量 DIAG_URL 指定）。
用法：python record_live.py 输出.mp4 [截图目录] [问题]
配解说：环境变量 DIAG_NARR 指向一个 JSON（各段解说音频的秒数，键为 A~F）。录制时每个画面会停留到
        对应解说说完，并输出每段解说的起始时刻（CUES），再用 add_narration.py 合成音轨。
依赖：playwright（用系统 Edge，channel="msedge"）、imageio-ffmpeg、pillow、numpy
"""
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

sys.stdout.reconfigure(encoding="utf-8")
URL = os.environ.get("DIAG_URL", "http://127.0.0.1:8770/")
OUT = sys.argv[1]
SNAP_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else None
QUESTION = sys.argv[3] if len(sys.argv) > 3 else "缸体线的设备最近 24 小时有没有异常？有的话出个诊断报告，看看备件够不够。"
FPS = 12
W, H, DSF = 1360, 765, 24 / 17      # 页面按 1360x765 排版（放得下左侧对话记录栏），放大渲染成 1920x1080
NARR: dict[str, float] = (json.loads(Path(os.environ["DIAG_NARR"]).read_text(encoding="utf-8-sig"))
                          if os.environ.get("DIAG_NARR") else {})

frames: list[tuple[float, bytes]] = []
marks: dict[str, float] = {}
cues: dict[str, float] = {}


def ease(p: float) -> float:
    return 2 * p * p if p < 0.5 else 1 - (-2 * p + 2) ** 2 / 2


def main() -> None:
    with sync_playwright() as pw:
        br = pw.chromium.launch(channel="msedge", headless=True)
        ctx = br.new_context(viewport={"width": W, "height": H}, device_scale_factor=DSF)
        pg = ctx.new_page()
        pg.goto(URL)
        pg.wait_for_function("window.__ready === true && window.__convReady === true", timeout=30000)
        pg.wait_for_timeout(400)
        t0 = time.perf_counter()

        def now() -> float:
            return time.perf_counter() - t0

        def grab() -> None:
            frames.append((now(), pg.screenshot(type="jpeg", quality=92)))

        def tick(t_start: float) -> None:
            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t_start)))

        def hold(seconds: float, until: str | None = None, limit: float = 300) -> None:
            """持续截图 seconds 秒；给了 until 则截到条件成立（最长 limit 秒）。"""
            start = time.perf_counter()
            while True:
                t = time.perf_counter()
                grab()
                if until is None and t - start >= seconds:
                    break
                if until is not None and (pg.evaluate(until) or t - start > limit):
                    break
                tick(t)

        def cue(k: str, lead: float = 0.3) -> None:
            """记下第 k 段解说的起始时刻（从现在起 lead 秒后开始说）。"""
            cues[k] = round(now() + lead, 2)

        def rest(k: str, default: float, tail: float = 0.45) -> None:
            """停留至少 default 秒；配解说时，停到第 k 段说完再多 tail 秒。"""
            left = cues[k] + NARR[k] + tail - now() if (k in NARR and k in cues) else 0.0
            hold(max(default, left))

        def snap(name: str) -> None:
            marks[name] = round(now(), 2)
            if SNAP_DIR:
                SNAP_DIR.mkdir(parents=True, exist_ok=True)
                (SNAP_DIR / f"{name}.jpg").write_bytes(frames[-1][1])

        def pan_chat(target_js: str, dur: float = 1.0) -> None:
            """把对话区平滑滚动到 target_js 算出的位置。"""
            pg.evaluate("document.querySelector('#chat').style.scrollBehavior = 'auto'")
            start = pg.evaluate("document.querySelector('#chat').scrollTop")
            end = pg.evaluate(target_js)
            steps = max(1, int(dur * FPS))
            for i in range(1, steps + 1):
                t = time.perf_counter()
                pg.evaluate(f"document.querySelector('#chat').scrollTop = {start + (end - start) * ease(i / steps):.1f}")
                grab()
                tick(t)

        def top_of(selector: str, index: int = 0) -> str:
            return f"""(() => {{
                const c = document.querySelector('#chat'), el = document.querySelectorAll({json.dumps(selector)})[{index}];
                if (!el) return c.scrollTop;
                const y = c.scrollTop + el.getBoundingClientRect().top - c.getBoundingClientRect().top - 14;
                return Math.max(0, Math.min(y, c.scrollHeight - c.clientHeight));
            }})()"""

        bottom = "(() => { const c = document.querySelector('#chat'); return c.scrollHeight - c.clientHeight; })()"

        # 1) 欢迎页 → 打字提问（解说 A：介绍 + 念出问题）
        cue("A")
        typing = len(QUESTION) * 0.095                       # 打字大约要用的秒数
        hold(max(1.5, NARR.get("A", 0) - typing - 0.2)); snap("01_welcome")
        pg.click("#q")
        for ch in QUESTION:
            pg.keyboard.type(ch)
            grab()
            time.sleep(0.025)
        rest("A", 0.6); snap("02_typed")
        pg.keyboard.press("Enter")

        # 2) 真实运行，直到智能体给出结论（解说 B：大模型怎么规划）
        cue("B", 0.4)
        hold(0, until="window.__done === true")
        print("DONE_AT=%.1f" % now(), flush=True)
        print("STATS=" + json.dumps(pg.evaluate("window.__stats"), ensure_ascii=False), flush=True)
        print("CONV=" + str(pg.evaluate("currentId")), flush=True)
        rest("B", 2.6); snap("03_conclusion")

        # 3) 回看：检测卡片（解说 C）→ 报告卡片（解说 D）
        cue("C")
        pan_chat(top_of(".tool.c-brand"), 1.0)
        rest("C", 2.6); snap("04_detect_cards")
        if pg.locator(".rbtn").count():
            cue("D")
            pan_chat(top_of(".tool.c-warn"), 0.9)
            rest("D", 2.2); snap("05_report_cards")

            # 4) 打开第一份诊断报告并下翻（解说 E）
            pg.locator(".rbtn").first.click()
            hold(0.8)
            fr = next((f for f in pg.frames if "/reports/" in f.url), None)
            if fr is not None:
                try:
                    fr.wait_for_selector(".plotly-graph-div .main-svg", timeout=8000)
                except Exception:
                    print("WARN: 趋势图未在 8 秒内渲染", flush=True)
                cue("E", 0.2)
                hold(1.6); snap("06_report_top")
                total = fr.evaluate("document.documentElement.scrollHeight - innerHeight")
                steps = int((8 if NARR else 10) * FPS)
                for i in range(1, steps + 1):
                    t = time.perf_counter()
                    fr.evaluate(f"window.scrollTo(0, {total * ease(i / steps):.1f})")
                    grab()
                    if i == steps // 2:
                        snap("07_report_mid")
                    tick(t)
                rest("E", 1.3); snap("08_report_charts")
            pg.click("#rx")
            hold(0.5)

        # 5) 回到结论（解说 F：收尾）
        cue("F", 0.4)
        pan_chat(bottom, 1.0)
        rest("F", 3.0, tail=1.0); snap("09_final")
        br.close()

    # 按真实时间戳重采样成恒定帧率（不加速、不剪辑）
    import imageio_ffmpeg
    size = Image.open(io.BytesIO(frames[0][1])).size
    w = imageio_ffmpeg.write_frames(OUT, size, fps=FPS, codec="libx264", quality=8,
                                    pix_fmt_out="yuv420p", macro_block_size=8)
    w.send(None)
    dur = frames[-1][0]
    j, cache = 0, {}
    for k in range(int(dur * FPS) + 1):
        t = k / FPS
        while j + 1 < len(frames) and frames[j + 1][0] <= t:
            j += 1
        if j not in cache:
            cache.clear()
            cache[j] = np.asarray(Image.open(io.BytesIO(frames[j][1])).convert("RGB")).tobytes()
        w.send(cache[j])
    w.close()
    print("MARKS=" + json.dumps(marks), flush=True)
    print("CUES=" + json.dumps(cues), flush=True)
    print(f"SAVED size={size[0]}x{size[1]} duration={dur:.1f}s shots={len(frames)} capture_fps={len(frames) / dur:.1f}", flush=True)


if __name__ == "__main__":
    main()
