"""命令行跑一轮智能体，打印事件流（用于联调，不需要网页）。用法：python run_cli.py [问题]"""
import asyncio
import json
import sys

from agent_core import DiagnosisAgent

Q = sys.argv[1] if len(sys.argv) > 1 else "缸体线的设备最近 24 小时有没有异常？有的话出个诊断报告，看看备件够不够。"


async def main():
    a = DiagnosisAgent()
    await a.start()
    print("model:", a.model, "| tools:", [t["function"]["name"] for t in a.tools], flush=True)
    text = ""
    try:
        async for ev in a.run(Q):
            t = ev["type"]
            if t == "text_delta":
                text += ev["content"]
                continue
            if t == "text_end":
                print("[assistant]", text.strip(), flush=True)
                text = ""
            elif t == "tool_call":
                print("[tool_call]", ev["name"], json.dumps(ev["args"], ensure_ascii=False), flush=True)
            elif t == "tool_result":
                s = json.dumps(ev["result"], ensure_ascii=False, default=str)
                print("[tool_result]", ev["name"], "error" if ev["is_error"] else "ok", f'{ev["seconds"]}s', s[:160], flush=True)
            elif t == "thinking":
                print(f"[round {ev['round']}]", flush=True)
            else:
                print(f"[{t}]", json.dumps({k: v for k, v in ev.items() if k != "type"}, ensure_ascii=False), flush=True)
    finally:
        await a.close()


asyncio.run(main())
