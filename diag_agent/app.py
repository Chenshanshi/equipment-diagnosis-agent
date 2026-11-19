"""设备诊断智能体网页：http://127.0.0.1:8770

运行：python diag_agent/app.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager

import uvicorn
from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse
from starlette.routing import Route

import conv_store
from agent_core import HERE, REPORT_DIR, DiagnosisAgent

AGENT: DiagnosisAgent | None = None
LOCK = asyncio.Lock()


@asynccontextmanager
async def lifespan(app):
    global AGENT
    AGENT = DiagnosisAgent()
    await AGENT.start()
    print(f"MCP 工具：{[t['function']['name'] for t in AGENT.tools]}；模型：{AGENT.model}", flush=True)
    yield
    await AGENT.close()


async def index(request: Request):
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-store"})


async def info(request: Request):
    return JSONResponse({"model": AGENT.model, "tools": [t["function"]["name"] for t in AGENT.tools]})


def _pack(ev: dict) -> dict:
    return {"data": json.dumps(ev, ensure_ascii=False, default=str)}


async def chat(request: Request):
    """一轮对话：流式返回事件，并把这一轮存进对话记录。"""
    body = await request.json()
    text = str(body.get("message", "")).strip()[:2000]
    if not text:
        return JSONResponse({"error": "empty message"}, status_code=400)

    conv = conv_store.load(body.get("conversation_id")) if body.get("conversation_id") else None
    is_new = conv is None
    if is_new:
        conv = conv_store.create(text)
    history = conv_store.history_messages(conv)

    async def gen():
        yield _pack({"type": "conversation", "id": conv["id"], "title": conv["title"], "new": is_new})
        # 新对话：让大模型同时起一个短标题，不占用主流程的时间
        title_task = asyncio.create_task(AGENT.make_title(text)) if is_new else None
        title_sent = False

        def title_event():
            nonlocal title_sent
            title_sent = True
            t = None if title_task.cancelled() or title_task.exception() else title_task.result()
            if not t:
                return None
            conv["title"] = t
            return _pack({"type": "title", "id": conv["id"], "title": t})

        events: list[dict] = []
        buf, final, seconds, calls = "", "", None, 0
        try:
            async with LOCK:  # 只有一个 MCP 会话，串行处理
                try:
                    async for ev in AGENT.run(text, history):
                        t = ev["type"]
                        if t == "text_delta":
                            buf += ev["content"]
                        elif t == "text_end":
                            events.append({"type": "text", "content": buf})
                            final, buf = buf, ""
                        elif t in ("tool_call", "tool_result", "error"):
                            events.append(ev)
                        elif t == "done":
                            seconds, calls = ev.get("seconds"), ev.get("tool_calls", 0)
                        yield _pack(ev)
                        if title_task and not title_sent and title_task.done():
                            m = title_event()
                            if m:
                                yield m
                except Exception as exc:
                    ev = {"type": "error", "message": f"{type(exc).__name__}: {exc}"}
                    events.append(ev)
                    yield _pack(ev)
        finally:
            # 无论正常结束、出错还是页面中途关闭，都把已经发生的内容存下来
            if buf:
                events.append({"type": "text", "content": buf})
            t_now = conv_store.now()
            conv["turns"].append({"user": text, "at": t_now, "seconds": seconds, "tool_calls": calls,
                                  "final": final, "events": events})
            conv["updated_at"] = t_now
            conv_store.save(conv)

        if title_task and not title_sent:
            try:
                await asyncio.wait_for(asyncio.shield(title_task), timeout=3)
            except Exception:
                pass
            if title_task.done():
                m = title_event()
                if m:
                    conv_store.save(conv)
                    yield m
            else:
                title_task.cancel()

    return EventSourceResponse(gen())


async def conversations(request: Request):
    return JSONResponse(conv_store.list_all(), headers={"Cache-Control": "no-store"})


async def conversation(request: Request):
    cid = request.path_params["cid"]
    if request.method == "DELETE":
        return JSONResponse({"deleted": conv_store.delete(cid)})
    conv = conv_store.load(cid)
    if conv is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(conv, headers={"Cache-Control": "no-store"})


async def report(request: Request):
    name = request.path_params["name"]
    path = (REPORT_DIR / name).resolve()
    if path.parent != REPORT_DIR.resolve() or not path.exists():
        return PlainTextResponse("not found", status_code=404)
    if name == "plotly.min.js":  # 报告里的趋势图用本地脚本，不依赖外网
        return FileResponse(path, media_type="text/javascript; charset=utf-8")
    if path.suffix != ".html":
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(path, media_type="text/html; charset=utf-8")


app = Starlette(
    routes=[
        Route("/", index),
        Route("/api/info", info),
        Route("/api/chat", chat, methods=["POST"]),
        Route("/api/conversations", conversations),
        Route("/api/conversations/{cid}", conversation, methods=["GET", "DELETE"]),
        Route("/reports/{name}", report),
    ],
    lifespan=lifespan,
)

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
