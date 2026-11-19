"""设备诊断智能体核心：DeepSeek 大模型（函数调用）+ MCP 工具。

流程：
1. 以 stdio 方式启动 diag_tools 的 MCP 服务，list_tools 拿到工具清单；
2. 把 MCP 工具的 inputSchema 转成 OpenAI 兼容的 tools 定义交给 DeepSeek；
3. 大模型自己决定调用哪个工具、传什么参数；我们通过 MCP 执行，把结果回填；
4. 循环直到大模型给出最终回答。全过程以事件流（dict）形式产出，供网页实时展示。
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # 仓库根目录
REPORT_DIR = ROOT / "diag_tools" / "reports"
MAX_ROUNDS = 8
MAX_TOOL_CHARS = 6000

SYSTEM_PROMPT = """你是汽车工厂机加车间的“设备诊断助手”，通过工具访问产线传感器数据和 MES 系统。

工作方式：
1. 用户提到产线或设备时，先调用 query_equipment_records（query_type="equipment"）查台账，确定设备编号；
2. 对相关设备逐台调用 detect_multisensor_anomaly 做多传感器异常检测；用户说“最近 N 小时”就传 last_hours=N；
3. 对检测出异常段的设备调用 generate_diagnosis_report 生成诊断报告（同样传 last_hours），报告会给出疑似故障、建议检查项、历史同类工单和备件库存；
4. 最后用中文给出简洁结论：先用一句话说总体情况；然后每台设备一行加粗小标题（设备编号、名称、状态），下面用 3～4 条要点说明依据（哪个传感器贡献最大）、疑似故障、建议检查项、备件是否充足（库存为 0 的要明确提醒采购）；最后给一句处理建议。

要求：
- 全程只用简体中文，包括每次调用工具前的一句说明（例如“我先查一下缸体线有哪些设备。”）；
- 只使用工具返回的数据，不要编造数值；传感器贡献用百分比表示（例如 53%）；
- 不要写报告的文件名或文件路径（界面上有查看报告的按钮）；
- 最终回答控制在 300 字以内，用 Markdown 列表。"""


def load_env() -> None:
    """读取同目录 .env（KEY=VALUE 每行一个），不覆盖已有环境变量。"""
    env = HERE / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _tool_payload(result) -> Any:
    sc = getattr(result, "structured_content", None)
    if sc is not None:
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    try:
        return json.loads(text)
    except ValueError:
        return text


class DiagnosisAgent:
    def __init__(self) -> None:
        load_env()
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise RuntimeError(f"未找到 DEEPSEEK_API_KEY，请在 {HERE / '.env'} 中填写")
        self.model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
        self.llm = AsyncOpenAI(api_key=key, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools: list[dict] = []

    async def start(self) -> None:
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "diag_tools.server"], cwd=str(ROOT),
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        listed = await self.session.list_tools()
        self.tools = [
            {"type": "function", "function": {
                "name": t.name,
                "description": (t.description or "")[:1000],
                "parameters": t.input_schema if hasattr(t, "input_schema") else t.inputSchema,
            }}
            for t in listed.tools
        ]

    async def close(self) -> None:
        await self._stack.aclose()

    async def make_title(self, question: str) -> str | None:
        """用大模型给对话起一个短标题；失败返回 None，调用方继续用问题的第一句当标题。"""
        for _ in range(2):  # 模型偶尔只返回思考过程、正文为空，这时再试一次
            try:
                r = await self.llm.chat.completions.create(
                    model=self.model, temperature=0.2, max_tokens=800,
                    messages=[
                        {"role": "system", "content": "你是标题生成器。用不超过 10 个字概括用户的问题，作为对话标题。"
                                                      "只输出标题本身，不要标点、引号和解释。"},
                        {"role": "user", "content": question[:300]},
                    ],
                )
                text = (r.choices[0].message.content or "") if r.choices else ""
            except Exception:
                return None
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
            title = lines[0].strip(" \t\"'“”‘’《》「」。.!！?？：:") if lines else ""
            if title:
                return title[:20]
        return None

    async def run(self, user_text: str, history: list[dict] | None = None) -> AsyncIterator[dict]:
        """执行一轮对话，逐步产出事件。"""
        assert self.session is not None
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += [m for m in (history or []) if m.get("role") in ("user", "assistant")]
        messages.append({"role": "user", "content": user_text})
        t_start = time.perf_counter()
        n_calls = 0

        for round_no in range(1, MAX_ROUNDS + 1):
            yield {"type": "thinking", "round": round_no}
            stream = await self.llm.chat.completions.create(
                model=self.model, messages=messages, tools=self.tools,
                tool_choice="auto", temperature=0.2, stream=True,
            )
            text, calls = "", {}
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text += delta.content
                    yield {"type": "text_delta", "content": delta.content}
                for tc in delta.tool_calls or []:
                    c = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        c["id"] = tc.id
                    if tc.function and tc.function.name:
                        c["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        c["arguments"] += tc.function.arguments
            if text:
                yield {"type": "text_end"}

            if not calls:
                yield {"type": "done", "seconds": round(time.perf_counter() - t_start, 1),
                       "tool_calls": n_calls, "model": self.model}
                return

            ordered = [calls[i] for i in sorted(calls)]
            messages.append({"role": "assistant", "content": text or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                for c in ordered
            ]})
            for c in ordered:
                try:
                    args = json.loads(c["arguments"] or "{}")
                except ValueError:
                    args = {}
                yield {"type": "tool_call", "id": c["id"], "name": c["name"], "args": args}
                t0 = time.perf_counter()
                try:
                    r = await self.session.call_tool(c["name"], args)
                    payload, is_err = _tool_payload(r), bool(getattr(r, "is_error", False))
                except Exception as exc:  # 工具异常也回填给模型，让它自行处理
                    payload, is_err = f"工具调用失败：{exc}", True
                n_calls += 1
                yield {"type": "tool_result", "id": c["id"], "name": c["name"], "result": payload,
                       "is_error": is_err, "seconds": round(time.perf_counter() - t0, 2)}
                content = json.dumps(payload, ensure_ascii=False, default=str)
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": content[:MAX_TOOL_CHARS]})

        yield {"type": "error", "message": f"超过最大轮数 {MAX_ROUNDS}，已停止"}
