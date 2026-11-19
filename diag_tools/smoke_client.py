"""通过 MCP 协议（streamable-http）调用服务，模拟智能体的完整调用链。

先启动服务：python -m diag_tools.server --transport streamable-http --port 8765
再运行：    python -m diag_tools.smoke_client
"""

import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8765/mcp"


def show(title, result):
    text = "".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    try:
        text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
    except ValueError:
        pass
    print(f"\n=== {title} ===\n{text[:1200]}")


async def main():
    async with streamable_http_client(URL) as (read, write, *_):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("工具：", [t.name for t in tools.tools])
            show("1 查缸体线设备", await s.call_tool("query_equipment_records", {"query_type": "equipment", "keyword": "缸体线"}))
            show("2 CNC-03 最近 24 小时异常检测", await s.call_tool("detect_multisensor_anomaly", {"equipment_id": "CNC-03", "last_hours": 24}))
            show("3 生成 CNC-03 诊断报告", await s.call_tool("generate_diagnosis_report", {"equipment_id": "CNC-03"}))


asyncio.run(main())
