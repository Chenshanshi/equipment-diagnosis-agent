"""MCP 服务启动入口。

    # 默认 stdio：由智能体（MCP 客户端）当作子进程启动
    python -m diag_tools.server

    # 以 HTTP 方式对外提供，供其他支持 MCP 的平台接入
    python -m diag_tools.server --transport streamable-http --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import logging
import sys

from mcp.server.mcpserver import MCPServer

from .tools import register

INSTRUCTIONS = """
产线设备智能诊断服务。

可用工具：
- query_equipment_records：查设备台账 / 维修工单 / 备件库存（只读）
- detect_multisensor_anomaly：5 路传感器联合异常检测（PyTorch 自编码器）
- generate_diagnosis_report：生成诊断报告（疑似故障 + 历史工单 + 备件）

推荐流程：先查台账拿到设备编号 → 对设备做异常检测 → 有异常再生成诊断报告。
故障类型是规则推断的"疑似"结论，回答时要提醒需现场确认。
"""


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="产线设备诊断 MCP 服务")
    p.add_argument("--transport", "-t", default="stdio", choices=["stdio", "sse", "streamable-http"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", "-p", type=int, default=8765)
    args = p.parse_args(argv)

    # 日志只能写到 stderr：stdio 方式下 stdout 是协议通道
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    mcp = MCPServer("设备诊断", instructions=INSTRUCTIONS)
    register(mcp)
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
