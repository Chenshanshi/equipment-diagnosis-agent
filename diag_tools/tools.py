"""三个 MCP 工具：模块级 async 函数，由 register(mcp) 统一注册到 MCP 服务。"""

from __future__ import annotations

import logging
from typing import Literal, Optional

from mcp.server.mcpserver import Context, MCPServer

from ._errors import tool_errors

from . import SENSOR_CN
from . import anomaly, diagnosis, mes_db

logger = logging.getLogger(__name__)



async def detect_multisensor_anomaly(
    ctx: Context,
    equipment_id: str,
    last_hours: Optional[float] = None,
) -> dict:
    """对一台产线设备做多传感器联合异常检测（PyTorch 自编码器）。

    同时看主轴转速、主电机电流、主轴振动、主轴温度、冷却液压力 5 路信号，
    判断它们之间的耦合关系是否偏离正常工况，并指出偏离最大的传感器。

    Args:
        ctx: MCP 上下文（未使用）。
        equipment_id: 设备编号，如 "CNC-02"。不知道编号时先调用
            query_equipment_records(query_type="equipment") 列出设备。
        last_hours: 只检测最近多少小时；不填则检测全部数据（5 天）。

    Returns:
        异常分数阈值、异常时间占比、最近几段异常的起止时间和主要传感器。
    """
    eid = equipment_id.strip().upper()
    logger.info("detect_multisensor_anomaly %s last_hours=%s", eid, last_hours)
    scores = anomaly.score_equipment(eid, last_hours)
    segs = anomaly.find_segments(scores)
    contrib_total = None
    if segs:
        acc: dict[str, float] = {}
        for s in segs:
            for k, v in s["contribution"].items():
                acc[k] = acc.get(k, 0.0) + v * s["minutes"]
        tot = sum(acc.values())
        contrib_total = {SENSOR_CN[k]: round(v / tot, 3) for k, v in sorted(acc.items(), key=lambda kv: -kv[1])}
    return {
        "equipment_id": eid,
        "period": [str(scores["timestamp"].iloc[0]), str(scores["timestamp"].iloc[-1])],
        "threshold": round(float(scores["threshold"].iloc[0]), 4),
        "max_score": round(float(scores["score"].max()), 4),
        "anomaly_ratio": round(float(scores["is_anomaly"].mean()), 4),
        "anomaly_segments": len(segs),
        "first_anomaly": str(segs[0]["start"]) if segs else None,
        "sensor_contribution": contrib_total,
        "latest_segments": [
            {
                "start": str(s["start"]),
                "end": str(s["end"]),
                "minutes": s["minutes"],
                "peak_score": round(s["peak_score"], 3),
                "top_sensors": [SENSOR_CN[k] for k in list(s["contribution"])[:2]],
            }
            for s in segs[-5:]
        ],
        "note": "异常分数为自编码器重构误差；阈值取正常数据 99.5% 分位数，连续超限 ≥10 min 计为一段。",
    }


async def query_equipment_records(
    ctx: Context,
    query_type: Literal["equipment", "work_orders", "spare_parts"],
    equipment_id: Optional[str] = None,
    keyword: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """查询 MES 中的设备台账、维修工单或备件库存（只读）。

    Args:
        ctx: MCP 上下文（未使用）。
        query_type: "equipment" 设备台账；"work_orders" 维修工单（按时间倒序）；
            "spare_parts" 备件库存（按设备型号匹配）。
        equipment_id: 设备编号，如 "CNC-03"；不填则查全部。
        keyword: 关键词模糊匹配，如 "轴承"、"冷却"、"刀具"。
        limit: 最多返回条数，1~200。

    Returns:
        记录列表，每条是一个字段字典。
    """
    logger.info("query_equipment_records %s %s %s", query_type, equipment_id, keyword)
    return mes_db.query(query_type, equipment_id, keyword, limit)


async def generate_diagnosis_report(
    ctx: Context,
    equipment_id: str,
    last_hours: Optional[float] = None,
) -> dict:
    """生成设备诊断报告：异常检测 + 疑似故障推断 + 历史同类工单 + 备件库存。

    报告保存为 HTML（含趋势图），返回结论摘要和文件路径。

    Args:
        ctx: MCP 上下文（未使用）。
        equipment_id: 设备编号，如 "CNC-02"。
        last_hours: 只分析最近多少小时；不填则分析全部数据。

    Returns:
        状态等级、疑似故障、建议检查项、同类历史工单数、可用备件和报告路径。
    """
    eid = equipment_id.strip().upper()
    logger.info("generate_diagnosis_report %s", eid)
    r = diagnosis.diagnose(eid, last_hours)
    path = diagnosis.write_report(r)
    return {
        "equipment_id": eid,
        "status": r["status"],
        "conclusion": r["conclusion"],
        "suspected_fault": r.get("suspected_fault"),
        "check_items": r.get("check_items"),
        "similar_history": [
            f"{h['order_id']} {h['created_at'][:10]} {h['fault_type']}：{h['action']}"
            for h in r.get("similar_history", [])
        ],
        "spare_parts": [
            f"{p['name']}（库存 {p['stock']}，{p['location']}）" for p in r.get("spare_parts", [])
        ],
        "report_path": path,
    }


TOOLS = (detect_multisensor_anomaly, query_equipment_records, generate_diagnosis_report)


def register(mcp: MCPServer) -> None:
    for fn in TOOLS:
        mcp.tool()(tool_errors(fn))
