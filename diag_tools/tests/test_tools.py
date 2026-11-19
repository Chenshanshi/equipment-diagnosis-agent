"""工具测试：数据生成、自编码器检测、MES 查询、诊断报告、MCP 注册。

运行：python -m pytest diag_tools/tests -q
"""

import asyncio
from pathlib import Path

import pytest

from diag_tools import MES_DB, SENSOR_CSV, anomaly, demo_data, diagnosis, mes_db, tools


@pytest.fixture(scope="session", autouse=True)
def demo_env():
    if not (SENSOR_CSV.exists() and MES_DB.exists()):
        demo_data.build_all()
    if not anomaly.MODEL_PATH.exists():
        anomaly.train()
    yield


def run(coro):
    return asyncio.run(coro)


# ---------- 工具一：异常检测 ----------

def test_normal_machine_has_no_alarm():
    r = run(tools.detect_multisensor_anomaly(None, "CNC-01"))
    assert r["anomaly_segments"] == 0
    assert r["anomaly_ratio"] < 0.01


@pytest.mark.parametrize("eid,top", [
    ("CNC-02", {"主轴振动", "主轴温度"}),
    ("CNC-03", {"冷却液压力", "主轴温度"}),
    ("CNC-04", {"主电机电流", "主轴振动"}),
])
def test_faulty_machines_are_detected_and_located(eid, top):
    r = run(tools.detect_multisensor_anomaly(None, eid))
    assert r["anomaly_segments"] > 0
    # 故障从第 3.5 天开始注入，告警不应早于第 3 天
    assert r["first_anomaly"] >= "2026-11-13"
    first_sensor = next(iter(r["sensor_contribution"]))
    assert first_sensor in top


def test_last_hours_window():
    r = run(tools.detect_multisensor_anomaly(None, "cnc-02", last_hours=6))
    assert r["equipment_id"] == "CNC-02"
    assert r["period"][0] >= "2026-11-14 18:00"


def test_unknown_equipment_refused():
    with pytest.raises(ValueError, match="未知设备编号"):
        run(tools.detect_multisensor_anomaly(None, "CNC-99"))
    with pytest.raises(ValueError):
        run(tools.detect_multisensor_anomaly(None, "CNC-01", last_hours=-1))


# ---------- 工具二：MES 查询 ----------

def test_list_equipment():
    rows = run(tools.query_equipment_records(None, "equipment"))
    assert [r["equipment_id"] for r in rows] == ["CNC-01", "CNC-02", "CNC-03", "CNC-04"]


def test_work_orders_filter_and_order():
    rows = run(tools.query_equipment_records(None, "work_orders", "CNC-02"))
    assert rows and all(r["equipment_id"] == "CNC-02" for r in rows)
    assert rows == sorted(rows, key=lambda r: r["created_at"], reverse=True)
    hit = run(tools.query_equipment_records(None, "work_orders", keyword="冷却"))
    assert hit and all(r["equipment_id"] == "CNC-03" for r in hit)


def test_spare_parts_by_model():
    rows = run(tools.query_equipment_records(None, "spare_parts", "CNC-03"))
    assert rows and all(r["applicable_model"] == "HMC-630" for r in rows)


def test_query_is_injection_safe_and_read_only():
    rows = run(tools.query_equipment_records(None, "work_orders", keyword="'; DROP TABLE work_orders; --"))
    assert rows == []
    assert run(tools.query_equipment_records(None, "work_orders"))  # 表还在
    with pytest.raises(ValueError):
        mes_db.query("users")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        mes_db.query("equipment", limit=0)


# ---------- 工具三：诊断报告 ----------

@pytest.mark.parametrize("eid,fault", [
    ("CNC-02", "主轴轴承磨损"),
    ("CNC-03", "冷却系统堵塞"),
    ("CNC-04", "刀具磨损"),
])
def test_report_infers_fault_and_links_history(eid, fault):
    r = run(tools.generate_diagnosis_report(None, eid))
    assert r["suspected_fault"] == fault
    assert r["status"] in {"注意", "严重"}
    assert r["similar_history"], "应关联到同类历史工单"
    assert r["spare_parts"]
    p = Path(r["report_path"])
    assert p.exists() and fault in p.read_text(encoding="utf-8")


def test_report_for_normal_machine():
    r = run(tools.generate_diagnosis_report(None, "CNC-01"))
    assert r["status"] == "正常" and r["suspected_fault"] is None
    assert Path(r["report_path"]).exists()


def test_infer_fault_rules():
    f, _, _ = diagnosis.infer_fault({"motor_current_a": 0.8, "spindle_temp_c": 0.2})
    assert f == "刀具磨损"
    f, _, _ = diagnosis.infer_fault({"spindle_speed_rpm": 0.9, "spindle_temp_c": 0.1})
    assert f == "未归类异常"


# ---------- MCP 注册 ----------

def test_tools_register_on_mcp_server():
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer("t")
    tools.register(mcp)
    names = {t.name for t in run(mcp.list_tools())}
    assert names == {"detect_multisensor_anomaly", "query_equipment_records", "generate_diagnosis_report"}
