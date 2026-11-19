"""生成演示数据：4 台数控加工中心 5 天的多传感器数据 + 模拟 MES 数据库。

传感器之间按物理关系耦合：
- 转速、电流由加工负载决定；振动随转速增大；
- 温度是负载的一阶滞后响应（时间常数 20 min）；冷却液压力随负载略降。

从第 3.5 天起注入三种渐进故障（CNC-01 一直正常，作为对照）：
- CNC-02 主轴轴承磨损：振动、温度逐渐升高
- CNC-03 冷却管路堵塞：冷却液压力下降、温度升高
- CNC-04 刀具磨损：电流升高，振动略增

运行：python -m diag_tools.demo_data
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from . import DATA_DIR, MES_DB, SENSOR_CSV, SENSORS

START = pd.Timestamp("2026-11-10 00:00")
DAYS = 5
STEPS = DAYS * 24 * 60  # 1 分钟一个点
FAULT_START_DAY = 3.5

EQUIPMENT = {
    "CNC-01": None,
    "CNC-02": "bearing",
    "CNC-03": "coolant",
    "CNC-04": "tool",
}


def _load_profile(rng: np.random.Generator) -> np.ndarray:
    """加工负载：两班制，工作时段内按 8~15 min 的工件节拍在空载/切削间切换。"""
    load = np.zeros(STEPS)
    t = 0
    while t < STEPS:
        hour = (t // 60) % 24
        working = 8 <= hour < 24  # 两班 8:00-24:00，夜里停机待机
        if working:
            dur = int(rng.integers(8, 16))
            load[t : t + dur] = rng.uniform(0.55, 0.9)
            t += dur
            gap = int(rng.integers(1, 4))  # 换件空载
            load[t : t + gap] = 0.12
            t += gap
        else:
            load[t : t + 30] = 0.05
            t += 30
    return load[:STEPS]


def _first_order(x: np.ndarray, tau: float, y0: float) -> np.ndarray:
    y = np.empty_like(x)
    a = 1.0 / tau
    prev = y0
    for i, v in enumerate(x):
        prev = prev + a * (v - prev)
        y[i] = prev
    return y


def simulate_machine(equipment_id: str, fault: str | None, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    load = _load_profile(rng)
    n = STEPS
    speed = 9000 * load + rng.normal(0, 60, n)
    current = 4 + 28 * load + rng.normal(0, 0.5, n)
    vib = 0.5 + 2.0 * (np.clip(speed, 0, None) / 9000) + rng.normal(0, 0.08, n)
    temp_target = 32 + 22 * load
    temp = _first_order(temp_target, tau=20, y0=34) + rng.normal(0, 0.25, n)
    pressure = 6.2 - 0.6 * load + rng.normal(0, 0.06, n)

    days = np.arange(n) / (24 * 60)
    ramp = np.clip((days - FAULT_START_DAY) / (DAYS - FAULT_START_DAY), 0, 1) ** 1.5
    working = load > 0.3

    if fault == "bearing":
        vib += ramp * (1.6 + 0.6 * rng.standard_normal(n).clip(-1, 1)) * working
        temp += _first_order(ramp * 9 * working, tau=15, y0=0)
    elif fault == "coolant":
        pressure -= ramp * 2.6
        temp += _first_order(ramp * 11 * working, tau=15, y0=0)
    elif fault == "tool":
        current += ramp * 12 * working
        vib += ramp * 0.4 * working

    return pd.DataFrame(
        {
            "timestamp": START + pd.to_timedelta(np.arange(n), unit="min"),
            "equipment_id": equipment_id,
            "spindle_speed_rpm": speed.round(0),
            "motor_current_a": current.round(2),
            "spindle_vibration_mm_s": vib.round(3),
            "spindle_temp_c": temp.round(2),
            "coolant_pressure_bar": pressure.round(3),
        }
    )


def build_sensor_csv() -> pd.DataFrame:
    frames = [
        simulate_machine(eid, fault, seed=100 + k)
        for k, (eid, fault) in enumerate(EQUIPMENT.items())
    ]
    df = pd.concat(frames, ignore_index=True)
    assert list(df.columns[2:]) == SENSORS
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SENSOR_CSV, index=False, encoding="utf-8")
    return df


SCHEMA = """
CREATE TABLE equipment (
    equipment_id     TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    model            TEXT NOT NULL,
    line             TEXT NOT NULL,
    workshop         TEXT NOT NULL,
    commission_date  TEXT NOT NULL,
    status           TEXT NOT NULL,
    owner            TEXT NOT NULL,
    last_maintenance TEXT
);
CREATE TABLE work_orders (
    order_id     TEXT PRIMARY KEY,
    equipment_id TEXT NOT NULL REFERENCES equipment(equipment_id),
    created_at   TEXT NOT NULL,
    closed_at    TEXT,
    fault_type   TEXT NOT NULL,
    description  TEXT NOT NULL,
    cause        TEXT,
    action       TEXT,
    downtime_h   REAL,
    technician   TEXT,
    status       TEXT NOT NULL
);
CREATE TABLE spare_parts (
    part_no          TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    applicable_model TEXT NOT NULL,
    fault_type       TEXT NOT NULL,
    stock            INTEGER NOT NULL,
    location         TEXT NOT NULL
);
"""

EQUIPMENT_ROWS = [
    ("CNC-01", "1号立式加工中心", "VMC-850", "缸盖线", "机加一车间", "2021-06-18", "运行", "设备科-王工", "2026-10-20"),
    ("CNC-02", "2号立式加工中心", "VMC-850", "缸盖线", "机加一车间", "2021-06-18", "运行", "设备科-王工", "2026-09-12"),
    ("CNC-03", "3号卧式加工中心", "HMC-630", "缸体线", "机加一车间", "2022-03-05", "运行", "设备科-李工", "2026-10-02"),
    ("CNC-04", "4号卧式加工中心", "HMC-630", "缸体线", "机加一车间", "2022-03-05", "运行", "设备科-李工", "2026-08-27"),
]

WORK_ORDER_ROWS = [
    ("WO-2025-0412", "CNC-02", "2025-04-12 09:20", "2025-04-13 16:00", "主轴轴承磨损",
     "主轴异响，振动值超过 4.5 mm/s 报警", "前轴承润滑脂老化，滚道点蚀", "更换主轴前轴承组，重新加注润滑脂", 30.5, "张师傅", "已关闭"),
    ("WO-2025-0930", "CNC-03", "2025-09-30 14:05", "2025-09-30 18:30", "冷却系统堵塞",
     "冷却液压力低报警，工件表面有烧伤", "过滤网被切屑堵塞", "清洗过滤网和管路，更换冷却液", 4.5, "刘师傅", "已关闭"),
    ("WO-2026-0115", "CNC-04", "2026-01-15 10:40", "2026-01-15 12:10", "刀具磨损",
     "主电机电流偏高，加工尺寸超差", "粗铣刀片磨损超限", "更换刀片，调整刀具寿命管理参数", 1.5, "赵师傅", "已关闭"),
    ("WO-2026-0322", "CNC-01", "2026-03-22 08:15", "2026-03-22 09:00", "其他",
     "换型调试", "计划性换型", "更换夹具并试切", 0.75, "张师傅", "已关闭"),
    ("WO-2026-0617", "CNC-02", "2026-06-17 13:30", "2026-06-17 15:10", "润滑不良",
     "主轴温度偏高", "润滑泵供油量不足", "清洗润滑泵滤芯，调整供油间隔", 1.7, "张师傅", "已关闭"),
    ("WO-2026-0805", "CNC-03", "2026-08-05 16:00", "2026-08-05 17:20", "冷却系统堵塞",
     "冷却液压力波动", "喷嘴部分堵塞", "疏通喷嘴", 1.3, "刘师傅", "已关闭"),
    ("WO-2026-0827", "CNC-04", "2026-08-27 09:00", "2026-08-27 15:00", "计划保养",
     "季度保养", "—", "导轨润滑、皮带张紧检查", 6.0, "赵师傅", "已关闭"),
    ("WO-2026-1020", "CNC-01", "2026-10-20 09:00", "2026-10-20 14:00", "计划保养",
     "季度保养", "—", "主轴、导轨、冷却系统例行保养", 5.0, "张师傅", "已关闭"),
]

SPARE_ROWS = [
    ("SP-B7014", "主轴前轴承组 7014C/P4", "VMC-850", "主轴轴承磨损", 2, "备件库 A-03"),
    ("SP-GRS01", "主轴润滑脂 NBU15", "VMC-850", "润滑不良", 12, "备件库 A-05"),
    ("SP-FLT20", "冷却液过滤网 200 目", "HMC-630", "冷却系统堵塞", 6, "备件库 B-11"),
    ("SP-NZ06", "冷却喷嘴 φ6", "HMC-630", "冷却系统堵塞", 20, "备件库 B-12"),
    ("SP-INS45", "粗铣刀片 SEKT1204", "HMC-630", "刀具磨损", 150, "刀具室 T-02"),
    ("SP-PMP01", "冷却泵 CDLF2", "HMC-630", "冷却系统堵塞", 0, "需采购"),
]


def build_mes_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if MES_DB.exists():
        MES_DB.unlink()
    con = sqlite3.connect(MES_DB)
    try:
        con.executescript(SCHEMA)
        con.executemany("INSERT INTO equipment VALUES (?,?,?,?,?,?,?,?,?)", EQUIPMENT_ROWS)
        con.executemany("INSERT INTO work_orders VALUES (?,?,?,?,?,?,?,?,?,?,?)", WORK_ORDER_ROWS)
        con.executemany("INSERT INTO spare_parts VALUES (?,?,?,?,?,?)", SPARE_ROWS)
        con.commit()
    finally:
        con.close()


def build_all() -> None:
    df = build_sensor_csv()
    build_mes_db()
    print(f"传感器数据：{SENSOR_CSV}（{len(df)} 行）")
    print(f"MES 数据库：{MES_DB}")


if __name__ == "__main__":
    build_all()
