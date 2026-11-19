"""产线设备诊断的三个 MCP 工具：多传感器异常检测、MES 台账/工单/备件查询、诊断报告。

- detect_multisensor_anomaly：PyTorch 自编码器，多传感器联合异常检测
- query_equipment_records：SQLite 模拟 MES，查设备台账、维修工单、备件
- generate_diagnosis_report：把检测结果和历史工单合成诊断报告（HTML）

演示数据是仿真生成的数控加工中心运行数据，见 demo_data.py。
"""

from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
DATA_DIR = PKG_DIR / "data"
MODEL_DIR = PKG_DIR / "models"
REPORT_DIR = PKG_DIR / "reports"

SENSOR_CSV = DATA_DIR / "sensor_data.csv"
MES_DB = DATA_DIR / "mes.db"

SENSORS = [
    "spindle_speed_rpm",
    "motor_current_a",
    "spindle_vibration_mm_s",
    "spindle_temp_c",
    "coolant_pressure_bar",
]

SENSOR_CN = {
    "spindle_speed_rpm": "主轴转速",
    "motor_current_a": "主电机电流",
    "spindle_vibration_mm_s": "主轴振动",
    "spindle_temp_c": "主轴温度",
    "coolant_pressure_bar": "冷却液压力",
}
