"""模拟 MES 的只读查询。

只开放固定的几类查询，SQL 全部参数化，不接受大模型拼出来的 SQL，
避免智能体误删数据或注入。
"""

from __future__ import annotations

import sqlite3
from typing import Literal

from . import MES_DB

QueryType = Literal["equipment", "work_orders", "spare_parts"]

_SQL = {
    "equipment": (
        "SELECT * FROM equipment WHERE (:eid IS NULL OR equipment_id = :eid) "
        "AND (:kw IS NULL OR name LIKE :kw OR line LIKE :kw OR model LIKE :kw) "
        "ORDER BY equipment_id LIMIT :lim"
    ),
    "work_orders": (
        "SELECT * FROM work_orders WHERE (:eid IS NULL OR equipment_id = :eid) "
        "AND (:kw IS NULL OR fault_type LIKE :kw OR description LIKE :kw OR cause LIKE :kw) "
        "ORDER BY created_at DESC LIMIT :lim"
    ),
    "spare_parts": (
        "SELECT * FROM spare_parts WHERE "
        "(:eid IS NULL OR applicable_model = (SELECT model FROM equipment WHERE equipment_id = :eid)) "
        "AND (:kw IS NULL OR name LIKE :kw OR fault_type LIKE :kw) "
        "ORDER BY part_no LIMIT :lim"
    ),
}


def _connect() -> sqlite3.Connection:
    if not MES_DB.exists():
        raise FileNotFoundError(
            f"未找到 MES 数据库 {MES_DB}，请先运行 python -m diag_tools.demo_data"
        )
    con = sqlite3.connect(f"file:{MES_DB.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def query(
    query_type: QueryType,
    equipment_id: str | None = None,
    keyword: str | None = None,
    limit: int = 20,
) -> list[dict]:
    if query_type not in _SQL:
        raise ValueError(f"query_type 只能是 {', '.join(_SQL)}")
    if not 1 <= limit <= 200:
        raise ValueError("limit 范围 1~200")
    params = {
        "eid": equipment_id.strip().upper() if equipment_id else None,
        "kw": f"%{keyword.strip()}%" if keyword and keyword.strip() else None,
        "lim": limit,
    }
    con = _connect()
    try:
        if params["eid"] is not None:
            hit = con.execute(
                "SELECT 1 FROM equipment WHERE equipment_id = ?", (params["eid"],)
            ).fetchone()
            if hit is None:
                ids = [r[0] for r in con.execute("SELECT equipment_id FROM equipment")]
                raise ValueError(f"未知设备编号 {equipment_id!r}，可选：{', '.join(ids)}")
        rows = con.execute(_SQL[query_type], params).fetchall()
    finally:
        con.close()
    return [dict(r) for r in rows]
