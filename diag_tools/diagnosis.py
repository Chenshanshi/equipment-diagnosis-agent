"""把异常检测结果 + MES 历史工单 + 备件库存合成诊断结论和 HTML 报告。

故障推断用的是可解释的规则：看异常段里哪几个传感器的误差占比最大，
对应到典型故障模式；再去工单库找同一设备的同类历史故障作为佐证。
结论是"疑似"，给维修人员做参考，不替代现场确认。
"""

from __future__ import annotations

import html
from datetime import datetime

import pandas as pd

from . import REPORT_DIR, SENSOR_CN, SENSORS
from . import anomaly, mes_db

# 主导传感器组合 -> (疑似故障, 建议检查项)
RULES = [
    ({"spindle_vibration_mm_s", "spindle_temp_c"}, "主轴轴承磨损", "检查主轴轴承游隙与润滑，听诊异响，必要时做振动频谱分析"),
    ({"coolant_pressure_bar", "spindle_temp_c"}, "冷却系统堵塞", "检查冷却液过滤网、喷嘴和冷却泵出口压力"),
    ({"coolant_pressure_bar"}, "冷却系统堵塞", "检查冷却液过滤网、喷嘴和冷却泵出口压力"),
    ({"motor_current_a", "spindle_vibration_mm_s"}, "刀具磨损", "检查刀片磨损量，核对刀具寿命计数，抽检加工尺寸"),
    ({"motor_current_a"}, "刀具磨损", "检查刀片磨损量，核对刀具寿命计数，抽检加工尺寸"),
    ({"spindle_temp_c"}, "润滑不良", "检查润滑泵供油量和润滑脂状态"),
    ({"spindle_vibration_mm_s"}, "主轴轴承磨损", "检查主轴轴承与动平衡"),
]


def infer_fault(contribution: dict[str, float]) -> tuple[str, str, list[str]]:
    """取累计占比达到 70% 的前几个传感器，按规则表匹配。"""
    top, acc = [], 0.0
    for k, v in sorted(contribution.items(), key=lambda kv: -kv[1]):
        top.append(k)
        acc += v
        if acc >= 0.7 or len(top) == 2:
            break
    s = set(top)
    for keys, fault, check in RULES:
        if keys <= s and (len(keys) > 1 or keys == {top[0]}):
            return fault, check, top
    return "未归类异常", "结合现场情况排查", top


def diagnose(equipment_id: str, last_hours: float | None = None) -> dict:
    eid = equipment_id.strip().upper()
    scores = anomaly.score_equipment(eid, last_hours)
    segs = anomaly.find_segments(scores)
    info = mes_db.query("equipment", equipment_id=eid)[0]

    n_win = len(scores)
    total_min = sum(s["minutes"] for s in segs)
    result: dict = {
        "equipment": info,
        "period": [str(scores["timestamp"].iloc[0]), str(scores["timestamp"].iloc[-1])],
        "threshold": float(scores["threshold"].iloc[0]),
        "windows": n_win,
        "anomaly_minutes": total_min,
        "segments": len(segs),
        "status": "正常",
    }
    if not segs:
        result["conclusion"] = "检测期间未发现持续异常，设备运行正常。"
        return result | {"scores": scores, "segment_list": []}

    # 用全部异常段的加权占比判断主要故障
    contrib = pd.Series(0.0, index=SENSORS)
    for s in segs:
        contrib += pd.Series(s["contribution"]) * s["minutes"]
    contrib = (contrib / contrib.sum()).sort_values(ascending=False)
    fault, check, top = infer_fault(contrib.to_dict())

    first = segs[0]["start"]
    recent = scores[scores["timestamp"] > scores["timestamp"].max() - pd.Timedelta(hours=12)]
    ratio_recent = float(recent["is_anomaly"].mean())
    level = "严重" if ratio_recent > 0.5 else ("注意" if ratio_recent > 0.1 else "轻微")

    history = mes_db.query("work_orders", equipment_id=eid, keyword=fault.replace("磨损", "").replace("堵塞", ""))
    parts = mes_db.query("spare_parts", equipment_id=eid, keyword=fault)
    result.update(
        status=level,
        suspected_fault=fault,
        dominant_sensors=[SENSOR_CN[t] for t in top],
        contribution={SENSOR_CN[k]: round(float(v), 3) for k, v in contrib.items()},
        first_anomaly=str(first),
        recent_12h_anomaly_ratio=round(ratio_recent, 3),
        check_items=check,
        similar_history=history,
        spare_parts=parts,
        conclusion=(
            f"{info['name']}自 {first:%m-%d %H:%M} 起出现持续异常，近 12 小时异常时间占比 "
            f"{ratio_recent:.0%}，主要由{'、'.join(SENSOR_CN[t] for t in top)}偏离正常耦合关系引起，"
            f"疑似{fault}。"
            + (f"该设备历史上有 {len(history)} 次同类工单。" if history else "")
        ),
    )
    return result | {"scores": scores, "segment_list": segs}


# ---------------------------------------------------------------------------
# HTML 报告
# ---------------------------------------------------------------------------

#: 状态 -> (主色, 浅底色)
_SEVERITY = {
    "正常": ("#12A150", "#E6F6EC"),
    "轻微": ("#B7791F", "#FFF7E0"),
    "注意": ("#C2610C", "#FFF1E3"),
    "严重": ("#D92D20", "#FDECEA"),
}

_REPORT_CSS = """
*{box-sizing:border-box}
body{margin:0;background:#F2F4F9;color:#0F1B3D;font:15px/1.65 "Microsoft YaHei UI","Microsoft YaHei","PingFang SC",sans-serif}
.page{max-width:1040px;margin:18px auto 28px;background:#fff;border-radius:16px;box-shadow:0 2px 18px rgba(15,27,61,.08);overflow:hidden}
.hero{display:flex;justify-content:space-between;align-items:flex-start;gap:16px;padding:22px 32px 20px;color:#fff;background:linear-gradient(100deg,#0A2585,#0935BF 52%,#2A66F5)}
.hero .k{font-size:12.5px;letter-spacing:3px;opacity:.78}
.hero h1{margin:2px 0 8px;font-size:26px;line-height:1.3}
.hero h1 .id{margin-left:12px;font:600 15px Consolas,monospace;padding:2px 10px;border-radius:8px;background:rgba(255,255,255,.18);vertical-align:3px}
.hero .meta{font-size:13.5px;opacity:.92}
.badge{flex:none;display:inline-flex;align-items:center;gap:8px;padding:6px 16px;border-radius:999px;background:#fff;color:var(--sev);font-weight:700;font-size:16px;margin-top:6px}
.badge i{width:9px;height:9px;border-radius:50%;background:var(--sev)}
.body{padding:20px 32px 26px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.kpi{border:1px solid #E5E9F2;border-radius:12px;padding:11px 14px 10px}
.kpi .l{font-size:12.5px;color:#8A94A8}
.kpi .v{font-size:20px;font-weight:700;margin-top:1px}
.kpi .v.sev{color:var(--sev)}
h2{display:flex;align-items:center;gap:9px;font-size:17px;margin:24px 0 10px}
h2:before{content:"";width:4px;height:17px;border-radius:2px;background:#0935BF}
.callout{background:var(--sev-soft);border-left:4px solid var(--sev);border-radius:8px;padding:12px 16px;font-size:15.5px}
.check{margin-top:10px;padding:10px 16px;background:#F5F7FB;border-radius:8px}
.check b{margin-right:10px;color:#0935BF}
.bars{display:grid;grid-template-columns:104px 1fr 52px;gap:9px 12px;align-items:center;max-width:760px}
.bars .n.top{font-weight:700;color:var(--sev)}
.track{height:14px;border-radius:7px;background:#EEF1F7;overflow:hidden}
.fill{height:100%;border-radius:7px;background:#B9C6E6}
.fill.top{background:var(--sev)}
.bars .p{font:600 14px Consolas,monospace;text-align:right;color:#46526E}
table{border-collapse:collapse;width:100%;font-size:14px}
th{background:#F3F6FC;color:#46526E;font-weight:600;text-align:left}
td,th{padding:8px 12px;border-bottom:1px solid #E5E9F2}
.pill{display:inline-block;padding:0 10px;border-radius:999px;font-size:13px;font-weight:600;line-height:22px}
.pill.ok{background:#E6F6EC;color:#12A150}
.pill.bad{background:#FDECEA;color:#D92D20}
.none{color:#8A94A8}
.foot{color:#8A94A8;font-size:12.5px;margin-top:16px;line-height:1.7}
"""


def _ensure_plotlyjs() -> None:
    """趋势图用报告目录里的本地 plotly.min.js，不依赖外网 CDN。"""
    js = REPORT_DIR / "plotly.min.js"
    if not js.exists():
        from plotly.offline import get_plotlyjs

        js.write_text(get_plotlyjs(), encoding="utf-8")


def _chart(scores: pd.DataFrame, raw: pd.DataFrame, top_cols: list[str], segs: list[dict] | None = None) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = 1 + len(top_cols)
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=["自编码器异常分数（对数坐标，红色虚线为阈值）"] + [SENSOR_CN[c] for c in top_cols])
    fig.add_trace(go.Scatter(x=scores["timestamp"], y=scores["score"], name="异常分数",
                             line=dict(width=1.8, color="#0935BF")), 1, 1)
    fig.add_trace(go.Scatter(x=scores["timestamp"], y=scores["threshold"], name="阈值",
                             line=dict(width=1.4, dash="dash", color="#D92D20")), 1, 1)
    for i, c in enumerate(top_cols, start=2):
        fig.add_trace(go.Scatter(x=raw["timestamp"], y=raw[c], name=SENSOR_CN[c],
                                 line=dict(width=1.1, color="#46526E")), i, 1)
    # 异常段用浅红色底纹标出
    for s in (segs or [])[-12:]:
        fig.add_vrect(x0=str(s["start"]), x1=str(s["end"]), fillcolor="rgba(217,45,32,0.07)",
                      line_width=0, row="all", col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1F7")
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F7", zeroline=False)
    fig.update_yaxes(type="log", dtick=1, row=1, col=1)
    fig.update_layout(height=230 * rows + 30, showlegend=False, template="plotly_white",
                      margin=dict(l=56, r=20, t=36, b=28),
                      font=dict(family="Microsoft YaHei, sans-serif", size=13, color="#46526E"))
    fig.update_annotations(font=dict(size=14, color="#0F1B3D"))
    return fig.to_html(full_html=False, include_plotlyjs="directory", config={"displayModeBar": False})


def write_report(result: dict) -> str:
    eq = result["equipment"]
    e = html.escape
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_plotlyjs()
    path = REPORT_DIR / f"诊断报告_{eq['equipment_id']}_{datetime.now():%Y%m%d_%H%M%S}.html"

    raw = anomaly.load_sensor_data()
    raw = raw[raw["equipment_id"] == eq["equipment_id"]]
    raw = raw[raw["timestamp"] >= pd.Timestamp(result["period"][0])]
    top_cols = [k for k, v in SENSOR_CN.items() if v in result.get("dominant_sensors", [])] or SENSORS[2:4]
    chart = _chart(result["scores"], raw, top_cols, result["segment_list"])

    def table(rows: list[dict], cols: list[tuple[str, str]], fmt: dict | None = None) -> str:
        """fmt: 列名 -> 返回已转义 HTML 的函数；其余单元格一律转义。"""
        if not rows:
            return '<p class="none">无</p>'
        fmt = fmt or {}
        head = "".join(f"<th>{e(c)}</th>" for _, c in cols)
        body = "".join(
            "<tr>" + "".join(
                f"<td>{fmt[k](r.get(k, '')) if k in fmt else e(str(r.get(k, '')))}</td>" for k, _ in cols
            ) + "</tr>"
            for r in rows
        )
        return f"<table><tr>{head}</tr>{body}</table>"

    def kpi(label: str, value: str, cls: str = "") -> str:
        return f'<div class="kpi"><div class="l">{e(label)}</div><div class="v {cls}">{e(value)}</div></div>'

    def stock_pill(v) -> str:
        n = int(v or 0)
        return f'<span class="pill bad">{n} · 需采购</span>' if n == 0 else f'<span class="pill ok">{n}</span>'

    abnormal = result["status"] != "正常"
    color, soft = _SEVERITY[result["status"]]
    if abnormal:
        kpis = (
            kpi("疑似故障", result["suspected_fault"], "sev")
            + kpi("近 12 小时异常占比", f"{result['recent_12h_anomaly_ratio']:.0%}", "sev")
            + kpi("首次异常", f"{pd.Timestamp(result['first_anomaly']):%m-%d %H:%M}")
            + kpi("异常段 / 累计时长", f"{result['segments']} 段 / {result['anomaly_minutes']} min")
        )
    else:
        kpis = (
            kpi("设备状态", "运行正常", "sev")
            + kpi("异常段", "0 段")
            + kpi("检测窗口", f"{result['windows']} 个")
            + kpi("告警阈值", f"{result['threshold']:.3f}")
        )

    doc = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>设备诊断报告 {e(eq['equipment_id'])}</title>
<style>:root{{--sev:{color};--sev-soft:{soft}}}{_REPORT_CSS}</style></head><body><div class="page">
<div class="hero"><div>
<div class="k">设备智能诊断报告</div>
<h1>{e(eq['name'])}<span class="id">{e(eq['equipment_id'])}</span></h1>
<div class="meta">{e(eq['workshop'])} · {e(eq['line'])}　型号 {e(eq['model'])}　责任人 {e(eq['owner'])}　上次保养 {e(str(eq['last_maintenance']))}</div>
<div class="meta">检测时段　{e(result['period'][0][:16])} ～ {e(result['period'][1][:16])}</div>
</div><div class="badge"><i></i>{e(result['status'])}</div></div>
<div class="body">
<div class="kpis">{kpis}</div>
<h2>一、结论</h2><div class="callout">{e(result['conclusion'])}</div>
"""
    if abnormal:
        seg_rows = [
            {"start": f"{s['start']:%m-%d %H:%M}", "end": f"{s['end']:%m-%d %H:%M}", "minutes": s["minutes"],
             "peak": f"{s['peak_score']:.2f}",
             "top": "、".join(SENSOR_CN[k] for k in list(s["contribution"])[:2])}
            for s in result["segment_list"][-8:]
        ]
        mx = max(result["contribution"].values()) or 1.0
        bars = "".join(
            f'<div class="n{" top" if i == 0 else ""}">{e(k)}</div>'
            f'<div class="track"><div class="fill{" top" if i == 0 else ""}" style="width:{v / mx:.0%}"></div></div>'
            f'<div class="p">{v:.0%}</div>'
            for i, (k, v) in enumerate(result["contribution"].items())
        )
        doc += f"""<div class="check"><b>建议检查</b>{e(result['check_items'])}</div>
<h2>二、异常定位（各传感器重构误差占比）</h2><div class="bars">{bars}</div>
<h2>三、异常时段（最近 8 段）</h2>{table(seg_rows, [("start", "开始"), ("end", "结束"), ("minutes", "持续/min"), ("peak", "峰值分数"), ("top", "主要传感器")])}
<h2>四、同类历史工单</h2>{table(result['similar_history'], [("order_id", "工单号"), ("created_at", "日期"), ("fault_type", "故障"), ("cause", "原因"), ("action", "处理"), ("downtime_h", "停机/h")])}
<h2>五、相关备件库存</h2>{table(result['spare_parts'], [("part_no", "料号"), ("name", "名称"), ("stock", "库存"), ("location", "库位")], {"stock": stock_pill})}
<h2>六、趋势图（浅红色底纹为异常段）</h2>{chart}
"""
    else:
        doc += f"""<h2>二、趋势图</h2>{chart}
"""
    doc += f"""<p class="foot">方法：多传感器 30 min 窗口自编码器（PyTorch），阈值取正常数据重构误差 {anomaly.THRESHOLD_Q}% 分位数（{result['threshold']:.3f}），连续超限 ≥ {anomaly.MIN_SEGMENT_MIN} min 记为异常段。故障类型为规则推断的"疑似"结论，需现场确认。</p>
</div></div></body></html>"""
    path.write_text(doc, encoding="utf-8")
    return str(path)
