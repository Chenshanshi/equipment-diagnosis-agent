"""多传感器自编码器异常检测（PyTorch）。

思路：只用正常工况数据训练自编码器，让它学会 5 个传感器之间的正常耦合关系
（负载→转速/电流、转速→振动、负载→温度滞后、负载→冷却压力）。故障会破坏这种
关系，重构误差随之升高。

- 输入：30 min 滑动窗口 × 5 个传感器，按正常数据的均值/方差标准化
- 网络：150 → 64 → 16 → 64 → 150，全连接 + ReLU
- 阈值：正常验证集窗口重构误差的 99.5% 分位数
- 告警：连续超过阈值 ≥ 10 min 才算一段异常，避免单点误报
- 定位：异常窗口里各传感器的重构误差占比，指出"哪个传感器不对劲"
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import MODEL_DIR, SENSOR_CSV, SENSORS

WINDOW = 30
TRAIN_END_DAY = 2.5  # 前 2.5 天训练
VAL_END_DAY = 3.5  # 2.5~3.5 天验证（仍为正常工况，用于定阈值）
MIN_SEGMENT_MIN = 10
THRESHOLD_Q = 99.5

MODEL_PATH = MODEL_DIR / "autoencoder.pt"
META_PATH = MODEL_DIR / "autoencoder_meta.json"


def _torch():
    try:
        import torch  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - 环境问题
        raise ValueError("需要安装 PyTorch：pip install torch") from exc
    return torch


def load_sensor_data() -> pd.DataFrame:
    if not SENSOR_CSV.exists():
        raise FileNotFoundError(
            f"未找到传感器数据 {SENSOR_CSV}，请先运行 python -m diag_tools.demo_data"
        )
    return pd.read_csv(SENSOR_CSV, parse_dates=["timestamp"])


def _windows(x: np.ndarray, stride: int = 1) -> np.ndarray:
    """(T, S) -> (N, WINDOW*S)，窗口按时间展平。"""
    idx = np.arange(0, len(x) - WINDOW + 1, stride)
    return np.stack([x[i : i + WINDOW].reshape(-1) for i in idx])


def _build_net(torch, dim: int):
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(dim, 64), nn.ReLU(),
        nn.Linear(64, 16), nn.ReLU(),
        nn.Linear(16, 64), nn.ReLU(),
        nn.Linear(64, dim),
    )


def _split_by_day(df: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    t0 = df["timestamp"].min()
    day = (df["timestamp"] - t0).dt.total_seconds() / 86400
    return df[(day >= lo) & (day < hi)]


def train(epochs: int = 40, seed: int = 0) -> dict:
    """用全部设备前 2.5 天的正常数据训练，2.5~3.5 天定阈值。"""
    torch = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = load_sensor_data()

    tr_parts, va_parts = [], []
    for _, g in df.groupby("equipment_id"):
        g = g.sort_values("timestamp")
        tr_parts.append(_split_by_day(g, 0, TRAIN_END_DAY)[SENSORS].to_numpy(float))
        va_parts.append(_split_by_day(g, TRAIN_END_DAY, VAL_END_DAY)[SENSORS].to_numpy(float))
    allx = np.concatenate(tr_parts)
    mean, std = allx.mean(0), allx.std(0) + 1e-6

    xtr = np.concatenate([_windows((p - mean) / std, stride=2) for p in tr_parts])
    xva = np.concatenate([_windows((p - mean) / std, stride=1) for p in va_parts])
    xtr_t = torch.tensor(xtr, dtype=torch.float32)
    xva_t = torch.tensor(xva, dtype=torch.float32)

    net = _build_net(torch, xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    history = []
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr_t), generator=gen)
        total = 0.0
        for i in range(0, len(perm), 256):
            batch = xtr_t[perm[i : i + 256]]
            opt.zero_grad()
            loss = loss_fn(net(batch), batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        history.append(total / len(xtr_t))

    net.eval()
    with torch.no_grad():
        sq_va = ((net(xva_t) - xva_t) ** 2).numpy()
    err_va = sq_va.mean(1)
    # 各传感器在正常数据上的平均误差，用来归一化"贡献度"
    sensor_base = sq_va.reshape(len(sq_va), WINDOW, len(SENSORS)).mean((0, 1))
    threshold = float(np.percentile(err_va, THRESHOLD_Q))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), MODEL_PATH)
    meta = {
        "sensors": SENSORS,
        "window": WINDOW,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "threshold": threshold,
        "sensor_baseline": sensor_base.tolist(),
        "train_windows": int(len(xtr)),
        "val_windows": int(len(xva)),
        "final_train_mse": history[-1],
        "epochs": epochs,
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


@dataclass
class _Model:
    net: object
    mean: np.ndarray
    std: np.ndarray
    threshold: float
    sensor_base: np.ndarray


_cache: dict[str, _Model] = {}


def load_model(auto_train: bool = True) -> _Model:
    if "m" in _cache:
        return _cache["m"]
    torch = _torch()
    if not (MODEL_PATH.exists() and META_PATH.exists()):
        if not auto_train:
            raise FileNotFoundError("模型未训练，请先运行 python -m diag_tools.anomaly")
        train()
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    net = _build_net(torch, WINDOW * len(SENSORS))
    net.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    net.eval()
    m = _Model(net, np.array(meta["mean"]), np.array(meta["std"]), float(meta["threshold"]),
               np.array(meta["sensor_baseline"]))
    _cache["m"] = m
    return m


def score_equipment(equipment_id: str, last_hours: float | None = None) -> pd.DataFrame:
    """返回每个窗口（以窗口末端时刻标记）的异常分数和各传感器误差。"""
    torch = _torch()
    df = load_sensor_data()
    known = sorted(df["equipment_id"].unique())
    if equipment_id not in known:
        raise ValueError(f"未知设备编号 {equipment_id!r}，可选：{', '.join(known)}")
    g = df[df["equipment_id"] == equipment_id].sort_values("timestamp")
    if last_hours is not None:
        if last_hours <= 0:
            raise ValueError("last_hours 必须为正数")
        g = g[g["timestamp"] > g["timestamp"].max() - pd.Timedelta(hours=last_hours)]
    if len(g) < WINDOW:
        raise ValueError(f"数据不足 {WINDOW} 分钟，无法检测")

    m = load_model()
    x = (g[SENSORS].to_numpy(float) - m.mean) / m.std
    w = _windows(x)
    with torch.no_grad():
        rec = m.net(torch.tensor(w, dtype=torch.float32)).numpy()
    sq = ((rec - w) ** 2).reshape(len(w), WINDOW, len(SENSORS))
    # 各传感器误差 ÷ 正常基线：>1 表示比正常时难重构，用于定位
    per_sensor = sq.mean(1) / m.sensor_base  # (N, S)
    out = pd.DataFrame(per_sensor, columns=SENSORS)
    out.insert(0, "score", sq.mean((1, 2)))
    out.insert(0, "timestamp", g["timestamp"].to_numpy()[WINDOW - 1 :])
    out["threshold"] = m.threshold
    out["is_anomaly"] = out["score"] > m.threshold
    return out


def find_segments(scores: pd.DataFrame, min_minutes: int = MIN_SEGMENT_MIN) -> list[dict]:
    flags = scores["is_anomaly"].to_numpy()
    segs, start = [], None
    for i, f in enumerate(np.append(flags, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_minutes:
                part = scores.iloc[start:i]
                contrib = (part[SENSORS] - 1).clip(lower=0).sum()
                contrib = (contrib / contrib.sum()).sort_values(ascending=False)
                segs.append(
                    {
                        "start": part["timestamp"].iloc[0],
                        "end": part["timestamp"].iloc[-1],
                        "minutes": int(i - start),
                        "peak_score": float(part["score"].max()),
                        "contribution": {k: float(v) for k, v in contrib.items()},
                    }
                )
            start = None
    return segs


if __name__ == "__main__":
    info = train()
    print(json.dumps({k: v for k, v in info.items() if k not in ("mean", "std")}, ensure_ascii=False, indent=2))
