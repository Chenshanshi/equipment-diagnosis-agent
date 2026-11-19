"""对话记录的本地存储：每个对话一个 JSON 文件，放在 conversations/ 目录。

文件内容：
{
  "id": "20261119-081530-a1b2", "title": "…", "created_at": "…", "updated_at": "…",
  "turns": [
    {"user": "问题", "at": "…", "seconds": 5.2, "tool_calls": 5, "final": "最终回答",
     "events": [{"type": "text", "content": "…"}, {"type": "tool_call", …}, {"type": "tool_result", …}]}
  ]
}
events 按发生顺序保存，网页用它原样重放出当时的工具卡片和回答。
"""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime
from pathlib import Path

# 默认存在程序目录下的 conversations/；设置环境变量 DIAG_CONV_DIR 可以换成别的目录（录演示视频时用，避免弄乱真实记录）
CONV_DIR = Path(os.environ.get("DIAG_CONV_DIR") or Path(__file__).resolve().parent / "conversations")
_ID_RE = re.compile(r"^[0-9A-Za-z_-]{8,40}$")


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _path(cid) -> Path | None:
    """只接受自己生成的那种 id，防止用 ../ 之类的路径读写别的文件。"""
    if not isinstance(cid, str) or not _ID_RE.match(cid):
        return None
    return CONV_DIR / f"{cid}.json"


def fallback_title(text: str) -> str:
    """大模型标题还没生成（或生成失败）时，先用问题的第一句当标题。"""
    t = re.split(r"[？?。！!；;\n]", text.strip(), maxsplit=1)[0].strip()
    if len(t) > 18:
        t = re.split(r"[，,]", t, maxsplit=1)[0].strip()
    return (t[:18] + "…") if len(t) > 18 else (t or "新对话")


def create(first_message: str) -> dict:
    t = now()
    return {
        "id": datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2),
        "title": fallback_title(first_message),
        "created_at": t,
        "updated_at": t,
        "turns": [],
    }


def load(cid) -> dict | None:
    p = _path(cid)
    if p is None or not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save(conv: dict) -> None:
    p = _path(conv["id"])
    if p is None:
        raise ValueError(f"非法的对话 id：{conv.get('id')!r}")
    CONV_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(conv, ensure_ascii=False, default=str), encoding="utf-8")


def delete(cid) -> bool:
    p = _path(cid)
    if p is None or not p.exists():
        return False
    p.unlink()
    return True


def list_all() -> list[dict]:
    """所有对话的摘要，最近更新的排在前面。"""
    out = []
    if CONV_DIR.exists():
        for p in CONV_DIR.glob("*.json"):
            try:
                c = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            out.append({
                "id": c.get("id", p.stem),
                "title": c.get("title") or "新对话",
                "created_at": c.get("created_at"),
                "updated_at": c.get("updated_at"),
                "turns": len(c.get("turns", [])),
            })
    out.sort(key=lambda c: c.get("updated_at") or "", reverse=True)
    return out


def history_messages(conv: dict, max_turns: int = 6) -> list[dict]:
    """把已保存的轮次转成大模型的上下文（只带问题和最终回答，不带工具原始返回）。"""
    msgs: list[dict] = []
    for t in conv.get("turns", [])[-max_turns:]:
        if t.get("user"):
            msgs.append({"role": "user", "content": t["user"]})
        if t.get("final"):
            msgs.append({"role": "assistant", "content": t["final"]})
    return msgs
