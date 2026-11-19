"""把工具里"调用方可以改正的错误"转成 MCP 的 ToolError，让大模型能看到具体原因。

工具内部用普通的 ValueError / FileNotFoundError 表示"设备编号不存在、数据还没生成"这类情况，
这样直接用 Python 调用和写测试都方便；注册到 MCP 服务时再包一层，转成 SDK 认识的 ToolError。
大模型收到的就是 is_error=True 加上原始提示（比如"未知设备编号，可选：CNC-01…"），可以据此改参数重试，
而不是一句笼统的"工具执行出错"。其他异常属于真正的程序错误，不做转换，交给 SDK 记录完整堆栈。
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from typing import Any

from mcp.server.mcpserver.exceptions import ToolError


def tool_errors(fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """装饰一个 async 工具函数；functools.wraps 保留签名和文档，自动生成的参数说明不受影响。"""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except ToolError:
            raise
        except (ValueError, FileNotFoundError) as exc:
            raise ToolError(str(exc)) from exc

    return wrapper
