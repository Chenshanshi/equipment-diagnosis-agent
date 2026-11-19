"""检查 .env 中的 DEEPSEEK_API_KEY 格式（不打印 Key 本身），并用它列出可用模型。"""
import asyncio
import os

from agent_core import load_env

load_env()
k = os.environ.get("DEEPSEEK_API_KEY", "")
print("has_key:", bool(k), "| starts_sk:", k.startswith("sk-"), "| length:", len(k),
      "| has_space_or_quote:", any(c in k for c in " \"'"))


async def main():
    from openai import AsyncOpenAI
    c = AsyncOpenAI(api_key=k, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    models = await c.models.list()
    print("models:", [m.id for m in models.data])
    print("configured model:", os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))


if k:
    asyncio.run(main())
