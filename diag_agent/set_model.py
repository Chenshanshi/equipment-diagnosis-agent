"""只改 .env / .env.example 里的 DEEPSEEK_MODEL 行（不读出、不打印 Key）。用法：python set_model.py deepseek-flash"""
import re
import sys
from pathlib import Path

model = sys.argv[1]
here = Path(__file__).resolve().parent
for name in (".env", ".env.example"):
    p = here / name
    t = p.read_text(encoding="utf-8-sig")
    t2, n = re.subn(r"(?m)^DEEPSEEK_MODEL=.*$", f"DEEPSEEK_MODEL={model}", t)
    if n == 0:
        t2 = t.rstrip("\r\n") + f"\r\nDEEPSEEK_MODEL={model}\r\n"
    p.write_text(t2, encoding="utf-8")
    print(name, "-> DEEPSEEK_MODEL =", model)
