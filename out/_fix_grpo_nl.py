# -*- coding: utf-8 -*-
path = r"D:\workspace\xllm\trainer\train_grpo.py"
with open(path, encoding="utf-8") as f:
    src = f.read()

old = "            )    except KeyboardInterrupt:"
new = """            )
    except KeyboardInterrupt:"""
assert src.count(old) == 1, f"found {src.count(old)}"
src = src.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print("OK, fixed missing newline")
