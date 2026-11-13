"""支持 python -m data_gen 直接运行。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())