#!/usr/bin/env python
"""Import-check the Streamlit dashboard dependencies without opening a browser."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

required = ["streamlit", "psycopg", "sqlalchemy", "plotly", "pandas"]
missing: list[str] = []
for name in required:
    if importlib.util.find_spec(name) is None:
        missing.append(name)

if missing:
    print("missing dashboard dependencies:", ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)

# Important: this checks the package-level import that previously caused
# startup failures through eager resources imports.
from resources.utils import detect_execution_environment

env = detect_execution_environment()
print({"ok": True, "platform": env.get("platform_system"), "is_wsl": env.get("is_wsl")})
