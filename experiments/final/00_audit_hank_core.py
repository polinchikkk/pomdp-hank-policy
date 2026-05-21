from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "experiments" / "archive" / "exp00_hank_core_audit.py"), run_name="__main__")
