from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "experiments" / "exp30_closed_loop_distributional_ssj.py"), run_name="__main__")
