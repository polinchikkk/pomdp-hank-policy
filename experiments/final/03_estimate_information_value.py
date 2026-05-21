from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(str(ROOT / "experiments" / "exp29_large_sample_joint_filter.py"), run_name="__main__")
