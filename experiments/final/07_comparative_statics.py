from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
runpy.run_path(
    str(ROOT / "experiments" / "archive" / "exp40_distributional_value_phase_diagram.py"),
    run_name="__main__",
)
