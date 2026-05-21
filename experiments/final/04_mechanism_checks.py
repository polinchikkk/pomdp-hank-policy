from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

for script in (
    "exp22_mechanism_optimal_rate_projection.py",
    "exp35_mechanism_residualized_crossfit.py",
    "exp40_transmission_state_value.py",
):
    runpy.run_path(str(ROOT / "experiments" / "archive" / script), run_name="__main__")
