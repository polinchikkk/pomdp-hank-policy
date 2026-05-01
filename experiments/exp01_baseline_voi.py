from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state_space import default_information_states


def experiment_plan() -> list[dict[str, str]]:
    return [
        {
            "information_state": spec.name,
            "label": spec.label,
            "role": spec.role,
        }
        for spec in default_information_states()
    ]


if __name__ == "__main__":
    for row in experiment_plan():
        print(f"{row['information_state']}: {row['label']} — {row['role']}")
