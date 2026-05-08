from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import build_information_state_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build information-state inputs from HANK/SSJ observations.")
    parser.add_argument("--observables-csv", default="outputs/ssj/hank_observables.csv")
    parser.add_argument("--observations-csv", default="outputs/ssj/hank_observations.csv")
    parser.add_argument("--filtered-states-csv", default="outputs/ssj/filtered_states.csv")
    parser.add_argument("--output-dir", default="outputs/ssj")
    args = parser.parse_args()

    filtered_path = Path(args.filtered_states_csv)
    output_dir = Path(args.output_dir)
    frame = build_information_state_inputs(
        observables_csv=Path(args.observables_csv),
        observations_csv=Path(args.observations_csv),
        filtered_states_csv=filtered_path if filtered_path.exists() else None,
        output_dir=output_dir,
    )
    print(f"Wrote {output_dir / 'information_state_inputs_long.csv'}")
    print(f"Rows: {len(frame)}")


if __name__ == "__main__":
    main()
