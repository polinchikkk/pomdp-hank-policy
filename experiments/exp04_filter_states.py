from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import build_filtered_states


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter HANK/SSJ states from noisy observations.")
    parser.add_argument("--observables-csv", default="outputs/ssj/hank_observables.csv")
    parser.add_argument("--observations-csv", default="outputs/ssj/hank_observations.csv")
    parser.add_argument("--observations-spec", default="outputs/ssj/hank_observations_spec.json")
    parser.add_argument("--output-dir", default="outputs/ssj")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    filtered = build_filtered_states(
        observables_csv=Path(args.observables_csv),
        observations_csv=Path(args.observations_csv),
        observations_spec_json=Path(args.observations_spec),
        output_dir=output_dir,
    )
    print(f"Wrote {output_dir / 'filtered_states.csv'}")
    print(f"Rows: {len(filtered)}")


if __name__ == "__main__":
    main()
