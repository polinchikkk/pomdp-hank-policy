from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj.shock_library import generate_stochastic_hank_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stochastic HANK/SSJ paths from the shock response library.")
    parser.add_argument("--shock-library", default="outputs/ssj/stochastic/shock_response_library.csv")
    parser.add_argument("--steady-values", default="outputs/ssj/stochastic/steady_distributional_values.json")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--num-trajectories", type=int, default=30)
    args = parser.parse_args()

    seeds = tuple(range(args.seed_start, args.seed_start + args.num_trajectories))
    output_dir = Path(args.output_dir)
    frame = generate_stochastic_hank_paths(
        shock_library_csv=Path(args.shock_library),
        steady_distributional_values_json=Path(args.steady_values),
        output_dir=output_dir,
        trajectory_seeds=seeds,
    )
    print(f"Wrote {output_dir / 'hank_observables.csv'}")
    print(f"Rows: {len(frame)}")


if __name__ == "__main__":
    main()
