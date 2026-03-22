from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hidden_state_baseline import run_stage3_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stage-3 hidden-state baseline pipeline.")
    parser.add_argument("--output-dir", default="outputs/stage3", help="Directory for generated artifacts.")
    parser.add_argument("--periods", type=int, default=240, help="Number of periods after burn-in.")
    parser.add_argument("--burn-in", type=int, default=80, help="Burn-in periods for hidden-state simulation.")
    parser.add_argument("--seed", type=int, default=202, help="Random seed for simulation and observations.")
    parser.add_argument(
        "--mc-runs",
        type=int,
        default=25,
        help="Number of Monte Carlo runs for multi-seed filtering diagnostics.",
    )
    args = parser.parse_args()

    results = run_stage3_pipeline(
        output_dir=args.output_dir,
        periods=args.periods,
        burn_in=args.burn_in,
        seed=args.seed,
        monte_carlo_runs=args.mc_runs,
    )
    print(json.dumps(results["filter_diagnostics"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
