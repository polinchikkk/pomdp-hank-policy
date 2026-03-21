from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rbc_baseline import run_stage1_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the stage-1 RBC baseline pipeline.")
    parser.add_argument("--output-dir", default="outputs/stage1", help="Directory for generated artifacts.")
    parser.add_argument("--periods", type=int, default=240, help="Simulation periods after burn-in.")
    parser.add_argument("--burn-in", type=int, default=80, help="Burn-in periods for stochastic simulation.")
    parser.add_argument("--irf-horizon", type=int, default=24, help="IRF horizon in periods.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stochastic simulation.")
    args = parser.parse_args()

    results = run_stage1_pipeline(
        output_dir=args.output_dir,
        periods=args.periods,
        burn_in=args.burn_in,
        irf_horizon=args.irf_horizon,
        seed=args.seed,
    )
    summary = results["diagnostics_summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
