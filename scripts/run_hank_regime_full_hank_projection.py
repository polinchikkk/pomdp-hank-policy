from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.policy_extensions import run_full_hank_projection_from_policy_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project selected stage-6 policy-rate paths through the full HANK transition solver."
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/hank_regime_learning_stage6_policy_extensions",
        help="Directory containing policy_paths.csv from the stage-6 policy extension experiments.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to --input-dir.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="Scenario to project. Repeat to project multiple scenarios.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        default=None,
        help="Policy name to project. Repeat to project multiple policies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_full_hank_projection_from_policy_paths(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        scenario_names=tuple(args.scenarios) if args.scenarios else ("thin_information_strong_gap",),
        policy_names=tuple(args.policies)
        if args.policies
        else (
            "classical_filtered_rule",
            "optimized_linear_estimated_state",
            "history_observables_rule",
        ),
    )


if __name__ == "__main__":
    main()
