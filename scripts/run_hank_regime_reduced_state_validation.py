from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.reduced_state_validation import run_reduced_state_validation


def _seed_range(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(int(start), int(start) + int(count)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the reduced state representation used as the stage-6 policy interface."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_learning_stage6_reduced_state_validation",
        help="Directory for validation outputs.",
    )
    parser.add_argument(
        "--policy-extension-dir",
        default="outputs/hank_regime_learning_stage6_policy_extensions",
        help="Directory containing stage-6 policy extension outputs.",
    )
    parser.add_argument("--train-start", type=int, default=700)
    parser.add_argument("--train-count", type=int, default=30)
    parser.add_argument("--test-start", type=int, default=900)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="Scenario to validate. Repeat to run multiple scenarios. Defaults to all four core scenarios.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        action="append",
        dest="horizons",
        default=None,
        help="Forecast horizon. Repeat to include multiple horizons. Defaults to 1, 4, and 8.",
    )
    parser.add_argument(
        "--run-full-hank-projection",
        action="store_true",
        help="Refresh full-HANK projection for the selected scenarios before ranking validation.",
    )
    parser.add_argument(
        "--common-scale-seed-count",
        type=int,
        default=20,
        help="Number of held-out test trajectories used in the expensive common-scale HANK projection block.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_reduced_state_validation(
        output_dir=args.output_dir,
        policy_extension_dir=args.policy_extension_dir,
        scenario_names=tuple(args.scenarios)
        if args.scenarios
        else (
            "macro_core_moderate_gap",
            "macro_core_strong_gap",
            "thin_information_moderate_gap",
            "thin_information_strong_gap",
        ),
        train_seeds=_seed_range(args.train_start, args.train_count),
        test_seeds=_seed_range(args.test_start, args.test_count),
        common_scale_projection_seed_count=int(args.common_scale_seed_count),
        forecast_horizons=tuple(args.horizons) if args.horizons else (1, 4, 8),
        run_full_hank_projection=bool(args.run_full_hank_projection),
    )


if __name__ == "__main__":
    main()
