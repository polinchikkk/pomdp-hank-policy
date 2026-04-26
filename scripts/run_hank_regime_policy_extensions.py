from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.policy_extensions import run_policy_extension_experiments


def _seed_range(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(int(start), int(start) + int(count)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stage-6 policy extension experiments: optimized linear rule and history-based observable rule."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_learning_stage6_policy_extensions",
        help="Directory for extension outputs.",
    )
    parser.add_argument("--validation-start", type=int, default=500)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--test-start", type=int, default=900)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Scenario to run. Repeat the flag to run multiple scenarios. Defaults to all four core scenarios.",
    )
    parser.add_argument(
        "--run-full-hank-projection",
        action="store_true",
        help="Also project selected reduced-state policy-rate paths through the full HANK transition solver.",
    )
    parser.add_argument(
        "--full-hank-scenario",
        action="append",
        dest="full_hank_scenarios",
        help="Scenario for full-HANK projection. Defaults to thin_information_strong_gap when the projection is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_policy_extension_experiments(
        output_dir=args.output_dir,
        validation_seeds=_seed_range(args.validation_start, args.validation_count),
        test_seeds=_seed_range(args.test_start, args.test_count),
        scenario_names=tuple(args.scenarios) if args.scenarios else (
            "macro_core_moderate_gap",
            "macro_core_strong_gap",
            "thin_information_moderate_gap",
            "thin_information_strong_gap",
        ),
        run_full_hank_projection=bool(args.run_full_hank_projection),
        full_hank_scenarios=tuple(args.full_hank_scenarios) if args.full_hank_scenarios else ("thin_information_strong_gap",),
    )


if __name__ == "__main__":
    main()
