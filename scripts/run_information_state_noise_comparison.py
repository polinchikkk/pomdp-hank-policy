from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hank_regime_learning_baseline.information_state_design import run_information_state_design
from hank_regime_learning_baseline.scenario_catalog import information_state_design_scenario_names


def _scale_dir_name(scale: float) -> str:
    return f"noise_{scale:g}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнить правила информационного состояния при разных уровнях шума наблюдений."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/information_state_design_noise_comparison",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--validation-start", type=int, default=500)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--test-start", type=int, default=900)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--max-rounds", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    validation_seeds = tuple(range(args.validation_start, args.validation_start + args.validation_count))
    test_seeds = tuple(range(args.test_start, args.test_start + args.test_count))
    scenario_names = tuple(args.scenario) if args.scenario else information_state_design_scenario_names()

    level_frames = []
    pairwise_frames = []
    for scale in args.noise_scale:
        result = run_information_state_design(
            output_dir=str(root / _scale_dir_name(scale)),
            scenario_names=scenario_names,
            validation_seeds=validation_seeds,
            test_seeds=test_seeds,
            horizon=args.horizon,
            max_rounds=args.max_rounds,
            noise_scale_multiplier=float(scale),
        )
        levels = result["information_state_levels"].copy()
        levels.insert(0, "noise_scale", float(scale))
        level_frames.append(levels)
        pairwise = result["information_state_pairwise"].copy()
        pairwise.insert(0, "noise_scale", float(scale))
        pairwise_frames.append(pairwise)

    pd.concat(level_frames, ignore_index=True).to_csv(root / "noise_comparison_levels.csv", index=False)
    pd.concat(pairwise_frames, ignore_index=True).to_csv(root / "noise_comparison_pairwise.csv", index=False)


if __name__ == "__main__":
    main()
