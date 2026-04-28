from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.posterior_information_state import (
    run_posterior_information_state,
)


def _seed_range(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(int(start), int(start) + int(count)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск сравнения правил на разных сводках апостериорной информации."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_learning_stage6_posterior_information_state",
        help="Папка для результатов.",
    )
    parser.add_argument("--validation-start", type=int, default=500)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--test-start", type=int, default=900)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Имя сценария. Флаг можно повторять несколько раз.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_posterior_information_state(
        output_dir=args.output_dir,
        scenario_names=tuple(args.scenarios)
        if args.scenarios
        else (
            "macro_core_moderate_gap",
            "macro_core_strong_gap",
            "thin_information_moderate_gap",
            "thin_information_strong_gap",
        ),
        validation_seeds=_seed_range(args.validation_start, args.validation_count),
        test_seeds=_seed_range(args.test_start, args.test_count),
    )


if __name__ == "__main__":
    main()
