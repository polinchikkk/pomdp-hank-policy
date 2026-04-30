from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.information_state_design import run_information_state_design
from hank_regime_learning_baseline.scenario_catalog import information_state_design_scenario_names


def _seed_range(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(int(start), int(start) + int(count)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск новой основной сетки: дизайн информационного состояния для правила ставки."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/information_state_design_main",
        help="Папка для результатов.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        default=None,
        help="Сценарий. Флаг можно повторять. По умолчанию берутся четыре основных сценария.",
    )
    parser.add_argument("--validation-start", type=int, default=500)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--test-start", type=int, default=900)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_information_state_design(
        output_dir=args.output_dir,
        scenario_names=tuple(args.scenarios) if args.scenarios else information_state_design_scenario_names(),
        validation_seeds=_seed_range(args.validation_start, args.validation_count),
        test_seeds=_seed_range(args.test_start, args.test_count),
        horizon=int(args.horizon),
        max_rounds=int(args.max_rounds),
        noise_scale_multiplier=float(args.noise_scale),
    )


if __name__ == "__main__":
    main()
