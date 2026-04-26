from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.policy_extensions import run_ppo_seed_robustness_check


def _seed_range(start: int, count: int) -> tuple[int, ...]:
    return tuple(range(int(start), int(start) + int(count)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Проверка устойчивости отрицательного результата для PPO."
    )
    parser.add_argument(
        "--base-input-dir",
        default="outputs/hank_regime_learning_stage6_policy_extensions",
        help="Папка с основным прогоном этапа 6, из которой берутся линейные правила для сравнения.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_learning_stage6_ppo_seed_robustness",
        help="Папка для результатов проверки устойчивости.",
    )
    parser.add_argument("--validation-start", type=int, default=500)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--test-start", type=int, default=900)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument(
        "--ppo-training-seed",
        action="append",
        dest="ppo_training_seeds",
        type=int,
        help="Номер запуска PPO. Флаг можно повторять несколько раз.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Сценарий для проверки. По умолчанию берутся все четыре основных сценария.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ppo_seed_robustness_check(
        base_input_dir=args.base_input_dir,
        output_dir=args.output_dir,
        ppo_training_seeds=tuple(args.ppo_training_seeds) if args.ppo_training_seeds else (11, 22, 33),
        validation_seeds=_seed_range(args.validation_start, args.validation_count),
        test_seeds=_seed_range(args.test_start, args.test_count),
        scenario_names=tuple(args.scenarios) if args.scenarios else (
            "macro_core_moderate_gap",
            "macro_core_strong_gap",
            "thin_information_moderate_gap",
            "thin_information_strong_gap",
        ),
    )


if __name__ == "__main__":
    main()
