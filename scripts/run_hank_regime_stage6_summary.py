from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.stage6_summary import run_stage6_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final stage-6 summary artifacts and text blocks.")
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_learning_stage6_summary",
        help="Directory for stage-6 summary outputs.",
    )
    parser.add_argument(
        "--architecture-dir",
        default="outputs/hank_regime_learning_stage6_architecture_ablation",
        help="Completed architecture-ablation directory.",
    )
    parser.add_argument(
        "--misspecification-dir",
        default="outputs/hank_regime_learning_stage6_misspecification_map",
        help="Completed misspecification-map directory.",
    )
    parser.add_argument(
        "--environment-shift-dir",
        default="outputs/hank_regime_learning_stage6_environment_shift",
        help="Completed environment-shift directory.",
    )
    args = parser.parse_args()
    run_stage6_summary(
        output_dir=args.output_dir,
        architecture_dir=args.architecture_dir,
        misspecification_dir=args.misspecification_dir,
        environment_shift_dir=args.environment_shift_dir,
    )


if __name__ == "__main__":
    main()
