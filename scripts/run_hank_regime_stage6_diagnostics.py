from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.stage6_diagnostics import run_stage6_diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build diagnostic figures for stage 6 regime-learning results."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_learning_stage6_diagnostics",
        help="Directory for diagnostic artifacts.",
    )
    parser.add_argument(
        "--architecture-dir",
        default="outputs/hank_regime_learning_stage6_architecture_ablation",
        help="Completed architecture-ablation directory.",
    )
    parser.add_argument(
        "--environment-shift-dir",
        default="outputs/hank_regime_learning_stage6_environment_shift",
        help="Completed environment-shift directory.",
    )
    args = parser.parse_args()
    run_stage6_diagnostics(
        output_dir=args.output_dir,
        architecture_dir=args.architecture_dir,
        environment_shift_dir=args.environment_shift_dir,
    )


if __name__ == "__main__":
    main()
