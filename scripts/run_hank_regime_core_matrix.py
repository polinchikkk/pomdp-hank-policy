from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline import run_core_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the clean stage-6 monetary-only comparison matrix from existing architecture-ablation outputs."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_learning_stage6_core_matrix",
        help="Directory for the clean stage-6 core-matrix summary outputs.",
    )
    parser.add_argument(
        "--architecture-dir",
        default="outputs/hank_regime_learning_stage6_architecture_ablation",
        help="Directory with architecture-ablation artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_core_matrix(output_dir=args.output_dir, architecture_dir=args.architecture_dir)


if __name__ == "__main__":
    main()
