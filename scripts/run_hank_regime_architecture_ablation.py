from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.architecture_ablation import run_architecture_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-6 architecture ablation in regime-switching HANK.")
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_learning_stage6_architecture_ablation",
        help="Directory for architecture ablation artifacts.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        dest="variants",
        default=None,
        help="Optional architecture-ablation variant name to run. May be passed multiple times.",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Re-run variants even if their output files already exist.",
    )
    args = parser.parse_args()
    run_architecture_ablation(
        output_dir=args.output_dir,
        variant_names=args.variants,
        skip_completed=not args.no_skip_completed,
    )


if __name__ == "__main__":
    main()
