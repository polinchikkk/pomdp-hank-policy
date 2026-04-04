from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.validation import run_deep_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deep validation for tuned regime-learning results.")
    parser.add_argument(
        "--base-dir",
        default="outputs/hank_regime_learning_stage6_validation_suite/oos_seeds",
        help="Base directory with out-of-sample tuned candidate results.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_learning_stage6_deep_validation",
        help="Directory for deep-validation artifacts.",
    )
    args = parser.parse_args()
    run_deep_validation(base_dir=args.base_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
