from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.environment_shift import run_environment_shift


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-6 environment-shift transfer evaluation in regime-switching HANK.")
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_learning_stage6_environment_shift",
        help="Directory for environment-shift artifacts.",
    )
    parser.add_argument(
        "--architecture-dir",
        default="outputs/hank_regime_learning_stage6_architecture_ablation",
        help="Directory with completed architecture-ablation outputs used to select the best learned architecture by scenario.",
    )
    parser.add_argument(
        "--retuned-csv",
        default="outputs/hank_regime_learning_stage6_deep_validation/retuned_classical/retuned_classical_best.csv",
        help="CSV with baseline-tuned simple-rule coefficients.",
    )
    args = parser.parse_args()
    run_environment_shift(
        output_dir=args.output_dir,
        architecture_dir=args.architecture_dir,
        retuned_csv=args.retuned_csv,
    )


if __name__ == "__main__":
    main()
