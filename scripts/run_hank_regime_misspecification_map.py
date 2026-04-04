from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.misspecification_map import run_misspecification_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage-6 architectural misspecification map in regime-switching HANK.")
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_learning_stage6_misspecification_map",
        help="Directory for misspecification-map artifacts.",
    )
    parser.add_argument(
        "--architecture-dir",
        default="outputs/hank_regime_learning_stage6_architecture_ablation",
        help="Directory with completed architecture-ablation artifacts used to select the best learned policy by scenario.",
    )
    args = parser.parse_args()
    run_misspecification_map(
        output_dir=args.output_dir,
        architecture_dir=args.architecture_dir,
    )


if __name__ == "__main__":
    main()
