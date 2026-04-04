from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run regime-switching HANK learning-policy baseline.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional subset of regime-learning scenario names.",
    )
    args = parser.parse_args()
    run_pipeline(output_dir=args.output_dir, scenario_names=args.scenarios)


if __name__ == "__main__":
    main()
