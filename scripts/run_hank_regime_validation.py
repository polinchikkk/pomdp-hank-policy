from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.tuning import run_best_candidate_validation_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run validation suite for the tuned raw-observation regime-learning candidate."
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args()
    kwargs = {}
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir
    run_best_candidate_validation_suite(**kwargs)


if __name__ == "__main__":
    main()
