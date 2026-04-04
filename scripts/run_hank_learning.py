from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_learning_policy_baseline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run stage 4: learning-based policy layer for partial-information full HANK."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_learning_stage4",
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional subset of stage-4 variants to run.",
    )
    args = parser.parse_args()
    run_pipeline(output_dir=args.output_dir, variant_names=args.variants)


if __name__ == "__main__":
    main()
