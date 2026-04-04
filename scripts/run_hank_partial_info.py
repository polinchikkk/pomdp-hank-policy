from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_partial_info_baseline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run partial-observability HANK baseline: reduced hidden state, Kalman filtering, and classical filter-plus-rule policy."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_partial_info_stage3",
        help="Directory for generated outputs.",
    )
    args = parser.parse_args()
    run_pipeline(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
