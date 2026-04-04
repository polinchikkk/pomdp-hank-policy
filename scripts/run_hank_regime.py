from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime_switching_baseline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run stage 5: regime-switching reduced-state HANK baseline under partial information."
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hank_regime_switching_stage5",
        help="Directory for generated outputs.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional subset of regime-switching scenarios to run.",
    )
    args = parser.parse_args()
    run_pipeline(output_dir=args.output_dir, scenario_names=args.scenarios)


if __name__ == "__main__":
    main()
