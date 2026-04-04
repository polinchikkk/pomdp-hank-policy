from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_regime_learning_baseline.tuning import run_universal_rawobs_misspecified_tuning


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run universal PPO/action tuning for raw-observation RL in regime-switching HANK."
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    args = parser.parse_args()
    kwargs = {}
    if args.output_dir is not None:
        kwargs["output_dir"] = args.output_dir
    run_universal_rawobs_misspecified_tuning(**kwargs)


if __name__ == "__main__":
    main()
