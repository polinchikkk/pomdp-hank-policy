from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run the HANK core.")
    parser.add_argument("--output-dir", default="outputs/hank_core", help="Directory for generated outputs.")
    args = parser.parse_args()
    run_pipeline(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
