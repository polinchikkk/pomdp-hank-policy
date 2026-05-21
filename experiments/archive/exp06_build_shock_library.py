from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj.shock_library import build_shock_response_library


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HANK shock response library for SSJ information experiments.")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic")
    parser.add_argument("--shock-size", type=float, default=0.001)
    parser.add_argument("--shocks", default="rstar,Z,G")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    shocks = tuple(value.strip() for value in args.shocks.split(",") if value.strip())
    frame = build_shock_response_library(output_dir=output_dir, shock_size=args.shock_size, shocks=shocks)
    print(f"Wrote {output_dir / 'shock_response_library.csv'}")
    print(f"Rows: {len(frame)}")


if __name__ == "__main__":
    main()
