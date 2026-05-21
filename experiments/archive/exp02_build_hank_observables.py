from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import build_hank_observable_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HANK/SSJ observables for information experiments.")
    parser.add_argument("--hank-core-dir", default="outputs/hank_core")
    parser.add_argument("--output-dir", default="outputs/ssj")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    frame = build_hank_observable_panel(
        hank_core_dir=Path(args.hank_core_dir),
        output_dir=output_dir,
    )
    print(f"Wrote {output_dir / 'hank_observables.csv'}")
    print(f"Rows: {len(frame)}")


if __name__ == "__main__":
    main()
