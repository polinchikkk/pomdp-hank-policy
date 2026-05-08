from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import SSJArtifactSpec, export_long_jacobian_to_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HANK/SSJ Jacobian artifacts for the information experiment.")
    parser.add_argument("--hank-core-dir", default="outputs/hank_core")
    parser.add_argument("--output-dir", default="outputs/ssj")
    parser.add_argument("--horizon", type=int, default=60)
    args = parser.parse_args()

    hank_core_dir = Path(args.hank_core_dir)
    jacobian_csv = hank_core_dir / "jacobian_summary.csv"
    if not jacobian_csv.exists():
        raise FileNotFoundError(
            f"Missing {jacobian_csv}. Run `python3 scripts/run_hank.py --output-dir {hank_core_dir}` first."
        )

    output_dir = Path(args.output_dir)
    export_long_jacobian_to_npz(
        jacobian_csv=jacobian_csv,
        output_path=output_dir / "jacobians.npz",
        spec=SSJArtifactSpec(
            source=str(jacobian_csv),
            horizon=args.horizon,
            input_name="monetary_policy_shock",
            note=(
                "Current artifact exports closed-loop HANK responses to a monetary policy shock. "
                "The next HANK/SSJ milestone is to add exogenous interest-rate-path and income-risk Jacobians."
            ),
        ),
    )
    print(f"Wrote {output_dir / 'jacobians.npz'}")


if __name__ == "__main__":
    main()
