from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import build_joint_kalman_filtered_states


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the joint Kalman filter for HANK/SSJ information states.")
    parser.add_argument("--observables-csv", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--observations-csv", default="outputs/ssj/stochastic/hank_observations.csv")
    parser.add_argument("--observations-spec", default="outputs/ssj/stochastic/hank_observations_spec.json")
    parser.add_argument("--scalar-filtered-states", default="outputs/ssj/stochastic/filtered_states.csv")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/state_space")
    parser.add_argument("--transition-shrinkage", type=float, default=0.15)
    parser.add_argument("--covariance-floor", type=float, default=1e-12)
    parser.add_argument("--max-spectral-radius", type=float, default=0.98)
    args = parser.parse_args()

    scalar_path = Path(args.scalar_filtered_states)
    frame = build_joint_kalman_filtered_states(
        observables_csv=Path(args.observables_csv),
        observations_csv=Path(args.observations_csv),
        observations_spec_json=Path(args.observations_spec),
        output_dir=Path(args.output_dir),
        scalar_filtered_states_csv=scalar_path if scalar_path.exists() else None,
        transition_shrinkage=args.transition_shrinkage,
        covariance_floor=args.covariance_floor,
        max_spectral_radius=args.max_spectral_radius,
    )
    output_dir = Path(args.output_dir)
    print(f"Wrote {output_dir / 'kalman_filtered_states.csv'}")
    print(f"Wrote {output_dir / 'filter_quality_joint.csv'}")
    print(f"Wrote {output_dir / 'posterior_covariances.npz'}")
    print(f"Rows: {len(frame)}")


if __name__ == "__main__":
    main()
