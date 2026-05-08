from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import build_noisy_observations


def main() -> None:
    parser = argparse.ArgumentParser(description="Build noisy observations from HANK/SSJ observables.")
    parser.add_argument("--observables-csv", default="outputs/ssj/hank_observables.csv")
    parser.add_argument("--output-dir", default="outputs/ssj")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--aggregate-noise-scale", type=float, default=None)
    parser.add_argument("--distribution-noise-scale", type=float, default=None)
    parser.add_argument("--noise-reference-csv", default=None)
    parser.add_argument("--seed-start", type=int, default=900)
    parser.add_argument("--num-seeds", type=int, default=50)
    args = parser.parse_args()

    seeds = tuple(range(args.seed_start, args.seed_start + args.num_seeds))
    output_dir = Path(args.output_dir)
    observations = build_noisy_observations(
        observables_csv=Path(args.observables_csv),
        output_dir=output_dir,
        seeds=seeds,
        noise_scale=args.noise_scale,
        aggregate_noise_scale=args.aggregate_noise_scale,
        distribution_noise_scale=args.distribution_noise_scale,
        noise_reference_csv=Path(args.noise_reference_csv) if args.noise_reference_csv else None,
    )
    print(f"Wrote {output_dir / 'hank_observations.csv'}")
    print(f"Rows: {len(observations)}")


if __name__ == "__main__":
    main()
