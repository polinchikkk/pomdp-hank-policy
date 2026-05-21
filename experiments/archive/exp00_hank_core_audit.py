from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank.audit import DEFAULT_SHOCK_TYPES, load_calibration_from_core, write_hank_core_audit  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit HANK core steady-state and transition artifacts.")
    parser.add_argument("--hank-core-dir", default="outputs/hank_core")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--shock-types", default=",".join(DEFAULT_SHOCK_TYPES))
    parser.add_argument("--transition-horizon", type=int, default=None)
    parser.add_argument("--residual-tolerance", type=float, default=1e-6)
    parser.add_argument("--skip-transition-solves", action="store_true")
    args = parser.parse_args()

    core_dir = Path(args.hank_core_dir)
    output_dir = Path(args.output_dir) if args.output_dir else core_dir / "audit"
    config = load_calibration_from_core(core_dir)
    shock_types = tuple(part.strip() for part in args.shock_types.split(",") if part.strip())
    steady, transitions = write_hank_core_audit(
        core_dir=core_dir,
        output_dir=output_dir,
        config=config,
        shock_types=shock_types,
        transition_horizon=args.transition_horizon,
        skip_transition_solves=bool(args.skip_transition_solves),
        residual_tolerance=float(args.residual_tolerance),
    )
    print(f"Wrote {output_dir / 'steady_state_audit.json'}")
    print(f"Wrote {output_dir / 'transition_audit.csv'}")
    print(f"Wrote {output_dir / 'report_hank_core_audit.md'}")
    failed = transitions[~transitions["residual_converged"].astype(bool)] if "residual_converged" in transitions else transitions
    if not failed.empty and not bool(args.skip_transition_solves):
        names = ", ".join(str(value) for value in failed["shock_type"].tolist())
        raise RuntimeError(f"HANK transition audit failed residual convergence for: {names}")
    if steady["distribution"]["mass_error"] > 1e-8 or steady["distribution"]["negative_entries"] > 0:
        raise RuntimeError("HANK steady-state distribution audit failed mass/nonnegativity checks.")


if __name__ == "__main__":
    main()
