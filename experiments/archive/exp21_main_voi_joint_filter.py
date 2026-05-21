from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import build_information_state_inputs, build_joint_kalman_filtered_states


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat the main value-of-information experiment with a joint Kalman filter.")
    parser.add_argument("--observables-csv", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--observations-csv", default="outputs/ssj/stochastic/hank_observations.csv")
    parser.add_argument("--observations-spec", default="outputs/ssj/stochastic/hank_observations_spec.json")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--scalar-filtered-states", default="outputs/ssj/stochastic/filtered_states.csv")
    parser.add_argument("--scalar-main-voi-dir", default="outputs/ssj/stochastic/main_voi")
    parser.add_argument("--state-space-dir", default="outputs/ssj/stochastic/state_space")
    parser.add_argument("--information-inputs-dir", default="outputs/ssj/stochastic/state_space/information_inputs")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/main_voi_joint_filter")
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--num-candidates", type=int, default=220)
    parser.add_argument("--candidate-seed", type=int, default=4027)
    parser.add_argument("--optimization-modes", default="random_candidates,grid_random,continuous")
    parser.add_argument("--primary-optimization-mode", default="continuous")
    parser.add_argument("--continuous-methods", default="L-BFGS-B")
    parser.add_argument("--num-starts", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--intercept-bound", type=float, default=0.01)
    parser.add_argument("--standardized-coefficient-bound", type=float, default=0.05)
    parser.add_argument("--policy-optimization-config", default="config/final_policy_optimization.yaml")
    args = parser.parse_args()

    state_space_dir = Path(args.state_space_dir)
    filtered = build_joint_kalman_filtered_states(
        observables_csv=Path(args.observables_csv),
        observations_csv=Path(args.observations_csv),
        observations_spec_json=Path(args.observations_spec),
        output_dir=state_space_dir,
        scalar_filtered_states_csv=Path(args.scalar_filtered_states) if Path(args.scalar_filtered_states).exists() else None,
    )
    print(f"Joint filter rows: {len(filtered)}")

    information_inputs_dir = Path(args.information_inputs_dir)
    inputs = build_information_state_inputs(
        observables_csv=Path(args.observables_csv),
        observations_csv=Path(args.observations_csv),
        filtered_states_csv=state_space_dir / "kalman_filtered_states.csv",
        output_dir=information_inputs_dir,
    )
    print(f"Information input rows: {len(inputs)}")

    output_dir = Path(args.output_dir)
    subprocess.run(
        [
            sys.executable,
            "experiments/archive/exp08_main_voi.py",
            "--information-inputs",
            str(information_inputs_dir / "information_state_inputs_long.csv"),
            "--hank-observables",
            args.observables_csv,
            "--jacobians",
            args.jacobians,
            "--output-dir",
            str(output_dir),
            "--validation-seeds",
            args.validation_seeds,
            "--test-seeds",
            args.test_seeds,
            "--num-candidates",
            str(args.num_candidates),
            "--candidate-seed",
            str(args.candidate_seed),
            "--optimization-modes",
            args.optimization_modes,
            "--primary-optimization-mode",
            args.primary_optimization_mode,
            "--continuous-methods",
            args.continuous_methods,
            "--num-starts",
            str(args.num_starts),
            "--maxiter",
            str(args.maxiter),
            "--intercept-bound",
            str(args.intercept_bound),
            "--standardized-coefficient-bound",
            str(args.standardized_coefficient_bound),
            "--policy-optimization-config",
            args.policy_optimization_config,
        ],
        check=True,
        cwd=ROOT,
    )
    _write_filter_version_comparison(
        joint_main_voi_dir=output_dir,
        scalar_main_voi_dir=Path(args.scalar_main_voi_dir),
        output_dir=output_dir,
    )
    print(f"Wrote {output_dir / 'filter_version_comparison.csv'}")


def _write_filter_version_comparison(
    *,
    joint_main_voi_dir: Path,
    scalar_main_voi_dir: Path,
    output_dir: Path,
) -> None:
    rows: list[dict[str, object]] = []
    if scalar_main_voi_dir.exists():
        rows.extend(_extract_version_rows("scalar_filter", "Скалярный фильтр", scalar_main_voi_dir))
    rows.extend(_extract_version_rows("joint_kalman_filter", "Совместный фильтр Калмана", joint_main_voi_dir))
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "filter_version_comparison.csv", index=False)
    comparison.to_latex(output_dir / "table_filter_version_comparison.tex", index=False, escape=False)


def _extract_version_rows(version: str, version_ru: str, main_voi_dir: Path) -> list[dict[str, object]]:
    summary_path = main_voi_dir / "main_voi_summary.csv"
    if not summary_path.exists():
        return []
    summary = pd.read_csv(summary_path)
    states = {
        "aggregate_only": "Шумные текущие агрегаты",
        "filtered_aggregates": "Фильтрованные агрегаты",
        "filtered_distribution": "Фильтрованные распределительные показатели",
        "full_information": "Полная информация",
    }
    rows: list[dict[str, object]] = []
    for _, row in summary[summary["scenario"] == "all"].iterrows():
        state = row["information_state"]
        if state not in states:
            continue
        rows.append(
            {
                "filter_version": version,
                "filter_version_ru": version_ru,
                "information_state": state,
                "information_state_ru": states[state],
                "mean_loss": row["mean_loss"],
                "ci_low": row["ci_low"],
                "ci_high": row["ci_high"],
                "num_trajectories": row["num_trajectories"],
            }
        )
    return rows


if __name__ == "__main__":
    main()
