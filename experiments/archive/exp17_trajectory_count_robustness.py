from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrajectoryCountCase:
    num_hank_paths: int
    output_dir: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run robustness check for the number of HANK/SSJ trajectories.")
    parser.add_argument("--shock-library", default="outputs/ssj/stochastic/shock_response_library.csv")
    parser.add_argument("--steady-values", default="outputs/ssj/stochastic/steady_distributional_values.json")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--baseline-main-voi", default="outputs/ssj/stochastic/main_voi")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/trajectory_count_robustness")
    parser.add_argument("--path-counts", default="100")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--observation-seed-start", type=int, default=900)
    parser.add_argument("--num-observation-seeds", type=int, default=12)
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--num-candidates", type=int, default=120)
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        TrajectoryCountCase(num_hank_paths=int(value), output_dir=output_dir / f"paths_{int(value)}")
        for value in args.path_counts.split(",")
        if value.strip()
    ]

    if not args.skip_runs:
        for case in cases:
            _run_case(
                case=case,
                shock_library=Path(args.shock_library),
                steady_values=Path(args.steady_values),
                jacobians=Path(args.jacobians),
                seed_start=args.seed_start,
                observation_seed_start=args.observation_seed_start,
                num_observation_seeds=args.num_observation_seeds,
                validation_seeds=args.validation_seeds,
                test_seeds=args.test_seeds,
                num_candidates=args.num_candidates,
            )

    summary = _summarize(
        baseline_main_voi=Path(args.baseline_main_voi),
        cases=cases,
        baseline_num_paths=_infer_baseline_num_paths(Path(args.baseline_main_voi)),
    )
    summary.to_csv(output_dir / "trajectory_count_robustness_summary.csv", index=False)
    summary.to_latex(output_dir / "table_trajectory_count_robustness.tex", index=False, escape=False, float_format="%.6g")
    _write_report(summary, output_dir / "report_trajectory_count_robustness.md")
    print(f"Wrote {output_dir / 'trajectory_count_robustness_summary.csv'}")
    print(f"Wrote {output_dir / 'report_trajectory_count_robustness.md'}")


def _run_case(
    *,
    case: TrajectoryCountCase,
    shock_library: Path,
    steady_values: Path,
    jacobians: Path,
    seed_start: int,
    observation_seed_start: int,
    num_observation_seeds: int,
    validation_seeds: str,
    test_seeds: str,
    num_candidates: int,
) -> None:
    case.output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "experiments/archive/exp07_generate_stochastic_hank_paths.py",
            "--shock-library",
            str(shock_library),
            "--steady-values",
            str(steady_values),
            "--output-dir",
            str(case.output_dir),
            "--seed-start",
            str(seed_start),
            "--num-trajectories",
            str(case.num_hank_paths),
        ]
    )
    _run(
        [
            "experiments/archive/exp03_build_observations.py",
            "--observables-csv",
            str(case.output_dir / "hank_observables.csv"),
            "--output-dir",
            str(case.output_dir),
            "--seed-start",
            str(observation_seed_start),
            "--num-seeds",
            str(num_observation_seeds),
        ]
    )
    _run(
        [
            "experiments/archive/exp04_filter_states.py",
            "--observables-csv",
            str(case.output_dir / "hank_observables.csv"),
            "--observations-csv",
            str(case.output_dir / "hank_observations.csv"),
            "--observations-spec",
            str(case.output_dir / "hank_observations_spec.json"),
            "--output-dir",
            str(case.output_dir),
        ]
    )
    _run(
        [
            "experiments/archive/exp05_build_information_inputs.py",
            "--observables-csv",
            str(case.output_dir / "hank_observables.csv"),
            "--observations-csv",
            str(case.output_dir / "hank_observations.csv"),
            "--filtered-states-csv",
            str(case.output_dir / "filtered_states.csv"),
            "--output-dir",
            str(case.output_dir),
        ]
    )
    _run(
        [
            "experiments/archive/exp08_main_voi.py",
            "--information-inputs",
            str(case.output_dir / "information_state_inputs_long.csv"),
            "--hank-observables",
            str(case.output_dir / "hank_observables.csv"),
            "--jacobians",
            str(jacobians),
            "--output-dir",
            str(case.output_dir / "main_voi"),
            "--validation-seeds",
            validation_seeds,
            "--test-seeds",
            test_seeds,
            "--num-candidates",
            str(num_candidates),
        ]
    )


def _summarize(
    *,
    baseline_main_voi: Path,
    cases: list[TrajectoryCountCase],
    baseline_num_paths: int,
) -> pd.DataFrame:
    rows = [_extract_case(baseline_num_paths, baseline_main_voi)]
    for case in cases:
        rows.append(_extract_case(case.num_hank_paths, case.output_dir / "main_voi"))
    return pd.DataFrame(rows).sort_values("num_hank_paths").reset_index(drop=True)


def _extract_case(num_hank_paths: int, main_voi_dir: Path) -> dict[str, object]:
    summary = pd.read_csv(main_voi_dir / "main_voi_summary.csv")
    pairwise = pd.read_csv(main_voi_dir / "pairwise_value_of_information.csv")
    overall = summary[summary["scenario"] == "all"].set_index("information_state")
    comparison = pairwise[
        (pairwise["scenario"] == "all")
        & (pairwise["comparison"] == "filtered_distribution_minus_filtered_aggregates")
    ].iloc[0]
    return {
        "num_hank_paths": int(num_hank_paths),
        "num_paired_evaluations": int(comparison["num_trajectories"]),
        "loss_filtered_aggregates": float(overall.loc["filtered_aggregates", "mean_loss"]),
        "loss_filtered_distribution": float(overall.loc["filtered_distribution", "mean_loss"]),
        "mvoi_dist": float(comparison["loss_reduction"]),
        "delta_filtered_distribution_minus_filtered_aggregates": float(comparison["mean_delta"]),
        "ci_low": float(comparison["ci_low"]),
        "ci_high": float(comparison["ci_high"]),
        "win_rate": float(comparison["win_rate"]),
        "tie_rate": float(comparison.get("tie_rate", 0.0)),
        "loss_rate": float(comparison["loss_rate"]),
    }


def _infer_baseline_num_paths(main_voi_dir: Path) -> int:
    losses = pd.read_csv(main_voi_dir / "trajectory_losses.csv", usecols=["scenario"])
    return int(losses["scenario"].nunique())


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Устойчивость к числу HANK/SSJ-траекторий",
        "",
        "Проверка повторяет основной расчёт при большем числе стохастических HANK/SSJ-путей.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {int(row['num_hank_paths'])} HANK/SSJ-путей: MVOI {row['mvoi_dist']:.6g}, "
            f"интервал для разности [{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
            f"доля выигрышей {row['win_rate']:.3g}, парных оценок {int(row['num_paired_evaluations'])}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
