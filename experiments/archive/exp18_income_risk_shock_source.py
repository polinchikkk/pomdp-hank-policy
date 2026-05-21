from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj.shock_library import EXTENDED_SHOCKS, build_shock_response_library, generate_stochastic_hank_paths


@dataclass(frozen=True)
class IncomeRiskShockSourceSpec:
    shocks: tuple[str, ...]
    num_hank_paths: int
    trajectory_seed_start: int
    observation_seed_start: int
    num_observation_seeds: int
    shock_standard_deviations: dict[str, float]
    shock_persistence: dict[str, float]
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the HANK/SSJ experiment with an explicit income-risk shock.")
    parser.add_argument("--baseline-main-voi", default="outputs/ssj/stochastic/main_voi")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/income_risk_shock_source")
    parser.add_argument("--shock-size", type=float, default=0.001)
    parser.add_argument("--num-trajectories", type=int, default=50)
    parser.add_argument("--trajectory-seed-start", type=int, default=1000)
    parser.add_argument("--observation-seed-start", type=int, default=900)
    parser.add_argument("--num-observation-seeds", type=int, default=12)
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--num-candidates", type=int, default=90)
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shocks = EXTENDED_SHOCKS
    shock_standard_deviations = {"rstar": 0.0007, "Z": 0.0007, "G": 0.0007, "sigma_z": 0.0005}
    shock_persistence = {"rstar": 0.65, "Z": 0.75, "G": 0.55, "sigma_z": 0.70}

    if not args.skip_runs:
        build_shock_response_library(
            output_dir=output_dir,
            shock_size=args.shock_size,
            shocks=shocks,
        )
        seeds = tuple(range(args.trajectory_seed_start, args.trajectory_seed_start + args.num_trajectories))
        generate_stochastic_hank_paths(
            shock_library_csv=output_dir / "shock_response_library.csv",
            steady_distributional_values_json=output_dir / "steady_distributional_values.json",
            output_dir=output_dir,
            trajectory_seeds=seeds,
            shock_standard_deviations=shock_standard_deviations,
            shock_persistence=shock_persistence,
        )
        _run_pipeline(
            observables_csv=output_dir / "hank_observables.csv",
            jacobians=Path(args.jacobians),
            output_dir=output_dir,
            observation_seed_start=args.observation_seed_start,
            num_observation_seeds=args.num_observation_seeds,
            validation_seeds=args.validation_seeds,
            test_seeds=args.test_seeds,
            num_candidates=args.num_candidates,
        )

    summary = pd.DataFrame(
        [
            _extract_case("baseline_shocks", "Базовые неполитические шоки", Path(args.baseline_main_voi)),
            _extract_case("with_income_risk_shock", "Базовые шоки + шок доходного риска", output_dir / "main_voi"),
        ]
    )
    summary.to_csv(output_dir / "income_risk_shock_source_summary.csv", index=False)
    summary.to_latex(output_dir / "table_income_risk_shock_source.tex", index=False, escape=False, float_format="%.6g")
    _write_report(summary, output_dir / "report_income_risk_shock_source.md")

    spec = IncomeRiskShockSourceSpec(
        shocks=shocks,
        num_hank_paths=args.num_trajectories,
        trajectory_seed_start=args.trajectory_seed_start,
        observation_seed_start=args.observation_seed_start,
        num_observation_seeds=args.num_observation_seeds,
        shock_standard_deviations=shock_standard_deviations,
        shock_persistence=shock_persistence,
        note=(
            "Проверка добавляет временный шок sigma_z в HANK transition solver. "
            "Это отдельный источник распределительной динамики, а не ручное масштабирование уже готовых рядов."
        ),
    )
    (output_dir / "income_risk_shock_source_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'income_risk_shock_source_summary.csv'}")
    print(f"Wrote {output_dir / 'report_income_risk_shock_source.md'}")


def _run_pipeline(
    *,
    observables_csv: Path,
    jacobians: Path,
    output_dir: Path,
    observation_seed_start: int,
    num_observation_seeds: int,
    validation_seeds: str,
    test_seeds: str,
    num_candidates: int,
) -> None:
    _run(
        [
            "experiments/archive/exp03_build_observations.py",
            "--observables-csv",
            str(observables_csv),
            "--output-dir",
            str(output_dir),
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
            str(observables_csv),
            "--observations-csv",
            str(output_dir / "hank_observations.csv"),
            "--observations-spec",
            str(output_dir / "hank_observations_spec.json"),
            "--output-dir",
            str(output_dir),
        ]
    )
    _run(
        [
            "experiments/archive/exp05_build_information_inputs.py",
            "--observables-csv",
            str(observables_csv),
            "--observations-csv",
            str(output_dir / "hank_observations.csv"),
            "--filtered-states-csv",
            str(output_dir / "filtered_states.csv"),
            "--output-dir",
            str(output_dir),
        ]
    )
    _run(
        [
            "experiments/archive/exp08_main_voi.py",
            "--information-inputs",
            str(output_dir / "information_state_inputs_long.csv"),
            "--hank-observables",
            str(observables_csv),
            "--jacobians",
            str(jacobians),
            "--output-dir",
            str(output_dir / "main_voi"),
            "--validation-seeds",
            validation_seeds,
            "--test-seeds",
            test_seeds,
            "--num-candidates",
            str(num_candidates),
        ]
    )


def _extract_case(case: str, description: str, main_voi_dir: Path) -> dict[str, object]:
    summary = pd.read_csv(main_voi_dir / "main_voi_summary.csv")
    pairwise = pd.read_csv(main_voi_dir / "pairwise_value_of_information.csv")
    overall = summary[summary["scenario"] == "all"].set_index("information_state")
    comparison = pairwise[
        (pairwise["scenario"] == "all")
        & (pairwise["comparison"] == "filtered_distribution_minus_filtered_aggregates")
    ].iloc[0]
    return {
        "case": case,
        "description": description,
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


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Шок доходного риска как отдельный HANK/SSJ-источник",
        "",
        "Проверка добавляет шок `sigma_z` в библиотеку HANK transition responses.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['description']}: MVOI {row['mvoi_dist']:.6g}, "
            f"интервал для разности [{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
            f"доля выигрышей {row['win_rate']:.3g}, парных оценок {int(row['num_paired_evaluations'])}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
