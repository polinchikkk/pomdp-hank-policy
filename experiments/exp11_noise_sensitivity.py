from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class NoiseCase:
    name: str
    axis: str
    aggregate_noise_scale: float
    distribution_noise_scale: float
    output_dir: Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run noise sensitivity for HANK/SSJ value-of-information experiment.")
    parser.add_argument("--observables-csv", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/noise_sensitivity")
    parser.add_argument("--baseline-main-voi", default="outputs/ssj/stochastic/main_voi")
    parser.add_argument("--scales", default="0.5,2.0")
    parser.add_argument("--seed-start", type=int, default=900)
    parser.add_argument("--num-seeds", type=int, default=12)
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scales = _parse_scales(args.scales)
    cases = _noise_cases(output_dir=output_dir, scales=scales)

    if not args.skip_runs:
        for case in cases:
            _run_case(
                case=case,
                observables_csv=Path(args.observables_csv),
                jacobians=Path(args.jacobians),
                seed_start=args.seed_start,
                num_seeds=args.num_seeds,
                validation_seeds=args.validation_seeds,
                test_seeds=args.test_seeds,
                num_candidates=args.num_candidates,
            )

    summary = _summarize(
        cases=cases,
        baseline_main_voi=Path(args.baseline_main_voi),
    )
    summary.to_csv(output_dir / "noise_sensitivity_summary.csv", index=False)
    summary.to_latex(output_dir / "table_noise_sensitivity.tex", index=False, escape=False, float_format="%.6g")
    _write_report(summary, output_dir / "report_noise_sensitivity.md")
    print(f"Wrote {output_dir / 'noise_sensitivity_summary.csv'}")
    print(f"Wrote {output_dir / 'report_noise_sensitivity.md'}")


def _run_case(
    *,
    case: NoiseCase,
    observables_csv: Path,
    jacobians: Path,
    seed_start: int,
    num_seeds: int,
    validation_seeds: str,
    test_seeds: str,
    num_candidates: int,
) -> None:
    case.output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "experiments/exp03_build_observations.py",
            "--observables-csv",
            str(observables_csv),
            "--output-dir",
            str(case.output_dir),
            "--seed-start",
            str(seed_start),
            "--num-seeds",
            str(num_seeds),
            "--aggregate-noise-scale",
            str(case.aggregate_noise_scale),
            "--distribution-noise-scale",
            str(case.distribution_noise_scale),
        ]
    )
    _run(
        [
            "experiments/exp04_filter_states.py",
            "--observables-csv",
            str(observables_csv),
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
            "experiments/exp05_build_information_inputs.py",
            "--observables-csv",
            str(observables_csv),
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
            "experiments/exp08_main_voi.py",
            "--information-inputs",
            str(case.output_dir / "information_state_inputs_long.csv"),
            "--hank-observables",
            str(observables_csv),
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


def _summarize(*, cases: list[NoiseCase], baseline_main_voi: Path) -> pd.DataFrame:
    rows = [_extract_case("baseline", "both", 1.0, 1.0, baseline_main_voi)]
    for case in cases:
        rows.append(
            _extract_case(
                case.name,
                case.axis,
                case.aggregate_noise_scale,
                case.distribution_noise_scale,
                case.output_dir / "main_voi",
            )
        )
    return pd.DataFrame(rows).sort_values(["axis", "aggregate_noise_scale", "distribution_noise_scale"]).reset_index(drop=True)


def _extract_case(
    name: str,
    axis: str,
    aggregate_noise_scale: float,
    distribution_noise_scale: float,
    main_voi_dir: Path,
) -> dict[str, object]:
    summary = pd.read_csv(main_voi_dir / "main_voi_summary.csv")
    pairwise = pd.read_csv(main_voi_dir / "pairwise_value_of_information.csv")
    gap = pd.read_csv(main_voi_dir / "full_information_gap.csv")

    overall = summary[summary["scenario"] == "all"].set_index("information_state")
    comparison = pairwise[
        (pairwise["scenario"] == "all")
        & (pairwise["comparison"] == "filtered_distribution_minus_filtered_aggregates")
    ].iloc[0]
    gap_overall = gap[gap["scenario"] == "all"].set_index("information_state")

    return {
        "case": name,
        "axis": axis,
        "aggregate_noise_scale": float(aggregate_noise_scale),
        "distribution_noise_scale": float(distribution_noise_scale),
        "loss_filtered_aggregates": float(overall.loc["filtered_aggregates", "mean_loss"]),
        "loss_filtered_distribution": float(overall.loc["filtered_distribution", "mean_loss"]),
        "mvoi_dist": float(comparison["loss_reduction"]),
        "delta_filtered_distribution_minus_filtered_aggregates": float(comparison["mean_delta"]),
        "ci_low": float(comparison["ci_low"]),
        "ci_high": float(comparison["ci_high"]),
        "win_rate": float(comparison["win_rate"]),
        "loss_rate": float(comparison["loss_rate"]),
        "share_gap_closed_filtered_distribution": float(
            gap_overall.loc["filtered_distribution", "share_of_gap_closed"]
        ),
        "num_trajectories": int(comparison["num_trajectories"]),
    }


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Чувствительность к шуму наблюдений",
        "",
        "Таблица показывает, как меняется предельная ценность распределительной информации при изменении шума агрегатных и распределительных наблюдений.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['case']}: шум агрегатов {row['aggregate_noise_scale']:.2g}, "
            f"шум распределительных показателей {row['distribution_noise_scale']:.2g}, "
            f"MVOI {row['mvoi_dist']:.6g}, интервал для разности "
            f"[{row['ci_low']:.6g}, {row['ci_high']:.6g}], win rate {row['win_rate']:.3g}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _noise_cases(*, output_dir: Path, scales: list[float]) -> list[NoiseCase]:
    cases: list[NoiseCase] = []
    for scale in scales:
        cases.append(
            NoiseCase(
                name=f"aggregate_noise_{_scale_label(scale)}",
                axis="aggregate",
                aggregate_noise_scale=scale,
                distribution_noise_scale=1.0,
                output_dir=output_dir / f"aggregate_noise_{_scale_label(scale)}",
            )
        )
        cases.append(
            NoiseCase(
                name=f"distribution_noise_{_scale_label(scale)}",
                axis="distribution",
                aggregate_noise_scale=1.0,
                distribution_noise_scale=scale,
                output_dir=output_dir / f"distribution_noise_{_scale_label(scale)}",
            )
        )
    return cases


def _parse_scales(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part.strip()]


def _scale_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
