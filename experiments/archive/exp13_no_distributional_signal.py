from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DISTRIBUTIONAL_COLUMNS = (
    "mean_mpc_centered",
    "share_low_liquidity_centered",
    "interest_exposure_centered",
)

DISTRIBUTIONAL_LEVELS = {
    "mean_mpc": "mean_mpc",
    "share_low_liquidity": "share_low_liquidity",
    "interest_exposure": "interest_exposure",
}


@dataclass(frozen=True)
class NoDistributionalSignalSpec:
    source_observables: str
    steady_distributional_values: str
    output_dir: str
    neutralized_columns: tuple[str, ...]
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the no-distributional-signal falsification check.")
    parser.add_argument("--source-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--steady-distributional-values", default="outputs/ssj/stochastic/steady_distributional_values.json")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/no_distributional_signal")
    parser.add_argument("--baseline-main-voi", default="outputs/ssj/stochastic/main_voi")
    parser.add_argument("--seed-start", type=int, default=900)
    parser.add_argument("--num-seeds", type=int, default=12)
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--num-candidates", type=int, default=90)
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    neutralized_observables = output_dir / "hank_observables.csv"
    if not args.skip_runs:
        _write_neutralized_observables(
            source_observables=Path(args.source_observables),
            steady_distributional_values=Path(args.steady_distributional_values),
            output_path=neutralized_observables,
            output_dir=output_dir,
        )
        _run_pipeline(
            observables_csv=neutralized_observables,
            jacobians=Path(args.jacobians),
            output_dir=output_dir,
            seed_start=args.seed_start,
            num_seeds=args.num_seeds,
            validation_seeds=args.validation_seeds,
            test_seeds=args.test_seeds,
            num_candidates=args.num_candidates,
        )

    summary = _summarize(
        baseline_main_voi=Path(args.baseline_main_voi),
        neutralized_main_voi=output_dir / "main_voi",
    )
    summary.to_csv(output_dir / "no_distributional_signal_summary.csv", index=False)
    summary.to_latex(output_dir / "table_no_distributional_signal.tex", index=False, escape=False, float_format="%.6g")
    _write_report(summary, output_dir / "report_no_distributional_signal.md")
    print(f"Wrote {output_dir / 'no_distributional_signal_summary.csv'}")
    print(f"Wrote {output_dir / 'report_no_distributional_signal.md'}")


def _write_neutralized_observables(
    *,
    source_observables: Path,
    steady_distributional_values: Path,
    output_path: Path,
    output_dir: Path,
) -> None:
    frame = pd.read_csv(source_observables)
    missing = sorted(set(DISTRIBUTIONAL_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{source_observables} is missing columns: {missing}")

    steady = json.loads(steady_distributional_values.read_text(encoding="utf-8"))
    result = frame.copy()
    for column in DISTRIBUTIONAL_COLUMNS:
        result[column] = 0.0
    for column, steady_key in DISTRIBUTIONAL_LEVELS.items():
        if column in result.columns and steady_key in steady:
            result[column] = float(steady[steady_key])
    result.to_csv(output_path, index=False)

    spec = NoDistributionalSignalSpec(
        source_observables=str(source_observables),
        steady_distributional_values=str(steady_distributional_values),
        output_dir=str(output_dir),
        neutralized_columns=(*DISTRIBUTIONAL_COLUMNS, *DISTRIBUTIONAL_LEVELS.keys()),
        note=(
            "Проверка выключает динамику распределительных статистик, сохраняя агрегатные HANK/SSJ-траектории. "
            "Если выигрыш был связан с содержательным распределительным сигналом, предельная ценность "
            "распределительной информации должна исчезнуть или резко уменьшиться."
        ),
    )
    (output_dir / "no_distributional_signal_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _run_pipeline(
    *,
    observables_csv: Path,
    jacobians: Path,
    output_dir: Path,
    seed_start: int,
    num_seeds: int,
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
            str(seed_start),
            "--num-seeds",
            str(num_seeds),
            "--distribution-noise-scale",
            "0.0",
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


def _summarize(*, baseline_main_voi: Path, neutralized_main_voi: Path) -> pd.DataFrame:
    rows = [
        _extract_case("actual_distributional_dynamics", "Фактическая распределительная динамика", baseline_main_voi),
        _extract_case("no_distributional_signal", "Без распределительного сигнала", neutralized_main_voi),
    ]
    return pd.DataFrame(rows)


def _extract_case(name: str, label: str, main_voi_dir: Path) -> dict[str, object]:
    summary = pd.read_csv(main_voi_dir / "main_voi_summary.csv")
    pairwise = pd.read_csv(main_voi_dir / "pairwise_value_of_information.csv")
    overall = summary[summary["scenario"] == "all"].set_index("information_state")
    comparison = pairwise[
        (pairwise["scenario"] == "all")
        & (pairwise["comparison"] == "filtered_distribution_minus_filtered_aggregates")
    ].iloc[0]
    return {
        "case": name,
        "description": label,
        "loss_filtered_aggregates": float(overall.loc["filtered_aggregates", "mean_loss"]),
        "loss_filtered_distribution": float(overall.loc["filtered_distribution", "mean_loss"]),
        "mvoi_dist": float(comparison["loss_reduction"]),
        "delta_filtered_distribution_minus_filtered_aggregates": float(comparison["mean_delta"]),
        "ci_low": float(comparison["ci_low"]),
        "ci_high": float(comparison["ci_high"]),
        "win_rate": float(comparison["win_rate"]),
        "tie_rate": float(comparison.get("tie_rate", 0.0)),
        "loss_rate": float(comparison["loss_rate"]),
        "num_trajectories": int(comparison["num_trajectories"]),
    }


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Проверка без распределительного сигнала",
        "",
        "В этой проверке агрегатные HANK/SSJ-траектории сохраняются, а динамика распределительных статистик выключается.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['description']}: MVOI {row['mvoi_dist']:.6g}, "
            f"интервал для разности [{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
            f"доля выигрышных траекторий {row['win_rate']:.3g}, "
            f"доля совпадений {row['tie_rate']:.3g}."
        )
    lines.extend(
        [
            "",
            "Интерпретация: при отсутствии распределительного сигнала предельная ценность распределительной информации должна исчезать или резко снижаться.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
