from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

CENTERED_DISTRIBUTIONAL_COLUMNS = (
    "mean_mpc_centered",
    "share_low_liquidity_centered",
    "interest_exposure_centered",
)

LEVEL_DISTRIBUTIONAL_COLUMNS = {
    "mean_mpc": "mean_mpc",
    "share_low_liquidity": "share_low_liquidity",
    "interest_exposure": "interest_exposure",
}


@dataclass(frozen=True)
class DistributionalSignalStrengthSpec:
    source_observables: str
    steady_distributional_values: str
    noise_reference_observables: str
    output_dir: str
    factors: tuple[float, ...]
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run distributional signal strength sensitivity.")
    parser.add_argument("--source-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--steady-distributional-values", default="outputs/ssj/stochastic/steady_distributional_values.json")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/distributional_signal_strength")
    parser.add_argument("--factors", default="0,0.5,1,2")
    parser.add_argument("--seed-start", type=int, default=900)
    parser.add_argument("--num-seeds", type=int, default=12)
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--num-candidates", type=int, default=90)
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    factors = tuple(float(value) for value in args.factors.split(",") if value.strip())

    if not args.skip_runs:
        for factor in factors:
            case_dir = output_dir / f"signal_{_factor_label(factor)}"
            case_dir.mkdir(parents=True, exist_ok=True)
            observables_csv = case_dir / "hank_observables.csv"
            _write_scaled_observables(
                source_observables=Path(args.source_observables),
                steady_distributional_values=Path(args.steady_distributional_values),
                output_path=observables_csv,
                factor=factor,
            )
            _run_pipeline(
                observables_csv=observables_csv,
                noise_reference_csv=Path(args.source_observables),
                jacobians=Path(args.jacobians),
                output_dir=case_dir,
                seed_start=args.seed_start,
                num_seeds=args.num_seeds,
                validation_seeds=args.validation_seeds,
                test_seeds=args.test_seeds,
                num_candidates=args.num_candidates,
            )

    summary = _summarize(output_dir=output_dir, factors=factors)
    summary.to_csv(output_dir / "distributional_signal_strength_summary.csv", index=False)
    summary.to_latex(output_dir / "table_distributional_signal_strength.tex", index=False, escape=False, float_format="%.6g")
    _write_report(summary, output_dir / "report_distributional_signal_strength.md")

    spec = DistributionalSignalStrengthSpec(
        source_observables=args.source_observables,
        steady_distributional_values=args.steady_distributional_values,
        noise_reference_observables=args.source_observables,
        output_dir=args.output_dir,
        factors=factors,
        note=(
            "Центрированные распределительные статистики умножаются на заданный фактор, "
            "а масштаб шума наблюдений считается по исходным HANK/SSJ-траекториям. "
            "Поэтому эксперимент меняет информативность распределительного сигнала относительно шума."
        ),
    )
    (output_dir / "distributional_signal_strength_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'distributional_signal_strength_summary.csv'}")
    print(f"Wrote {output_dir / 'report_distributional_signal_strength.md'}")


def _write_scaled_observables(
    *,
    source_observables: Path,
    steady_distributional_values: Path,
    output_path: Path,
    factor: float,
) -> None:
    frame = pd.read_csv(source_observables)
    steady = json.loads(steady_distributional_values.read_text(encoding="utf-8"))
    missing = sorted(set(CENTERED_DISTRIBUTIONAL_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{source_observables} is missing columns: {missing}")

    result = frame.copy()
    for column in CENTERED_DISTRIBUTIONAL_COLUMNS:
        result[column] = float(factor) * result[column].to_numpy(dtype=float)
    for column, steady_key in LEVEL_DISTRIBUTIONAL_COLUMNS.items():
        centered = f"{column}_centered"
        if column in result.columns and centered in result.columns and steady_key in steady:
            result[column] = float(steady[steady_key]) + result[centered].to_numpy(dtype=float)
    result.to_csv(output_path, index=False)


def _run_pipeline(
    *,
    observables_csv: Path,
    noise_reference_csv: Path,
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
            "--noise-reference-csv",
            str(noise_reference_csv),
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


def _summarize(*, output_dir: Path, factors: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for factor in factors:
        rows.append(_extract_case(factor, output_dir / f"signal_{_factor_label(factor)}" / "main_voi"))
    return pd.DataFrame(rows).sort_values("distributional_signal_factor").reset_index(drop=True)


def _extract_case(factor: float, main_voi_dir: Path) -> dict[str, object]:
    summary = pd.read_csv(main_voi_dir / "main_voi_summary.csv")
    pairwise = pd.read_csv(main_voi_dir / "pairwise_value_of_information.csv")
    overall = summary[summary["scenario"] == "all"].set_index("information_state")
    comparison = pairwise[
        (pairwise["scenario"] == "all")
        & (pairwise["comparison"] == "filtered_distribution_minus_filtered_aggregates")
    ].iloc[0]
    return {
        "distributional_signal_factor": float(factor),
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
        "# Чувствительность к силе распределительного сигнала",
        "",
        "Масштаб шума наблюдений фиксируется по исходным HANK/SSJ-траекториям. Меняется только амплитуда распределительных отклонений.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- фактор {row['distributional_signal_factor']:.3g}: MVOI {row['mvoi_dist']:.6g}, "
            f"интервал для разности [{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
            f"доля выигрышей {row['win_rate']:.3g}, доля совпадений {row['tie_rate']:.3g}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _factor_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
