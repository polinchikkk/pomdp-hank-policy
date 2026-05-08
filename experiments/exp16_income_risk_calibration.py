from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank.calibration import default_calibration
from hank.sjacobian import solve_sequence_space_jacobian
from hank.steady_state import solve_steady_state
from hank_ssj.artifacts import SSJArtifactSpec, export_long_jacobian_to_npz
from hank_ssj.shock_library import build_shock_response_library, generate_stochastic_hank_paths


@dataclass(frozen=True)
class IncomeRiskCalibrationSpec:
    output_dir: str
    sigma_z_values: tuple[float, ...]
    num_trajectories: int
    seed_start: int
    observation_seed_start: int
    num_observation_seeds: int
    validation_seeds: str
    test_seeds: str
    num_candidates: int
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HANK/SSJ value-of-information sensitivity to idiosyncratic income risk.")
    parser.add_argument("--output-dir", default="outputs/ssj/income_risk_calibration")
    parser.add_argument("--sigma-z-values", default="0.84,0.88,0.92")
    parser.add_argument("--num-trajectories", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1200)
    parser.add_argument("--observation-seed-start", type=int, default=900)
    parser.add_argument("--num-observation-seeds", type=int, default=12)
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--num-candidates", type=int, default=70)
    parser.add_argument("--shock-size", type=float, default=0.001)
    parser.add_argument("--skip-runs", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma_z_values = tuple(float(value) for value in args.sigma_z_values.split(",") if value.strip())

    if not args.skip_runs:
        for sigma_z in sigma_z_values:
            case_dir = output_dir / f"sigma_z_{_value_label(sigma_z)}"
            case_dir.mkdir(parents=True, exist_ok=True)
            _run_case(
                sigma_z=sigma_z,
                case_dir=case_dir,
                shock_size=args.shock_size,
                seed_start=args.seed_start,
                num_trajectories=args.num_trajectories,
                observation_seed_start=args.observation_seed_start,
                num_observation_seeds=args.num_observation_seeds,
                validation_seeds=args.validation_seeds,
                test_seeds=args.test_seeds,
                num_candidates=args.num_candidates,
            )

    summary = _summarize(output_dir=output_dir, sigma_z_values=sigma_z_values)
    summary.to_csv(output_dir / "income_risk_calibration_summary.csv", index=False)
    summary.to_latex(output_dir / "table_income_risk_calibration.tex", index=False, escape=False, float_format="%.6g")
    _write_report(summary, output_dir / "report_income_risk_calibration.md")
    spec = IncomeRiskCalibrationSpec(
        output_dir=args.output_dir,
        sigma_z_values=sigma_z_values,
        num_trajectories=int(args.num_trajectories),
        seed_start=int(args.seed_start),
        observation_seed_start=int(args.observation_seed_start),
        num_observation_seeds=int(args.num_observation_seeds),
        validation_seeds=args.validation_seeds,
        test_seeds=args.test_seeds,
        num_candidates=int(args.num_candidates),
        note=(
            "Для каждого значения sigma_z заново решается HANK steady state, пересчитываются SSJ-отклики, "
            "строятся стохастические HANK/SSJ-траектории и затем оценивается ценность распределительной информации."
        ),
    )
    (output_dir / "income_risk_calibration_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'income_risk_calibration_summary.csv'}")
    print(f"Wrote {output_dir / 'report_income_risk_calibration.md'}")


def _run_case(
    *,
    sigma_z: float,
    case_dir: Path,
    shock_size: float,
    seed_start: int,
    num_trajectories: int,
    observation_seed_start: int,
    num_observation_seeds: int,
    validation_seeds: str,
    test_seeds: str,
    num_candidates: int,
) -> None:
    config = replace(default_calibration(), sigma_z=float(sigma_z), output_dir=str(case_dir / "hank_core"))
    bundle = solve_steady_state(config)
    jacobian_df = solve_sequence_space_jacobian(bundle, config.shock_T)
    jacobian_csv = case_dir / "jacobian_summary.csv"
    jacobian_df.to_csv(jacobian_csv, index=False)
    export_long_jacobian_to_npz(
        jacobian_csv=jacobian_csv,
        output_path=case_dir / "jacobians.npz",
        spec=SSJArtifactSpec(
            source=str(jacobian_csv),
            horizon=int(config.shock_T),
            input_name="monetary_policy_shock",
            note="SSJ monetary-policy Jacobian recomputed for income-risk calibration sensitivity.",
        ),
    )
    build_shock_response_library(output_dir=case_dir, shock_size=shock_size, config=config)
    generate_stochastic_hank_paths(
        shock_library_csv=case_dir / "shock_response_library.csv",
        steady_distributional_values_json=case_dir / "steady_distributional_values.json",
        output_dir=case_dir,
        trajectory_seeds=tuple(range(seed_start, seed_start + num_trajectories)),
    )
    _run(
        [
            "experiments/exp03_build_observations.py",
            "--observables-csv",
            str(case_dir / "hank_observables.csv"),
            "--output-dir",
            str(case_dir),
            "--seed-start",
            str(observation_seed_start),
            "--num-seeds",
            str(num_observation_seeds),
        ]
    )
    _run(
        [
            "experiments/exp04_filter_states.py",
            "--observables-csv",
            str(case_dir / "hank_observables.csv"),
            "--observations-csv",
            str(case_dir / "hank_observations.csv"),
            "--observations-spec",
            str(case_dir / "hank_observations_spec.json"),
            "--output-dir",
            str(case_dir),
        ]
    )
    _run(
        [
            "experiments/exp05_build_information_inputs.py",
            "--observables-csv",
            str(case_dir / "hank_observables.csv"),
            "--observations-csv",
            str(case_dir / "hank_observations.csv"),
            "--filtered-states-csv",
            str(case_dir / "filtered_states.csv"),
            "--output-dir",
            str(case_dir),
        ]
    )
    _run(
        [
            "experiments/exp08_main_voi.py",
            "--information-inputs",
            str(case_dir / "information_state_inputs_long.csv"),
            "--hank-observables",
            str(case_dir / "hank_observables.csv"),
            "--jacobians",
            str(case_dir / "jacobians.npz"),
            "--output-dir",
            str(case_dir / "main_voi"),
            "--validation-seeds",
            validation_seeds,
            "--test-seeds",
            test_seeds,
            "--num-candidates",
            str(num_candidates),
        ]
    )


def _summarize(*, output_dir: Path, sigma_z_values: tuple[float, ...]) -> pd.DataFrame:
    rows = []
    for sigma_z in sigma_z_values:
        case_dir = output_dir / f"sigma_z_{_value_label(sigma_z)}"
        rows.append(_extract_case(sigma_z=sigma_z, case_dir=case_dir))
    return pd.DataFrame(rows).sort_values("sigma_z").reset_index(drop=True)


def _extract_case(*, sigma_z: float, case_dir: Path) -> dict[str, object]:
    summary = pd.read_csv(case_dir / "main_voi" / "main_voi_summary.csv")
    pairwise = pd.read_csv(case_dir / "main_voi" / "pairwise_value_of_information.csv")
    gap = pd.read_csv(case_dir / "main_voi" / "full_information_gap.csv")
    steady = json.loads((case_dir / "steady_distributional_values.json").read_text(encoding="utf-8"))
    overall = summary[summary["scenario"] == "all"].set_index("information_state")
    comparison = pairwise[
        (pairwise["scenario"] == "all")
        & (pairwise["comparison"] == "filtered_distribution_minus_filtered_aggregates")
    ].iloc[0]
    gap_overall = gap[gap["scenario"] == "all"].set_index("information_state")
    return {
        "sigma_z": float(sigma_z),
        "steady_mean_mpc": float(steady["mean_mpc"]),
        "steady_share_low_liquidity": float(steady["share_low_liquidity"]),
        "steady_interest_exposure": float(steady["interest_exposure"]),
        "loss_filtered_aggregates": float(overall.loc["filtered_aggregates", "mean_loss"]),
        "loss_filtered_distribution": float(overall.loc["filtered_distribution", "mean_loss"]),
        "mvoi_dist": float(comparison["loss_reduction"]),
        "delta_filtered_distribution_minus_filtered_aggregates": float(comparison["mean_delta"]),
        "ci_low": float(comparison["ci_low"]),
        "ci_high": float(comparison["ci_high"]),
        "win_rate": float(comparison["win_rate"]),
        "tie_rate": float(comparison.get("tie_rate", 0.0)),
        "loss_rate": float(comparison["loss_rate"]),
        "share_gap_closed_filtered_distribution": float(gap_overall.loc["filtered_distribution", "share_of_gap_closed"]),
        "num_trajectories": int(comparison["num_trajectories"]),
    }


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Чувствительность к доходному риску HANK",
        "",
        "Для каждого значения sigma_z заново пересчитываются HANK steady state, SSJ-отклики и HANK/SSJ-траектории.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- sigma_z={row['sigma_z']:.3g}: mean MPC {row['steady_mean_mpc']:.4g}, "
            f"низкая ликвидность {row['steady_share_low_liquidity']:.4g}, "
            f"MVOI {row['mvoi_dist']:.6g}, "
            f"интервал для разности [{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
            f"доля выигрышей {row['win_rate']:.3g}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _value_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
