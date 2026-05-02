from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp01_baseline_voi import run_experiment
from state_space.local_environment import LocalEnvironmentConfig


AGGREGATE_NOISE_GRID = (0.5, 1.0, 2.0)
MPC_CHANNEL_GRID = (0.75, 1.25, 1.75)


def run_distributional_value_grid(
    *,
    output_dir: Path,
    horizon: int,
    validation_count: int,
    test_count: int,
    num_candidates: int,
    distributional_observation_noise: float,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for aggregate_noise in AGGREGATE_NOISE_GRID:
        for mpc_strength in MPC_CHANNEL_GRID:
            scenario_name = f"agg_noise_{aggregate_noise:g}_mpc_{mpc_strength:g}"
            result = run_experiment(
                scenario=scenario_name,
                output_dir=output_dir / scenario_name,
                horizon=horizon,
                validation_count=validation_count,
                test_count=test_count,
                num_candidates=num_candidates,
                environment_config=LocalEnvironmentConfig(
                    horizon=horizon,
                    aggregate_observation_noise=aggregate_noise,
                    distributional_observation_noise=distributional_observation_noise,
                    mpc_channel_strength=mpc_strength,
                    heterogeneity="high" if mpc_strength >= 1.75 else "baseline",
                ),
            )
            pairwise = result["pairwise"].set_index(["left", "right"])
            gap = result["gap_summary"].set_index("information_state")
            dist_vs_agg = pairwise.loc[("distributional", "aggregate_only")]
            rows.append(
                {
                    "scenario": scenario_name,
                    "aggregate_observation_noise": aggregate_noise,
                    "mpc_channel_strength": mpc_strength,
                    "distributional_observation_noise": distributional_observation_noise,
                    "distributional_value_vs_aggregate": float(dist_vs_agg["loss_reduction"]),
                    "distributional_value_ci_low": float(dist_vs_agg["loss_reduction_ci_low"]),
                    "distributional_value_ci_high": float(dist_vs_agg["loss_reduction_ci_high"]),
                    "win_rate": float(dist_vs_agg["win_rate"]),
                    "share_of_full_information_gap_closed": float(
                        gap.loc["distributional", "share_of_full_information_gap_closed"]
                    ),
                    "aggregate_loss": float(gap.loc["aggregate_only", "mean_loss"]),
                    "distributional_loss": float(gap.loc["distributional", "mean_loss"]),
                    "full_information_loss": float(gap.loc["full_information", "mean_loss"]),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "distributional_value_grid.csv", index=False)
    (output_dir / "table_distributional_value_grid.tex").write_text(
        summary.to_latex(index=False, float_format="%.6f"),
        encoding="utf-8",
    )
    _write_report(output_dir, summary)
    return summary


def _write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    best = summary.loc[summary["distributional_value_vs_aggregate"].idxmax()]
    worst = summary.loc[summary["distributional_value_vs_aggregate"].idxmin()]
    lines = [
        "# Эксперимент 6. Сетка ценности распределительной информации",
        "",
        "Сетка варьирует шум агрегатных наблюдений и силу распределительного канала.",
        "",
        f"- Наибольшая ценность: `{best['distributional_value_vs_aggregate']:.6f}` в сценарии `{best['scenario']}`.",
        f"- Наименьшая ценность: `{worst['distributional_value_vs_aggregate']:.6f}` в сценарии `{worst['scenario']}`.",
        "",
        "Положительное значение означает снижение потерь при переходе от агрегатной информации к распределительному информационному состоянию.",
    ]
    (output_dir / "report_exp06_distributional_value_grid.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run dense grid for distributional information value.")
    parser.add_argument("--output-dir", default="outputs/exp06_distributional_value_grid")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=30)
    parser.add_argument("--num-candidates", type=int, default=220)
    parser.add_argument("--distributional-observation-noise", type=float, default=1.0)
    args = parser.parse_args()

    run_distributional_value_grid(
        output_dir=Path(args.output_dir),
        horizon=args.horizon,
        validation_count=args.validation_count,
        test_count=args.test_count,
        num_candidates=args.num_candidates,
        distributional_observation_noise=args.distributional_observation_noise,
    )


if __name__ == "__main__":
    main()
