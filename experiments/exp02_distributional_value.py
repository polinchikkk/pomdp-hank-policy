from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp01_baseline_voi import run_experiment


SCENARIOS = (
    "baseline",
    "high_aggregate_noise",
    "high_heterogeneity",
    "noisy_distributional_data",
)


def run_scenario_comparison(
    *,
    output_dir: Path,
    horizon: int,
    validation_count: int,
    test_count: int,
    num_candidates: int,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in SCENARIOS:
        scenario_dir = output_dir / scenario
        result = run_experiment(
            scenario=scenario,
            output_dir=scenario_dir,
            horizon=horizon,
            validation_count=validation_count,
            test_count=test_count,
            num_candidates=num_candidates,
        )
        gap = result["gap_summary"].set_index("information_state")
        pairwise = result["pairwise"].set_index(["left", "right"])
        dist_row = gap.loc["distributional"]
        full_row = gap.loc["full_information"]
        dist_vs_agg = pairwise.loc[("distributional", "aggregate_only")]
        dist_vs_filtered = pairwise.loc[("distributional", "filtered_aggregates")]
        rows.append(
            {
                "scenario": scenario,
                "aggregate_loss": float(gap.loc["aggregate_only", "mean_loss"]),
                "filtered_aggregate_loss": float(gap.loc["filtered_aggregates", "mean_loss"]),
                "distributional_loss": float(dist_row["mean_loss"]),
                "full_information_loss": float(full_row["mean_loss"]),
                "distributional_value_vs_aggregate": float(dist_vs_agg["loss_reduction"]),
                "distributional_value_vs_aggregate_ci_low": float(dist_vs_agg["loss_reduction_ci_low"]),
                "distributional_value_vs_aggregate_ci_high": float(dist_vs_agg["loss_reduction_ci_high"]),
                "distributional_win_rate_vs_aggregate": float(dist_vs_agg["win_rate"]),
                "distributional_value_vs_filtered": float(dist_vs_filtered["loss_reduction"]),
                "distributional_value_vs_filtered_ci_low": float(dist_vs_filtered["loss_reduction_ci_low"]),
                "distributional_value_vs_filtered_ci_high": float(dist_vs_filtered["loss_reduction_ci_high"]),
                "distributional_win_rate_vs_filtered": float(dist_vs_filtered["win_rate"]),
                "share_of_full_information_gap_closed": float(dist_row["share_of_full_information_gap_closed"]),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "scenario_distributional_value.csv", index=False)
    (output_dir / "table_scenario_distributional_value.tex").write_text(
        summary.to_latex(index=False, float_format="%.6f"),
        encoding="utf-8",
    )
    _write_report(output_dir, summary)
    return summary


def _write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Эксперимент 2. Когда распределительная информация важна",
        "",
        "В таблице сравнивается ценность распределительного информационного состояния в нескольких средах.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- `{row['scenario']}`: ценность против агрегатов "
            f"`{row['distributional_value_vs_aggregate']:.6f}`, "
            f"доля выигрышных траекторий `{row['distributional_win_rate_vs_aggregate']:.3f}`, "
            f"доля закрытия разрыва `{row['share_of_full_information_gap_closed']:.3f}`."
        )
    lines.extend(
        [
            "",
            "Положительное значение означает, что добавление распределительных статистик снижает средние потери политики. Отрицательное значение означает, что в данной калибровке дополнительные статистики не улучшают выбранное линейное правило.",
        ]
    )
    (output_dir / "report_exp02_distributional_value.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario comparison for distributional information value.")
    parser.add_argument("--output-dir", default="outputs/exp02_distributional_value")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--validation-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--num-candidates", type=int, default=350)
    args = parser.parse_args()

    run_scenario_comparison(
        output_dir=Path(args.output_dir),
        horizon=args.horizon,
        validation_count=args.validation_count,
        test_count=args.test_count,
        num_candidates=args.num_candidates,
    )


if __name__ == "__main__":
    main()
