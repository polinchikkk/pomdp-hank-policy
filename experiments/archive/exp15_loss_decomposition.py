from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.optimize_rules import bootstrap_interval


COMPARISONS = (
    ("filtered_aggregates", "aggregate_only", "Ценность фильтрованных агрегатов"),
    ("observed_distribution", "aggregate_only", "Ценность наблюдаемых распределительных показателей"),
    ("filtered_distribution", "filtered_aggregates", "Предельная ценность распределительной информации"),
    ("full_information", "filtered_distribution", "Оставшийся разрыв до полной информации"),
)

COMPONENTS = (
    ("inflation_loss", "Инфляция"),
    ("output_gap_loss", "Разрыв выпуска"),
    ("consumption_loss", "Потребление"),
    ("rate_smoothing_loss", "Сглаживание ставки"),
    ("stability_penalty", "Штраф устойчивости"),
    ("total_loss", "Итого"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Decompose pairwise policy-loss reductions by loss component.")
    parser.add_argument("--trajectory-losses", default="outputs/ssj/stochastic/main_voi/trajectory_losses.csv")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/main_voi")
    args = parser.parse_args()

    losses = pd.read_csv(args.trajectory_losses)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decomposition = _decompose(losses)
    decomposition.to_csv(output_dir / "loss_component_decomposition.csv", index=False)
    _write_latex(decomposition, output_dir / "table_loss_component_decomposition.tex")
    _write_report(decomposition, output_dir / "report_loss_component_decomposition.md")
    print(f"Wrote {output_dir / 'loss_component_decomposition.csv'}")
    print(f"Wrote {output_dir / 'report_loss_component_decomposition.md'}")


def _decompose(losses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario, frame in _scenario_groups(losses):
        for left, right, label in COMPARISONS:
            merged = _paired_frame(frame, left=left, right=right)
            for component, component_label in COMPONENTS:
                reduction = merged[f"{component}_right"].to_numpy(dtype=float) - merged[f"{component}_left"].to_numpy(dtype=float)
                ci_low, ci_high = bootstrap_interval(reduction)
                rows.append(
                    {
                        "scenario": scenario,
                        "comparison": f"{left}_minus_{right}",
                        "comparison_ru": label,
                        "left": left,
                        "right": right,
                        "component": component,
                        "component_ru": component_label,
                        "num_trajectories": int(reduction.size),
                        "mean_reduction": float(np.mean(reduction)),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "share_of_total_reduction": np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    total = (
        result[result["component"] == "total_loss"]
        .set_index(["scenario", "comparison"])["mean_reduction"]
        .to_dict()
    )
    shares = []
    for row in result.itertuples(index=False):
        denominator = float(total.get((row.scenario, row.comparison), np.nan))
        shares.append(float(row.mean_reduction / denominator) if abs(denominator) > 1e-14 else np.nan)
    result["share_of_total_reduction"] = shares
    return result


def _paired_frame(frame: pd.DataFrame, *, left: str, right: str) -> pd.DataFrame:
    keys = ["scenario", "observation_seed"]
    left_frame = frame[frame["information_state"] == left][[*keys, *(component for component, _ in COMPONENTS)]]
    right_frame = frame[frame["information_state"] == right][[*keys, *(component for component, _ in COMPONENTS)]]
    return left_frame.merge(
        right_frame,
        on=keys,
        how="inner",
        suffixes=("_left", "_right"),
        validate="one_to_one",
    )


def _scenario_groups(losses: pd.DataFrame):
    for scenario, frame in losses.groupby("scenario", sort=False):
        yield scenario, frame
    yield "all", losses


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    for column in numeric:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_report(frame: pd.DataFrame, path: Path) -> None:
    overall = frame[(frame["scenario"] == "all") & (frame["comparison"] == "filtered_distribution_minus_filtered_aggregates")]
    lines = [
        "# Разложение снижения потерь",
        "",
        "Таблица показывает, через какие компоненты функции потерь проходит выигрыш от распределительной информации.",
        "",
    ]
    if not overall.empty:
        lines.append("## Предельная ценность распределительной информации")
        lines.append("")
        for _, row in overall.iterrows():
            if row["component"] == "total_loss":
                continue
            lines.append(
                f"- {row['component_ru']}: снижение {row['mean_reduction']:.6g}, "
                f"доля общего снижения {row['share_of_total_reduction']:.3g}."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
