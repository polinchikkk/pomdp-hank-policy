from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RUNS = {
    "actual": "Фактическая распределительная информация",
    "permuted": "Перемешанные распределительные ряды",
    "fake": "Искусственные распределительные ряды",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize falsification tests for distributional information.")
    parser.add_argument("--actual-dir", default="outputs/ssj/stochastic/main_voi")
    parser.add_argument("--permuted-dir", default="outputs/ssj/stochastic/placebo/main_voi_permuted")
    parser.add_argument("--fake-dir", default="outputs/ssj/stochastic/placebo/main_voi_fake")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/placebo")
    args = parser.parse_args()

    run_dirs = {
        "actual": Path(args.actual_dir),
        "permuted": Path(args.permuted_dir),
        "fake": Path(args.fake_dir),
    }
    rows = [_extract_row(name, path) for name, path in run_dirs.items()]
    summary = pd.DataFrame(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "placebo_summary.csv", index=False)
    summary.to_latex(output_dir / "table_placebo_summary.tex", index=False, escape=False, float_format="%.6g")
    _write_report(summary, output_dir / "report_placebo_tests.md")
    print(f"Wrote {output_dir / 'placebo_summary.csv'}")
    print(f"Wrote {output_dir / 'report_placebo_tests.md'}")


def _extract_row(name: str, run_dir: Path) -> dict[str, object]:
    pairwise = pd.read_csv(run_dir / "pairwise_value_of_information.csv")
    summary = pd.read_csv(run_dir / "main_voi_summary.csv")
    comparison = pairwise[
        (pairwise["scenario"] == "all")
        & (pairwise["comparison"] == "filtered_distribution_minus_filtered_aggregates")
    ].iloc[0]
    filtered_distribution = summary[
        (summary["scenario"] == "all")
        & (summary["information_state"] == "filtered_distribution")
    ].iloc[0]
    return {
        "run": name,
        "описание": RUNS[name],
        "loss_filtered_distribution": float(filtered_distribution["mean_loss"]),
        "mvoi_delta": float(comparison["mean_delta"]),
        "loss_reduction": float(comparison["loss_reduction"]),
        "ci_low": float(comparison["ci_low"]),
        "ci_high": float(comparison["ci_high"]),
        "win_rate": float(comparison["win_rate"]),
        "loss_rate": float(comparison["loss_rate"]),
        "num_trajectories": int(comparison["num_trajectories"]),
    }


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Проверки с искусственной распределительной информацией",
        "",
        "Сравнение показывает, сохраняется ли предельная ценность распределительной информации, если распределительные признаки разрушить.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['описание']}: снижение потерь {row['loss_reduction']:.6g}, "
            f"95% интервал для разности [{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
            f"доля выигрышных траекторий {row['win_rate']:.3g}."
        )
    lines.extend(
        [
            "",
            "Интерпретация: выигрыш должен оставаться у фактической распределительной информации и исчезать у перемешанных или искусственных рядов.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
