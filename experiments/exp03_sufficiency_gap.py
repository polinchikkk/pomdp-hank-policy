from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_sufficiency_gap_summary(
    *,
    source_dir: Path,
    output_dir: Path,
    almost_sufficient_threshold: float = 0.05,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for gap_file in sorted(source_dir.glob("*/gap_closure.csv")):
        scenario = gap_file.parent.name
        gap = pd.read_csv(gap_file).set_index("information_state")
        aggregate_loss = float(gap.loc["aggregate_only", "mean_loss"])
        distributional_loss = float(gap.loc["distributional", "mean_loss"])
        full_loss = float(gap.loc["full_information", "mean_loss"])
        sufficiency_gap = aggregate_loss - full_loss
        distributional_gap = distributional_loss - full_loss
        gap_pct = sufficiency_gap / aggregate_loss if abs(aggregate_loss) > 1e-14 else float("nan")
        rows.append(
            {
                "scenario": scenario,
                "aggregate_loss": aggregate_loss,
                "distributional_loss": distributional_loss,
                "full_information_loss": full_loss,
                "sufficiency_gap": sufficiency_gap,
                "sufficiency_gap_pct_of_aggregate": gap_pct,
                "distributional_gap_to_full": distributional_gap,
                "share_of_gap_closed_by_distribution": float(
                    gap.loc["distributional", "share_of_full_information_gap_closed"]
                ),
                "aggregate_information_almost_sufficient": bool(gap_pct <= almost_sufficient_threshold),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "sufficiency_gap_summary.csv", index=False)
    (output_dir / "table_sufficiency_gap.tex").write_text(
        summary.to_latex(index=False, float_format="%.6f"),
        encoding="utf-8",
    )
    _write_report(output_dir, summary, almost_sufficient_threshold)
    return summary


def _write_report(output_dir: Path, summary: pd.DataFrame, threshold: float) -> None:
    lines = [
        "# Эксперимент 3. Почти достаточность агрегатной информации",
        "",
        f"Агрегатная информация считается почти достаточной, если разрыв до полной информации не превышает `{100 * threshold:.1f}%` от потерь агрегатного правила.",
        "",
    ]
    for _, row in summary.iterrows():
        status = "почти достаточна" if row["aggregate_information_almost_sufficient"] else "не является почти достаточной"
        lines.append(
            f"- `{row['scenario']}`: разрыв `{row['sufficiency_gap']:.6f}` "
            f"({100 * row['sufficiency_gap_pct_of_aggregate']:.2f}% от агрегатных потерь), "
            f"агрегатная информация {status}; распределительный блок закрывает "
            f"`{row['share_of_gap_closed_by_distribution']:.3f}` разрыва."
        )
    (output_dir / "report_exp03_sufficiency_gap.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sufficiency-gap summary from scenario results.")
    parser.add_argument("--source-dir", default="outputs/exp02_distributional_value")
    parser.add_argument("--output-dir", default="outputs/exp03_sufficiency_gap")
    parser.add_argument("--almost-sufficient-threshold", type=float, default=0.05)
    args = parser.parse_args()

    build_sufficiency_gap_summary(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        almost_sufficient_threshold=args.almost_sufficient_threshold,
    )


if __name__ == "__main__":
    main()
