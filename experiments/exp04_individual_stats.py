from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy import fit_linear_rule, project_rule_to_information_state
from policy.optimize_rules import compare_paired_losses
from state_space import LocalHANKInformationEnvironment, default_information_states, scenario_config


INFORMATION_STATES = (
    "filtered_aggregates",
    "distributional_mpc",
    "distributional_liquidity",
    "distributional",
)


def run_individual_stats_experiment(
    *,
    scenario: str,
    output_dir: Path,
    horizon: int,
    validation_count: int,
    test_count: int,
    num_candidates: int,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_seeds = list(range(500, 500 + validation_count))
    test_seeds = list(range(900, 900 + test_count))
    environment = LocalHANKInformationEnvironment(scenario_config(scenario, horizon=horizon))

    fitted = {}
    fitted["filtered_aggregates"] = fit_linear_rule(
        environment=environment,
        information_state="filtered_aggregates",
        validation_seeds=validation_seeds,
        num_candidates=num_candidates,
        seed=3101,
    )
    for offset, information_state in enumerate(
        ("distributional_mpc", "distributional_liquidity"),
        start=1,
    ):
        fitted[information_state] = fit_linear_rule(
            environment=environment,
            information_state=information_state,
            validation_seeds=validation_seeds,
            num_candidates=num_candidates,
            seed=3101 + 19 * offset,
            extra_candidates=[
                project_rule_to_information_state(
                    fitted["filtered_aggregates"].rule,
                    information_state,
                )
            ],
        )
    fitted["distributional"] = fit_linear_rule(
        environment=environment,
        information_state="distributional",
        validation_seeds=validation_seeds,
        num_candidates=num_candidates,
        seed=3101 + 19 * 3,
        extra_candidates=[
            project_rule_to_information_state(fitted["filtered_aggregates"].rule, "distributional"),
            project_rule_to_information_state(fitted["distributional_mpc"].rule, "distributional"),
            project_rule_to_information_state(fitted["distributional_liquidity"].rule, "distributional"),
        ],
    )

    losses = _losses_frame(environment, fitted, test_seeds)
    pairwise = _pairwise_frame(losses)
    selected = _selected_frame(fitted)

    losses.to_csv(output_dir / "test_losses.csv", index=False)
    pairwise.to_csv(output_dir / "individual_stat_value.csv", index=False)
    selected.to_csv(output_dir / "selected_rules.csv", index=False)
    (output_dir / "table_individual_stat_value.tex").write_text(
        pairwise.to_latex(index=False, float_format="%.6f"),
        encoding="utf-8",
    )
    _write_report(output_dir, scenario, pairwise)
    return {"losses": losses, "pairwise": pairwise, "selected": selected}


def _losses_frame(environment, fitted, test_seeds: list[int]) -> pd.DataFrame:
    labels = {spec.name: spec.label for spec in default_information_states()}
    rows = []
    for information_state, result in fitted.items():
        for seed in test_seeds:
            sim = environment.simulate(policy=result.rule, information_state=information_state, seed=seed)
            rows.append(
                {
                    "seed": seed,
                    "information_state": information_state,
                    "label": labels[information_state],
                    "total_loss": sim.total_loss,
                    "inflation_loss": sim.inflation_loss,
                    "output_loss": sim.output_loss,
                    "rate_loss": sim.rate_loss,
                }
            )
    return pd.DataFrame(rows)


def _pairwise_frame(losses: pd.DataFrame) -> pd.DataFrame:
    pivot = losses.pivot(index="seed", columns="information_state", values="total_loss")
    rows = []
    for left, label in (
        ("distributional_mpc", "Средняя MPC против оценённых агрегатов"),
        ("distributional_liquidity", "Доля низколиквидных против оценённых агрегатов"),
        ("distributional", "MPC и доля низколиквидных против оценённых агрегатов"),
    ):
        comparison = compare_paired_losses(
            left_name=left,
            right_name="filtered_aggregates",
            left_losses=pivot[left].to_numpy(dtype=float),
            right_losses=pivot["filtered_aggregates"].to_numpy(dtype=float),
            tie_eps=1e-12,
        )
        row = asdict(comparison)
        row["comparison_label"] = label
        row["loss_reduction"] = -row["mean_delta"]
        row["loss_reduction_ci_low"] = -row["ci_high"]
        row["loss_reduction_ci_high"] = -row["ci_low"]
        rows.append(row)
    return pd.DataFrame(rows)


def _selected_frame(fitted) -> pd.DataFrame:
    rows = []
    for information_state, result in fitted.items():
        rows.append(
            {
                "information_state": information_state,
                "validation_loss": result.validation_loss,
                "num_candidates": result.num_candidates,
                "feature_names": list(result.rule.spec.feature_names),
                "coefficients": list(result.rule.coefficients),
                "lagged_rate_weight": result.rule.lagged_rate_weight,
            }
        )
    return pd.DataFrame(rows)


def _write_report(output_dir: Path, scenario: str, pairwise: pd.DataFrame) -> None:
    lines = [
        "# Эксперимент 4. Отдельная роль распределительных статистик",
        "",
        f"Сценарий: `{scenario}`.",
        "",
    ]
    for _, row in pairwise.iterrows():
        lines.append(
            f"- {row['comparison_label']}: снижение потерь `{row['loss_reduction']:.6f}`, "
            f"95% ДИ `[{row['loss_reduction_ci_low']:.6f}, {row['loss_reduction_ci_high']:.6f}]`, "
            f"доля выигрышных траекторий `{row['win_rate']:.3f}`."
        )
    lines.append("")
    lines.append("Если снижение потерь близко к нулю, соответствующая статистика не добавляет самостоятельной ценности сверх оценённых агрегатов в данном классе правил.")
    (output_dir / "report_exp04_individual_stats.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run individual distributional statistics experiment.")
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--output-dir", default="outputs/exp04_individual_stats")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--validation-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--num-candidates", type=int, default=350)
    args = parser.parse_args()

    run_individual_stats_experiment(
        scenario=args.scenario,
        output_dir=Path(args.output_dir),
        horizon=args.horizon,
        validation_count=args.validation_count,
        test_count=args.test_count,
        num_candidates=args.num_candidates,
    )


if __name__ == "__main__":
    main()
