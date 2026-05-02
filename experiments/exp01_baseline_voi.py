from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy import fit_linear_rule, project_rule_to_information_state
from policy.linear_rules import coefficient_vector
from policy.optimize_rules import bootstrap_interval, compare_paired_losses
from state_space import LocalHANKInformationEnvironment, default_information_states, scenario_config


MAIN_INFORMATION_STATES = (
    "aggregate_only",
    "filtered_aggregates",
    "distributional",
    "full_information",
)


PAIRWISE_COMPARISONS = (
    ("filtered_aggregates", "aggregate_only", "Оценённые агрегаты против только агрегатов"),
    ("distributional", "filtered_aggregates", "Распределительная информация против оценённых агрегатов"),
    ("distributional", "aggregate_only", "Распределительная информация против только агрегатов"),
    ("full_information", "distributional", "Полная информация против распределительной информации"),
    ("full_information", "aggregate_only", "Полная информация против только агрегатов"),
)


def run_experiment(
    *,
    scenario: str,
    output_dir: Path,
    horizon: int,
    validation_count: int,
    test_count: int,
    num_candidates: int,
) -> dict[str, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(exist_ok=True)

    validation_seeds = list(range(500, 500 + validation_count))
    test_seeds = list(range(900, 900 + test_count))
    environment = LocalHANKInformationEnvironment(scenario_config(scenario, horizon=horizon))

    fitted = {}
    for index, information_state in enumerate(MAIN_INFORMATION_STATES):
        extra_candidates = []
        if information_state == "distributional" and "filtered_aggregates" in fitted:
            filtered_candidate = project_rule_to_information_state(
                fitted["filtered_aggregates"].rule,
                "distributional",
            )
            extra_candidates.append(filtered_candidate)
            for aux_offset, aux_state in enumerate(("distributional_mpc", "distributional_liquidity"), start=1):
                aux_fit = fit_linear_rule(
                    environment=environment,
                    information_state=aux_state,
                    validation_seeds=validation_seeds,
                    num_candidates=num_candidates,
                    seed=4040 + 23 * aux_offset,
                    extra_candidates=[
                        project_rule_to_information_state(fitted["filtered_aggregates"].rule, aux_state)
                    ],
                )
                extra_candidates.append(project_rule_to_information_state(aux_fit.rule, "distributional"))
        if information_state == "full_information" and "distributional" in fitted:
            extra_candidates.append(
                project_rule_to_information_state(fitted["distributional"].rule, "full_information")
            )
        fitted[information_state] = fit_linear_rule(
            environment=environment,
            information_state=information_state,
            validation_seeds=validation_seeds,
            num_candidates=num_candidates,
            seed=2027 + 17 * index,
            extra_candidates=extra_candidates,
        )

    selected_rules = _selected_rules_frame(fitted)
    test_losses = _test_losses_frame(environment, fitted, test_seeds)
    loss_summary = _loss_summary_frame(test_losses)
    gap_summary = _gap_summary_frame(loss_summary)
    pairwise = _pairwise_frame(test_losses)
    loss_decomposition = _loss_decomposition_frame(test_losses)

    selected_rules.to_csv(output_dir / "selected_rules.csv", index=False)
    test_losses.to_csv(output_dir / "test_losses.csv", index=False)
    loss_summary.to_csv(output_dir / "information_state_losses.csv", index=False)
    gap_summary.to_csv(output_dir / "gap_closure.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_value_of_information.csv", index=False)
    loss_decomposition.to_csv(output_dir / "loss_decomposition.csv", index=False)

    _write_tex_table(gap_summary, tables_dir / "table_01_gap_closure.tex")
    _write_tex_table(pairwise, tables_dir / "table_02_pairwise_value.tex")
    _write_tex_table(loss_decomposition, tables_dir / "table_03_loss_decomposition.tex")
    _write_report(
        output_dir=output_dir,
        scenario=scenario,
        horizon=horizon,
        validation_count=validation_count,
        test_count=test_count,
        num_candidates=num_candidates,
        gap_summary=gap_summary,
        pairwise=pairwise,
        loss_decomposition=loss_decomposition,
    )
    _write_spec(
        output_dir=output_dir,
        scenario=scenario,
        horizon=horizon,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
        num_candidates=num_candidates,
    )

    return {
        "selected_rules": selected_rules,
        "test_losses": test_losses,
        "loss_summary": loss_summary,
        "gap_summary": gap_summary,
        "pairwise": pairwise,
        "loss_decomposition": loss_decomposition,
    }


def _selected_rules_frame(fitted) -> pd.DataFrame:
    labels = {spec.name: spec.label for spec in default_information_states()}
    rows = []
    for information_state, result in fitted.items():
        vector = coefficient_vector(result.rule)
        rows.append(
            {
                "information_state": information_state,
                "label": labels[information_state],
                "validation_loss": result.validation_loss,
                "num_candidates": result.num_candidates,
                "intercept": result.rule.intercept,
                "lagged_rate_weight": result.rule.lagged_rate_weight,
                "feature_names": json.dumps(result.rule.spec.feature_names, ensure_ascii=False),
                "coefficients": json.dumps(list(result.rule.coefficients), ensure_ascii=False),
                "feature_scales": json.dumps(result.feature_scales, ensure_ascii=False),
                "coefficient_vector": json.dumps(vector.tolist(), ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def _test_losses_frame(environment, fitted, test_seeds: list[int]) -> pd.DataFrame:
    labels = {spec.name: spec.label for spec in default_information_states()}
    rows = []
    for information_state, result in fitted.items():
        for seed in test_seeds:
            sim = environment.simulate(
                policy=result.rule,
                information_state=information_state,
                seed=seed,
            )
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


def _loss_summary_frame(test_losses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for information_state, group in test_losses.groupby("information_state", sort=False):
        losses = group["total_loss"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_interval(losses)
        rows.append(
            {
                "information_state": information_state,
                "label": group["label"].iloc[0],
                "num_trajectories": int(losses.size),
                "mean_loss": float(np.mean(losses)),
                "std_loss": float(np.std(losses, ddof=1)) if losses.size > 1 else 0.0,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "mean_inflation_loss": float(group["inflation_loss"].mean()),
                "mean_output_loss": float(group["output_loss"].mean()),
                "mean_rate_loss": float(group["rate_loss"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _gap_summary_frame(loss_summary: pd.DataFrame) -> pd.DataFrame:
    by_state = loss_summary.set_index("information_state")
    aggregate_loss = float(by_state.loc["aggregate_only", "mean_loss"])
    full_loss = float(by_state.loc["full_information", "mean_loss"])
    full_gap = aggregate_loss - full_loss
    rows = []
    for _, row in loss_summary.iterrows():
        mean_loss = float(row["mean_loss"])
        improvement = aggregate_loss - mean_loss
        gap_to_full = mean_loss - full_loss
        share_closed = np.nan if abs(full_gap) <= 1e-14 else improvement / full_gap
        rows.append(
            {
                **row.to_dict(),
                "improvement_vs_aggregate": improvement,
                "improvement_vs_aggregate_pct": 100.0 * improvement / aggregate_loss,
                "gap_to_full_information": gap_to_full,
                "share_of_full_information_gap_closed": float(share_closed),
            }
        )
    return pd.DataFrame(rows)


def _pairwise_frame(test_losses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot = test_losses.pivot(index="seed", columns="information_state", values="total_loss")
    for left, right, label in PAIRWISE_COMPARISONS:
        comparison = compare_paired_losses(
            left_name=left,
            right_name=right,
            left_losses=pivot[left].to_numpy(dtype=float),
            right_losses=pivot[right].to_numpy(dtype=float),
            tie_eps=1e-12,
        )
        row = asdict(comparison)
        row["comparison_label"] = label
        row["loss_reduction"] = -row["mean_delta"]
        row["loss_reduction_ci_low"] = -row["ci_high"]
        row["loss_reduction_ci_high"] = -row["ci_low"]
        rows.append(row)
    return pd.DataFrame(rows)


def _loss_decomposition_frame(test_losses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for left, right, label in PAIRWISE_COMPARISONS:
        left_frame = test_losses[test_losses["information_state"] == left].set_index("seed")
        right_frame = test_losses[test_losses["information_state"] == right].set_index("seed")
        row = {
            "left": left,
            "right": right,
            "comparison_label": label,
            "num_trajectories": int(len(left_frame.index.intersection(right_frame.index))),
        }
        for component in ("inflation_loss", "output_loss", "rate_loss", "total_loss"):
            delta = left_frame.loc[right_frame.index, component].to_numpy(dtype=float) - right_frame[component].to_numpy(
                dtype=float
            )
            row[f"delta_{component}"] = float(np.mean(delta))
            row[f"reduction_{component}"] = -float(np.mean(delta))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_tex_table(frame: pd.DataFrame, path: Path) -> None:
    path.write_text(frame.to_latex(index=False, float_format="%.6f"), encoding="utf-8")


def _write_report(
    *,
    output_dir: Path,
    scenario: str,
    horizon: int,
    validation_count: int,
    test_count: int,
    num_candidates: int,
    gap_summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    loss_decomposition: pd.DataFrame,
) -> None:
    labels = gap_summary.set_index("information_state")["label"].to_dict()
    losses = gap_summary.set_index("information_state")["mean_loss"].to_dict()
    pairwise_by_left_right = pairwise.set_index(["left", "right"])
    dist_vs_agg = pairwise_by_left_right.loc[("distributional", "aggregate_only")]
    dist_vs_filtered = pairwise_by_left_right.loc[("distributional", "filtered_aggregates")]
    gap_dist = gap_summary.set_index("information_state").loc["distributional"]
    decomp = loss_decomposition.set_index(["left", "right"])
    dist_agg_decomp = decomp.loc[("distributional", "aggregate_only")]

    lines = [
        "# Эксперимент 1. Базовая ценность информации",
        "",
        f"Сценарий: `{scenario}`.",
        f"Горизонт: `{horizon}` периодов.",
        f"Валидационные траектории: `{validation_count}`.",
        f"Тестовые траектории: `{test_count}`.",
        f"Кандидатов на одно правило: `{num_candidates}`.",
        "",
        "## Средние потери",
        "",
    ]
    for state in MAIN_INFORMATION_STATES:
        lines.append(f"- {labels[state]}: `{losses[state]:.6f}`.")

    lines.extend(
        [
            "",
            "## Главный показатель",
            "",
            "Добавление распределительных статистик относительно правила только по агрегатам даёт:",
            f"- снижение потерь: `{dist_vs_agg['loss_reduction']:.6f}`;",
            f"- 95% доверительный интервал: `[{dist_vs_agg['loss_reduction_ci_low']:.6f}, {dist_vs_agg['loss_reduction_ci_high']:.6f}]`;",
            f"- доля выигрышных траекторий: `{dist_vs_agg['win_rate']:.3f}`;",
            f"- доля равных траекторий: `{dist_vs_agg['tie_rate']:.3f}`;",
            f"- доля закрытия разрыва до полной информации: `{gap_dist['share_of_full_information_gap_closed']:.3f}`.",
            "",
            "## Дополнительная проверка",
            "",
            "Переход от оценённых агрегатов к распределительному информационному состоянию даёт:",
            f"- снижение потерь: `{dist_vs_filtered['loss_reduction']:.6f}`;",
            f"- 95% доверительный интервал: `[{dist_vs_filtered['loss_reduction_ci_low']:.6f}, {dist_vs_filtered['loss_reduction_ci_high']:.6f}]`;",
            f"- доля выигрышных траекторий: `{dist_vs_filtered['win_rate']:.3f}`.",
            "",
            "## Разложение главного сравнения",
            "",
            "Для сравнения распределительного состояния с агрегатной информацией снижение потерь раскладывается так:",
            f"- инфляционный компонент: `{dist_agg_decomp['reduction_inflation_loss']:.6f}`;",
            f"- компонент разрыва выпуска: `{dist_agg_decomp['reduction_output_loss']:.6f}`;",
            f"- компонент сглаживания ставки: `{dist_agg_decomp['reduction_rate_loss']:.6f}`;",
            "",
            "Интерпретация должна быть осторожной: этот блок измеряет ценность информации в локальной редуцированной среде, а не решает полную задачу оптимальной политики в HANK.",
        ]
    )
    (output_dir / "report_exp01_baseline_voi.md").write_text("\n".join(lines), encoding="utf-8")


def _write_spec(
    *,
    output_dir: Path,
    scenario: str,
    horizon: int,
    validation_seeds: list[int],
    test_seeds: list[int],
    num_candidates: int,
) -> None:
    spec = {
        "experiment": "baseline_value_of_information",
        "scenario": scenario,
        "horizon": horizon,
        "validation_seeds": validation_seeds,
        "test_seeds": test_seeds,
        "num_candidates_per_rule": num_candidates,
        "information_states": list(MAIN_INFORMATION_STATES),
        "loss": "inflation_gap^2 + 0.5 output_gap^2 + 0.05 rate_change^2",
    }
    (output_dir / "experiment_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment 1: baseline value of distributional information.")
    parser.add_argument("--scenario", default="baseline")
    parser.add_argument("--output-dir", default="outputs/exp01_baseline_voi")
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--validation-count", type=int, default=20)
    parser.add_argument("--test-count", type=int, default=50)
    parser.add_argument("--num-candidates", type=int, default=350)
    args = parser.parse_args()

    run_experiment(
        scenario=args.scenario,
        output_dir=Path(args.output_dir),
        horizon=args.horizon,
        validation_count=args.validation_count,
        test_count=args.test_count,
        num_candidates=args.num_candidates,
    )


if __name__ == "__main__":
    main()
