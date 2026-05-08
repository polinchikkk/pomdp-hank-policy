from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import HankSSJPolicyEnvironment, PolicyLossWeights, SSJPolicyEvaluationSpec
from policy.fit_linear_rules import fit_linear_rule
from policy.linear_rules import LinearRule, coefficient_vector, rule_spec_for_information_state
from policy.optimize_rules import bootstrap_interval, compare_paired_losses


INFORMATION_STATES = (
    "aggregate_only",
    "aggregate_history",
    "filtered_aggregates",
    "observed_distribution",
    "filtered_distribution_mpc",
    "filtered_distribution_liquidity",
    "filtered_distribution_exposure",
    "filtered_distribution",
    "full_information",
)

PAIRWISE_COMPARISONS = (
    ("aggregate_history", "aggregate_only", "История агрегатов против текущих агрегатов"),
    ("filtered_aggregates", "aggregate_only", "Фильтрованные агрегаты против текущих агрегатов"),
    ("observed_distribution", "aggregate_only", "Наблюдаемые распределительные показатели против текущих агрегатов"),
    ("filtered_distribution_mpc", "filtered_aggregates", "Ценность средней MPC"),
    ("filtered_distribution_liquidity", "filtered_aggregates", "Ценность доли низколиквидных домохозяйств"),
    ("filtered_distribution_exposure", "filtered_aggregates", "Ценность процентной экспозиции"),
    ("filtered_distribution", "filtered_aggregates", "Предельная ценность распределительной информации"),
    ("filtered_distribution", "observed_distribution", "Фильтрация распределительных показателей против шумных наблюдений"),
    ("full_information", "aggregate_only", "Полная информация против текущих агрегатов"),
    ("full_information", "filtered_distribution", "Оставшийся разрыв до полной информации"),
)

STATE_LABEL_RU = {
    "aggregate_only": "Текущие агрегаты",
    "aggregate_history": "История агрегатов",
    "filtered_aggregates": "Фильтрованные агрегаты",
    "observed_distribution": "Наблюдаемые распределительные показатели",
    "filtered_distribution_mpc": "Фильтрованные агрегаты + MPC",
    "filtered_distribution_liquidity": "Фильтрованные агрегаты + низкая ликвидность",
    "filtered_distribution_exposure": "Фильтрованные агрегаты + процентная экспозиция",
    "filtered_distribution": "Фильтрованные распределительные показатели",
    "full_information": "Полная информация",
}


@dataclass(frozen=True)
class MainVOISpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    output_dir: str
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    information_states: tuple[str, ...]
    num_candidates: int
    candidate_seed: int
    loss_weights: dict[str, float]
    evaluator: dict[str, object]
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the main HANK/SSJ value-of-information experiment.")
    parser.add_argument("--information-inputs", default="outputs/ssj/information_state_inputs_long.csv")
    parser.add_argument("--hank-observables", default="outputs/ssj/hank_observables.csv")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/main_voi")
    parser.add_argument("--validation-seeds", default="900:924")
    parser.add_argument("--test-seeds", default="925:949")
    parser.add_argument("--num-candidates", type=int, default=320)
    parser.add_argument("--candidate-seed", type=int, default=2027)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_seeds = _parse_seed_range(args.validation_seeds)
    test_seeds = _parse_seed_range(args.test_seeds)
    loss_weights = PolicyLossWeights()
    environment = HankSSJPolicyEnvironment.from_files(
        information_inputs_csv=Path(args.information_inputs),
        hank_observables_csv=Path(args.hank_observables),
        jacobians_npz=Path(args.jacobians),
        loss_weights=loss_weights,
    )

    fitted: dict[str, LinearRule] = {}
    rule_rows: list[dict[str, object]] = []
    for index, information_state in enumerate(INFORMATION_STATES):
        extra_candidates = _supervised_candidates(
            environment=environment,
            information_state=information_state,
            validation_seeds=validation_seeds,
        )
        fit = fit_linear_rule(
            environment=environment,
            information_state=information_state,
            validation_seeds=validation_seeds,
            num_candidates=args.num_candidates,
            seed=args.candidate_seed + index,
            extra_candidates=extra_candidates,
        )
        fitted[information_state] = fit.rule
        rule_rows.extend(_rule_rows(fit.rule, fit.validation_loss, fit.num_candidates, fit.feature_scales))

    fitted_rules = pd.DataFrame(rule_rows)
    fitted_rules.to_csv(output_dir / "fitted_policy_rules.csv", index=False)

    losses = _evaluate_rules(environment, fitted, test_seeds)
    losses.to_csv(output_dir / "trajectory_losses.csv", index=False)

    summary = _summary_table(losses)
    summary.to_csv(output_dir / "main_voi_summary.csv", index=False)
    _write_latex(summary, output_dir / "table_main_voi_summary.tex")

    pairwise = _pairwise_table(losses)
    pairwise.to_csv(output_dir / "pairwise_value_of_information.csv", index=False)
    _write_latex(pairwise, output_dir / "table_pairwise_value_of_information.tex")

    gap = _gap_table(summary)
    gap.to_csv(output_dir / "full_information_gap.csv", index=False)
    _write_latex(gap, output_dir / "table_full_information_gap.tex")

    spec = MainVOISpec(
        information_inputs=args.information_inputs,
        hank_observables=args.hank_observables,
        jacobians=args.jacobians,
        output_dir=args.output_dir,
        validation_seeds=tuple(validation_seeds),
        test_seeds=tuple(test_seeds),
        information_states=INFORMATION_STATES,
        num_candidates=int(args.num_candidates),
        candidate_seed=int(args.candidate_seed),
        loss_weights=asdict(loss_weights),
        evaluator=asdict(
            SSJPolicyEvaluationSpec(
                information_inputs=args.information_inputs,
                hank_observables=args.hank_observables,
                jacobians=args.jacobians,
                note=(
                    "Правила оцениваются на HANK/SSJ-траекториях. "
                    "Контрфактическое влияние альтернативной ставки рассчитывается через локальную SSJ-проекцию."
                ),
            )
        ),
        note=(
            "Это основной первый эксперимент новой линии: все информационные состояния сравниваются "
            "при одном классе настраиваемых линейных правил. Старое вручную заданное состояние не используется."
        ),
    )
    (output_dir / "main_voi_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(summary, pairwise, gap, output_dir / "report_main_voi.md")

    print(f"Wrote {output_dir / 'main_voi_summary.csv'}")
    print(f"Wrote {output_dir / 'pairwise_value_of_information.csv'}")
    print(f"Wrote {output_dir / 'full_information_gap.csv'}")


def _parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


def _evaluate_rules(
    environment: HankSSJPolicyEnvironment,
    fitted: dict[str, LinearRule],
    test_seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in environment.scenarios:
        for seed in test_seeds:
            for state, rule in fitted.items():
                loss = environment.simulate_scenario(
                    policy=rule,
                    information_state=state,
                    scenario=scenario,
                    seed=seed,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "observation_seed": int(seed),
                        "information_state": state,
                        "information_state_ru": STATE_LABEL_RU[state],
                        **asdict(loss),
                    }
                )
    return pd.DataFrame(rows)


def _supervised_candidates(
    *,
    environment: HankSSJPolicyEnvironment,
    information_state: str,
    validation_seeds: list[int],
    ridge: float = 1e-8,
) -> list[LinearRule]:
    """Build strong linear candidates by approximating the local SSJ-optimal rate path."""

    spec = rule_spec_for_information_state(information_state)
    candidates: list[LinearRule] = []
    for lagged_rate_weight in (0.0, 0.35, 0.60, 0.80, 0.90):
        x_blocks: list[np.ndarray] = []
        y_blocks: list[np.ndarray] = []
        for scenario in environment.scenarios:
            target = environment.optimal_rate_path(scenario=scenario)
            target_lag = np.r_[0.0, target[:-1]]
            transformed_target = target - lagged_rate_weight * target_lag
            for seed in validation_seeds:
                features = environment.feature_matrix(
                    scenario=scenario,
                    information_state=information_state,
                    seed=seed,
                    feature_names=spec.feature_names,
                )
                periods = min(features.shape[0], transformed_target.size)
                x_blocks.append(features[:periods])
                y_blocks.append(transformed_target[:periods])
        x = np.vstack(x_blocks)
        y = np.concatenate(y_blocks)
        design = np.column_stack([np.ones(x.shape[0]), x])
        penalty = ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        candidates.append(
            LinearRule(
                spec=spec,
                intercept=float(beta[0]),
                coefficients=tuple(float(value) for value in beta[1:]),
                lagged_rate_weight=float(lagged_rate_weight),
            )
        )
    return candidates


def _summary_table(losses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, frame in _scenario_groups(losses):
        for state, state_frame in frame.groupby("information_state", sort=False):
            values = state_frame["total_loss"].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_interval(values)
            rows.append(
                {
                    "scenario": scenario,
                    "information_state": state,
                    "information_state_ru": STATE_LABEL_RU[state],
                    "num_trajectories": int(values.size),
                    "mean_loss": float(np.mean(values)),
                    "std_loss": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "inflation_loss": float(state_frame["inflation_loss"].mean()),
                    "output_gap_loss": float(state_frame["output_gap_loss"].mean()),
                    "consumption_loss": float(state_frame["consumption_loss"].mean()),
                    "rate_smoothing_loss": float(state_frame["rate_smoothing_loss"].mean()),
                    "stability_penalty": float(state_frame["stability_penalty"].mean()),
                }
            )
    return pd.DataFrame(rows)


def _pairwise_table(losses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, frame in _scenario_groups(losses):
        pivot = frame.pivot_table(
            index=["scenario", "observation_seed"],
            columns="information_state",
            values="total_loss",
            aggfunc="first",
        )
        for left, right, label in PAIRWISE_COMPARISONS:
            comparison = compare_paired_losses(
                left_name=left,
                right_name=right,
                left_losses=pivot[left].to_numpy(dtype=float),
                right_losses=pivot[right].to_numpy(dtype=float),
                tie_eps=1e-10,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "comparison": f"{left}_minus_{right}",
                    "comparison_ru": label,
                    "left": left,
                    "right": right,
                    "left_ru": STATE_LABEL_RU[left],
                    "right_ru": STATE_LABEL_RU[right],
                    "num_trajectories": comparison.num_trajectories,
                    "mean_delta": comparison.mean_delta,
                    "loss_reduction": -comparison.mean_delta,
                    "ci_low": comparison.ci_low,
                    "ci_high": comparison.ci_high,
                    "win_rate": comparison.win_rate,
                    "tie_rate": comparison.tie_rate,
                    "loss_rate": comparison.loss_rate,
                }
            )
    return pd.DataFrame(rows)


def _gap_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, frame in summary.groupby("scenario", sort=False):
        by_state = frame.set_index("information_state")
        aggregate = float(by_state.loc["aggregate_only", "mean_loss"])
        full = float(by_state.loc["full_information", "mean_loss"])
        denominator = aggregate - full
        for state, row in by_state.iterrows():
            mean_loss = float(row["mean_loss"])
            improvement = aggregate - mean_loss
            gap = mean_loss - full
            rows.append(
                {
                    "scenario": scenario,
                    "information_state": state,
                    "information_state_ru": STATE_LABEL_RU[state],
                    "mean_loss": mean_loss,
                    "improvement_vs_aggregate": improvement,
                    "gap_to_full_information": gap,
                    "share_of_gap_closed": improvement / denominator if abs(denominator) > 1e-14 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _scenario_groups(losses: pd.DataFrame):
    for scenario, frame in losses.groupby("scenario", sort=False):
        yield scenario, frame
    yield "all", losses


def _rule_rows(
    rule: LinearRule,
    validation_loss: float,
    num_candidates: int,
    feature_scales: dict[str, float],
) -> list[dict[str, object]]:
    rows = [
        {
            "information_state": rule.spec.information_state,
            "information_state_ru": STATE_LABEL_RU[rule.spec.information_state],
            "term": "intercept",
            "coefficient": rule.intercept,
            "validation_loss": validation_loss,
            "num_candidates": int(num_candidates),
            "feature_scale": np.nan,
        },
        {
            "information_state": rule.spec.information_state,
            "information_state_ru": STATE_LABEL_RU[rule.spec.information_state],
            "term": "lagged_rate",
            "coefficient": rule.lagged_rate_weight,
            "validation_loss": validation_loss,
            "num_candidates": int(num_candidates),
            "feature_scale": np.nan,
        },
    ]
    for name, coefficient in zip(rule.spec.feature_names, rule.coefficients):
        rows.append(
            {
                "information_state": rule.spec.information_state,
                "information_state_ru": STATE_LABEL_RU[rule.spec.information_state],
                "term": name,
                "coefficient": coefficient,
                "validation_loss": validation_loss,
                "num_candidates": int(num_candidates),
                "feature_scale": feature_scales.get(name, np.nan),
            }
        )
    rows[0]["coefficient_vector"] = coefficient_vector(rule).tolist()
    return rows


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    for column in numeric:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_report(summary: pd.DataFrame, pairwise: pd.DataFrame, gap: pd.DataFrame, path: Path) -> None:
    overall_summary = summary[summary["scenario"] == "all"].sort_values("mean_loss")
    main_pair = pairwise[
        (pairwise["scenario"] == "all")
        & (pairwise["comparison"] == "filtered_distribution_minus_filtered_aggregates")
    ]
    gap_overall = gap[gap["scenario"] == "all"].sort_values("mean_loss")

    lines = [
        "# Основной эксперимент по ценности информации",
        "",
        "Эксперимент сравнивает информационные наборы на HANK/SSJ-траекториях. ",
        "Все правила имеют один и тот же простой класс: настраиваемое линейное правило ставки.",
        "",
        "## Лучшие информационные наборы по средним потерям",
        "",
    ]
    for _, row in overall_summary.head(6).iterrows():
        lines.append(f"- {row['information_state_ru']}: {row['mean_loss']:.6g}")

    if not main_pair.empty:
        row = main_pair.iloc[0]
        lines.extend(
            [
                "",
                "## Предельная ценность распределительной информации",
                "",
                (
                    f"Разность потерь между фильтрованным распределительным набором и фильтрованными агрегатами: "
                    f"{row['mean_delta']:.6g}. Отрицательное значение означает снижение потерь."
                ),
                (
                    f"Доли на тестовых траекториях: выигрыш {row['win_rate']:.2f}, "
                    f"совпадение {row['tie_rate']:.2f}, ухудшение {row['loss_rate']:.2f}."
                ),
            ]
        )

    lines.extend(["", "## Разрыв до полной информации", ""])
    for _, row in gap_overall.iterrows():
        lines.append(
            f"- {row['information_state_ru']}: разрыв {row['gap_to_full_information']:.6g}, "
            f"закрытая доля {row['share_of_gap_closed']:.3g}"
        )

    lines.extend(
        [
            "",
            "## Важная оговорка",
            "",
            (
                "Это локальная SSJ-проекция альтернативных траекторий ставки, а не глобальное решение "
                "полной задачи оптимальной политики. Старое вручную заданное редуцированное состояние здесь не используется."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
