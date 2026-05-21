from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.inference import cluster_bootstrap_ci, paired_bootstrap_ci, sign_flip_test  # noqa: E402
from policy.linear_rules import LinearRule, rule_spec_for_information_state  # noqa: E402
from policy.lqg_oracle import (  # noqa: E402
    LQGLossWeights,
    LinearControlSystem,
    load_linear_control_system,
    load_state_space_filter_spec,
    simulate_lqg_path,
    simulate_lqr_full_state_path,
    simulate_simple_filtered_rule_path,
    solve_finite_horizon_lqr_with_rate_smoothing,
    solve_lqr_with_rate_smoothing,
)


CONTROLLER_LABEL_RU = {
    "simple_filtered_aggregates": "Простое правило: фильтрованные агрегаты",
    "simple_filtered_distribution": "Простое правило: агрегаты + распр. сигналы",
    "lqg_aggregate_observations": "LQG: агрегатные наблюдения",
    "lqg_distribution_observations": "LQG: агрегатные и распределительные наблюдения",
    "lqr_full_state": "LQR: полная информация",
}

PAIRWISE_COMPARISONS = (
    (
        "simple_filtered_distribution",
        "simple_filtered_aggregates",
        "Предельная ценность распределительной информации в простом правиле",
    ),
    (
        "lqg_distribution_observations",
        "lqg_aggregate_observations",
        "Предельная ценность распределительных наблюдений в LQG",
    ),
    (
        "lqg_aggregate_observations",
        "simple_filtered_aggregates",
        "Расстояние простого агрегатного правила до LQG",
    ),
    (
        "lqg_distribution_observations",
        "simple_filtered_distribution",
        "Расстояние простого распределительного правила до LQG",
    ),
    (
        "lqr_full_state",
        "lqg_distribution_observations",
        "Оставшийся разрыв LQG до полной информации",
    ),
)


@dataclass(frozen=True)
class LQGOracleExperimentSpec:
    state_space_spec: str
    hank_observables: str
    hank_observations: str
    jacobians: str
    fitted_policy_rules: str
    output_dir: str
    test_observation_seeds: tuple[int, ...]
    policy_input_source: str
    max_abs_rate: float | None
    max_abs_rate_change: float | None
    loss_weights: dict[str, float]
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LQG/LQR information checks for the linear state-space task.")
    parser.add_argument("--state-space-spec", default="outputs/ssj/stochastic/large_sample/test/state_space/state_space_spec.json")
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/large_sample/test/hank_observables.csv")
    parser.add_argument("--hank-observations", default="outputs/ssj/stochastic/large_sample/test/hank_observations.csv")
    parser.add_argument(
        "--jacobians",
        default="outputs/ssj/stochastic/closed_loop_distributional_ssj/jacobians_distributional_augmented.npz",
    )
    parser.add_argument("--fallback-jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--fitted-policy-rules", default="outputs/ssj/stochastic/large_sample/fitted_policy_rules.csv")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/lqg_oracle")
    parser.add_argument("--test-seeds", default="960:999")
    parser.add_argument("--policy-input-source", default="transition_regression", choices=("ssj_one_step", "transition_regression"))
    parser.add_argument("--riccati-horizon", default="finite", choices=("finite", "infinite"))
    parser.add_argument("--max-abs-rate", type=float, default=0.0)
    parser.add_argument("--max-abs-rate-change", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = ROOT / "article" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    jacobians = Path(args.jacobians)
    if not jacobians.exists():
        jacobians = Path(args.fallback_jacobians)
    observables = pd.read_csv(args.hank_observables)
    observations = pd.read_csv(args.hank_observations)
    test_seeds = _parse_seed_range(args.test_seeds)
    weights = LQGLossWeights()

    system = load_linear_control_system(
        state_space_spec_json=Path(args.state_space_spec),
        jacobians_npz=jacobians,
        observables=observables,
        policy_input_source=args.policy_input_source,
    )
    aggregate_spec = load_state_space_filter_spec(Path(args.state_space_spec), "filtered_aggregates")
    distribution_spec = load_state_space_filter_spec(Path(args.state_space_spec), "filtered_distribution")
    horizon = int(observables.groupby("scenario")["period"].nunique().max())
    if args.riccati_horizon == "finite":
        lqr_solution = solve_finite_horizon_lqr_with_rate_smoothing(system=system, horizon=horizon, weights=weights)
    else:
        lqr_solution = solve_lqr_with_rate_smoothing(system=system, weights=weights)
    simple_agg = _load_fitted_rule(Path(args.fitted_policy_rules), "filtered_aggregates")
    simple_dist = _load_fitted_rule(Path(args.fitted_policy_rules), "filtered_distribution")

    max_abs_rate = None if args.max_abs_rate <= 0 else float(args.max_abs_rate)
    max_abs_rate_change = None if args.max_abs_rate_change <= 0 else float(args.max_abs_rate_change)

    losses = _evaluate_controllers(
        system=system,
        aggregate_spec=aggregate_spec,
        distribution_spec=distribution_spec,
        lqr_solution=lqr_solution,
        simple_agg=simple_agg,
        simple_dist=simple_dist,
        observables=observables,
        observations=observations,
        test_seeds=test_seeds,
        weights=weights,
        max_abs_rate=max_abs_rate,
        max_abs_rate_change=max_abs_rate_change,
    )
    losses.to_csv(output_dir / "lqg_oracle_trajectory_losses.csv", index=False)

    summary = _summary_table(losses)
    summary.to_csv(output_dir / "lqg_oracle_summary.csv", index=False)
    summary.to_latex(output_dir / "table_lqg_oracle_summary.tex", index=False, escape=False)

    pairwise = _pairwise_table(losses)
    pairwise.to_csv(output_dir / "lqg_oracle_pairwise.csv", index=False)
    pairwise.to_latex(output_dir / "table_lqg_oracle_pairwise.tex", index=False, escape=False)

    gains = _gain_table(system, lqr_solution)
    gains.to_csv(output_dir / "lqg_oracle_gains.csv", index=False)

    _plot_summary(summary, figures_dir / "fig_lqg_oracle_comparison.pdf")
    _plot_summary(summary, figures_dir / "fig_lqg_oracle_ru.pdf")

    spec = LQGOracleExperimentSpec(
        state_space_spec=args.state_space_spec,
        hank_observables=args.hank_observables,
        hank_observations=args.hank_observations,
        jacobians=str(jacobians),
        fitted_policy_rules=args.fitted_policy_rules,
        output_dir=args.output_dir,
        test_observation_seeds=tuple(test_seeds),
        policy_input_source=args.policy_input_source,
        max_abs_rate=max_abs_rate,
        max_abs_rate_change=max_abs_rate_change,
        loss_weights=asdict(weights),
        note=(
            "LQG/LQR строится для той же совместной линейной state-space спецификации, что и фильтр Калмана. "
            "Простые правила не переоцениваются: это правила из основного эксперимента, применённые в той же "
            "линейной системе. Сравнение показывает расстояние ограниченных правил до оптимального линейного "
            "регулятора и проверяет, сохраняется ли ценность распределительных наблюдений в LQG."
        ),
    )
    spec_payload = asdict(spec)
    spec_payload["riccati_horizon"] = args.riccati_horizon
    spec_payload["horizon_periods"] = horizon
    (output_dir / "lqg_oracle_spec.json").write_text(json.dumps(spec_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(
        summary=summary,
        pairwise=pairwise,
        gains=gains,
        lqr_solution=lqr_solution,
        system=system,
        output_path=output_dir / "report_lqg_oracle.md",
    )
    print(f"Wrote {output_dir / 'lqg_oracle_summary.csv'}")
    print(f"Wrote {output_dir / 'lqg_oracle_pairwise.csv'}")
    print(f"Wrote {figures_dir / 'fig_lqg_oracle_comparison.pdf'}")
    print(f"Wrote {figures_dir / 'fig_lqg_oracle_ru.pdf'}")


def _evaluate_controllers(
    *,
    system: LinearControlSystem,
    aggregate_spec,
    distribution_spec,
    lqr_solution,
    simple_agg: LinearRule,
    simple_dist: LinearRule,
    observables: pd.DataFrame,
    observations: pd.DataFrame,
    test_seeds: list[int],
    weights: LQGLossWeights,
    max_abs_rate: float | None,
    max_abs_rate_change: float | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario, base_path in observables.sort_values(["scenario", "period"]).groupby("scenario", sort=False):
        base_path = base_path.sort_values("period").reset_index(drop=True)
        for observation_seed in test_seeds:
            obs_path = observations[
                (observations["scenario"] == scenario)
                & (observations["observation_seed"] == observation_seed)
            ].sort_values("period").reset_index(drop=True)
            if obs_path.empty:
                continue
            controller_results = {
                "simple_filtered_aggregates": simulate_simple_filtered_rule_path(
                    system=system,
                    observation_spec=aggregate_spec,
                    rule=simple_agg,
                    base_path=base_path,
                    observations=obs_path,
                    weights=weights,
                    max_abs_rate=max_abs_rate,
                    max_abs_rate_change=max_abs_rate_change,
                ),
                "simple_filtered_distribution": simulate_simple_filtered_rule_path(
                    system=system,
                    observation_spec=distribution_spec,
                    rule=simple_dist,
                    base_path=base_path,
                    observations=obs_path,
                    weights=weights,
                    max_abs_rate=max_abs_rate,
                    max_abs_rate_change=max_abs_rate_change,
                ),
                "lqg_aggregate_observations": simulate_lqg_path(
                    system=system,
                    observation_spec=aggregate_spec,
                    lqr_solution=lqr_solution,
                    base_path=base_path,
                    observations=obs_path,
                    weights=weights,
                    max_abs_rate=max_abs_rate,
                    max_abs_rate_change=max_abs_rate_change,
                ),
                "lqg_distribution_observations": simulate_lqg_path(
                    system=system,
                    observation_spec=distribution_spec,
                    lqr_solution=lqr_solution,
                    base_path=base_path,
                    observations=obs_path,
                    weights=weights,
                    max_abs_rate=max_abs_rate,
                    max_abs_rate_change=max_abs_rate_change,
                ),
                "lqr_full_state": simulate_lqr_full_state_path(
                    system=system,
                    lqr_solution=lqr_solution,
                    base_path=base_path,
                    weights=weights,
                    max_abs_rate=max_abs_rate,
                    max_abs_rate_change=max_abs_rate_change,
                ),
            }
            for controller, (loss, rates) in controller_results.items():
                rows.append(
                    {
                        "scenario": scenario,
                        "observation_seed": int(observation_seed),
                        "controller": controller,
                        "controller_ru": CONTROLLER_LABEL_RU[controller],
                        "total_loss": loss.total_loss,
                        "inflation_loss": loss.inflation_loss,
                        "output_gap_loss": loss.output_gap_loss,
                        "consumption_loss": loss.consumption_loss,
                        "rate_smoothing_loss": loss.rate_smoothing_loss,
                        "mean_abs_rate": float(np.mean(np.abs(rates))),
                        "max_abs_rate": float(np.max(np.abs(rates))),
                    }
                )
    return pd.DataFrame(rows)


def _summary_table(losses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for controller, group in losses.groupby("controller", sort=False):
        values = group["total_loss"].to_numpy(dtype=float)
        ci_low, ci_high = paired_bootstrap_ci(values, seed=2061, n_boot=4_000)
        rows.append(
            {
                "controller": controller,
                "controller_ru": CONTROLLER_LABEL_RU[controller],
                "num_trajectories": int(len(group)),
                "mean_loss": float(values.mean()),
                "median_loss": float(np.median(values)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "inflation_loss": float(group["inflation_loss"].mean()),
                "output_gap_loss": float(group["output_gap_loss"].mean()),
                "consumption_loss": float(group["consumption_loss"].mean()),
                "rate_smoothing_loss": float(group["rate_smoothing_loss"].mean()),
                "mean_abs_rate": float(group["mean_abs_rate"].mean()),
                "max_abs_rate": float(group["max_abs_rate"].max()),
            }
        )
    order = {name: index for index, name in enumerate(CONTROLLER_LABEL_RU)}
    return pd.DataFrame(rows).sort_values("controller", key=lambda col: col.map(order)).reset_index(drop=True)


def _pairwise_table(losses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    wide = losses.pivot_table(
        index=["scenario", "observation_seed"],
        columns="controller",
        values="total_loss",
        aggfunc="first",
    ).reset_index()
    for left, right, comparison_ru in PAIRWISE_COMPARISONS:
        pair = wide.dropna(subset=[left, right]).copy()
        delta = pair[left].to_numpy(dtype=float) - pair[right].to_numpy(dtype=float)
        ci_low, ci_high = paired_bootstrap_ci(delta, seed=2062, n_boot=6_000)
        cluster_low, cluster_high = cluster_bootstrap_ci(
            delta,
            pair["scenario"].to_numpy(),
            seed=2063,
            n_boot=6_000,
        )
        rows.append(
            {
                "comparison": f"{left}_minus_{right}",
                "comparison_ru": comparison_ru,
                "left": left,
                "right": right,
                "left_ru": CONTROLLER_LABEL_RU[left],
                "right_ru": CONTROLLER_LABEL_RU[right],
                "num_trajectories": int(delta.size),
                "num_shock_paths": int(pair["scenario"].nunique()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "loss_reduction": float(-delta.mean()),
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "cluster_ci_low": cluster_low,
                "cluster_ci_high": cluster_high,
                "sign_flip_p_value": sign_flip_test(delta, seed=2064, n_perm=8_000),
                "win_rate": float(np.mean(delta < -1e-12)),
                "tie_rate": float(np.mean(np.abs(delta) <= 1e-12)),
                "loss_rate": float(np.mean(delta > 1e-12)),
            }
        )
    return pd.DataFrame(rows)


def _gain_table(system: LinearControlSystem, lqr_solution) -> pd.DataFrame:
    terms = [*system.state_names, "lagged_rate"]
    rows: list[dict[str, object]] = []
    if lqr_solution.K_path is None:
        for term, gain in zip(terms, lqr_solution.K.reshape(-1)):
            rows.append(
                {
                    "period": "infinite_horizon",
                    "term": term,
                    "riccati_K": float(gain),
                    "policy_response_minus_K": float(-gain),
                }
            )
    else:
        for period in range(lqr_solution.K_path.shape[0]):
            for term, gain in zip(terms, lqr_solution.K_path[period].reshape(-1)):
                rows.append(
                    {
                        "period": int(period),
                        "term": term,
                        "riccati_K": float(gain),
                        "policy_response_minus_K": float(-gain),
                    }
                )
    return pd.DataFrame(rows)


def _load_fitted_rule(path: Path, information_state: str) -> LinearRule:
    rules = pd.read_csv(path)
    rows = rules[rules["information_state"] == information_state]
    if rows.empty:
        raise ValueError(f"Fitted rule table {path} does not contain {information_state}.")
    coefficient_by_term = {
        str(row["term"]): float(row["coefficient"])
        for _, row in rows.iterrows()
    }
    spec = rule_spec_for_information_state(information_state)
    return LinearRule(
        spec=spec,
        intercept=float(coefficient_by_term.get("intercept", 0.0)),
        coefficients=tuple(float(coefficient_by_term.get(name, 0.0)) for name in spec.feature_names),
        lagged_rate_weight=float(coefficient_by_term.get("lagged_rate", 0.0)),
    )


def _plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    ordered = summary.copy()
    labels = [
        "Простое: агрегаты",
        "Простое: + распр. сигналы",
        "LQG: агрегаты",
        "LQG: + распр. сигналы",
        "LQR: полная инф.",
    ]
    colors = ["#7a8fa6", "#365f8c", "#e2a447", "#b45f2a", "#355e3b"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    y_pos = np.arange(len(ordered))
    mean = ordered["mean_loss"].to_numpy(dtype=float) * 1e4
    low = ordered["ci_low"].to_numpy(dtype=float) * 1e4
    high = ordered["ci_high"].to_numpy(dtype=float) * 1e4
    yerr = np.vstack(
        [
            mean - low,
            high - mean,
        ]
    )
    ax.barh(y_pos, mean, color=colors, edgecolor="#202020", linewidth=0.5)
    ax.errorbar(mean, y_pos, xerr=yerr, fmt="none", ecolor="#202020", capsize=3, linewidth=1.0)
    ax.set_yticks(y_pos, labels, fontsize=12)
    ax.set_xlabel("Средние потери × 10⁴", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.96, bottom=0.18)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    *,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    gains: pd.DataFrame,
    lqr_solution,
    system: LinearControlSystem,
    output_path: Path,
) -> None:
    lqg_value = pairwise[pairwise["comparison"] == "lqg_distribution_observations_minus_lqg_aggregate_observations"].iloc[0]
    simple_value = pairwise[pairwise["comparison"] == "simple_filtered_distribution_minus_simple_filtered_aggregates"].iloc[0]
    lines = [
        "# LQG/LQR information check",
        "",
        "Проверка строит верхний ориентир для той же совместной линейной state-space задачи, которая используется в совместном фильтре Калмана.",
        "",
        "## Главный результат",
        "",
        f"- В простом классе правил предельное снижение потерь от распределительной информации: {-simple_value['mean_delta']:.6g}.",
        f"- В LQG-классе предельное снижение потерь от распределительных наблюдений: {-lqg_value['mean_delta']:.6g}.",
        f"- Кластерный 95%-й интервал для LQG-эффекта: [{lqg_value['cluster_ci_low']:.6g}, {lqg_value['cluster_ci_high']:.6g}] по разности `dist - agg`.",
        "",
        "## Riccati diagnostics",
        "",
        f"- Источник матрицы B: `{system.policy_input_source}`.",
        f"- Спектральный радиус A: {np.max(np.abs(np.linalg.eigvals(system.A))):.4f}.",
        f"- Спектральный радиус замкнутой LQR-системы: {lqr_solution.closed_loop_spectral_radius:.4f}.",
        f"- Riccati iterations: {lqr_solution.iterations}.",
        f"- Riccati converged: {lqr_solution.converged}.",
        "",
        "## Интерпретация",
        "",
        "Если LQG-эффект положителен, то ценность распределительных наблюдений не является артефактом выбранного простого правила. "
        "Если эффект слабее или статистически неустойчив, основной вывод надо формулировать осторожнее: распределительная информация помогает "
        "в ограниченном классе правил, но оптимальный линейный регулятор уже извлекает значительную часть полезной информации из агрегатов.",
        "",
        "## Files",
        "",
        "- `lqg_oracle_summary.csv`",
        "- `lqg_oracle_pairwise.csv`",
        "- `lqg_oracle_gains.csv`",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
