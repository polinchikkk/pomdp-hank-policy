from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import ClosedLoopSSJEnvironment, PolicyLossWeights
from hank_ssj.closed_loop_environment import diagnostics_to_row
from policy.linear_rules import LinearRule, rule_spec_for_information_state
from policy.optimize_rules import bootstrap_interval, compare_paired_losses


PAIRWISE_COMPARISONS = (
    ("filtered_aggregates", "aggregate_only", "Фильтрованные агрегаты против текущих агрегатов"),
    ("observed_distribution", "aggregate_only", "Наблюдаемые распределительные показатели против текущих агрегатов"),
    ("filtered_distribution_mpc", "filtered_aggregates", "Ценность средней MPC"),
    ("filtered_distribution_liquidity", "filtered_aggregates", "Ценность доли низколиквидных домохозяйств"),
    ("filtered_distribution_exposure", "filtered_aggregates", "Ценность процентной экспозиции"),
    ("filtered_distribution", "filtered_aggregates", "Все распределительные статистики против фильтрованных агрегатов"),
    ("full_information", "aggregate_only", "Полная информация против текущих агрегатов"),
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
class ClosedLoopExperimentSpec:
    hank_observables: str
    hank_observations: str
    jacobians: str
    state_space_spec: str
    fitted_policy_rules: str
    output_dir: str
    modes: tuple[str, ...]
    test_seeds: tuple[int, ...]
    max_iterations: int
    min_iterations: int
    tolerance: float
    damping: float
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen policy rules in closed-loop local SSJ projection.")
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--hank-observations", default="outputs/ssj/stochastic/hank_observations.csv")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--state-space-spec", default="outputs/ssj/stochastic/state_space/state_space_spec.json")
    parser.add_argument("--fitted-policy-rules", default="outputs/ssj/stochastic/main_voi_joint_filter/fitted_policy_rules.csv")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/closed_loop")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--modes", default="partial_local_projection,closed_loop_local_projection")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--min-iterations", type=int, default=2)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--damping", type=float, default=0.75)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    test_seeds = _parse_seed_range(args.test_seeds)
    policies = _load_fitted_rules(Path(args.fitted_policy_rules))
    environment = ClosedLoopSSJEnvironment.from_files(
        hank_observables_csv=Path(args.hank_observables),
        hank_observations_csv=Path(args.hank_observations),
        jacobians_npz=Path(args.jacobians),
        state_space_spec_json=Path(args.state_space_spec),
        loss_weights=PolicyLossWeights(),
    )

    loss_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for mode in modes:
        for scenario in environment.scenarios:
            for seed in test_seeds:
                for state, policy in policies.items():
                    result = environment.simulate_scenario(
                        policy=policy,
                        information_state=state,
                        scenario=scenario,
                        seed=seed,
                        mode=mode,
                        max_iterations=args.max_iterations,
                        min_iterations=args.min_iterations,
                        tolerance=args.tolerance,
                        damping=args.damping,
                    )
                    loss_rows.append(
                        {
                            "mode": mode,
                            "scenario": scenario,
                            "observation_seed": int(seed),
                            "information_state": state,
                            "information_state_ru": STATE_LABEL_RU[state],
                            **asdict(result.loss),
                        }
                    )
                    diagnostic_rows.append(diagnostics_to_row(result.diagnostics))

    losses = pd.DataFrame(loss_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary = _summary_table(losses)
    pairwise = _pairwise_table(losses)
    convergence = _convergence_summary(diagnostics)

    losses.to_csv(output_dir / "trajectory_losses_closed_loop.csv", index=False)
    diagnostics.to_csv(output_dir / "convergence_diagnostics.csv", index=False)
    summary.to_csv(output_dir / "main_voi_closed_loop_summary.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_closed_loop_value_of_information.csv", index=False)
    convergence.to_csv(output_dir / "convergence_summary.csv", index=False)
    _write_latex(summary, output_dir / "table_main_voi_closed_loop_summary.tex")
    _write_latex(pairwise, output_dir / "table_pairwise_closed_loop_value_of_information.tex")
    _write_latex(convergence, output_dir / "table_convergence_summary.tex")

    spec = ClosedLoopExperimentSpec(
        hank_observables=args.hank_observables,
        hank_observations=args.hank_observations,
        jacobians=args.jacobians,
        state_space_spec=args.state_space_spec,
        fitted_policy_rules=args.fitted_policy_rules,
        output_dir=args.output_dir,
        modes=modes,
        test_seeds=tuple(test_seeds),
        max_iterations=int(args.max_iterations),
        min_iterations=int(args.min_iterations),
        tolerance=float(args.tolerance),
        damping=float(args.damping),
        note=(
            "Правила заморожены после main_voi_joint_filter. Closed-loop режим пересчитывает "
            "контрфактическое состояние, наблюдения и фильтрованные признаки фиксированным числом итераций."
        ),
    )
    (output_dir / "closed_loop_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(summary, pairwise, convergence, diagnostics, output_dir / "report_closed_loop.md")
    print(f"Wrote {output_dir / 'main_voi_closed_loop_summary.csv'}")
    print(f"Wrote {output_dir / 'convergence_diagnostics.csv'}")
    print(f"Wrote {output_dir / 'report_closed_loop.md'}")


def _load_fitted_rules(path: Path) -> dict[str, LinearRule]:
    frame = pd.read_csv(path)
    policies: dict[str, LinearRule] = {}
    for state, group in frame.groupby("information_state", sort=False):
        spec = rule_spec_for_information_state(state)
        terms = group.set_index("term")["coefficient"].to_dict()
        policies[state] = LinearRule(
            spec=spec,
            intercept=float(terms["intercept"]),
            coefficients=tuple(float(terms[name]) for name in spec.feature_names),
            lagged_rate_weight=float(terms.get("lagged_rate", 0.0)),
        )
    return policies


def _parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


def _summary_table(losses: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (mode, scenario), frame in _mode_scenario_groups(losses):
        for state, state_frame in frame.groupby("information_state", sort=False):
            values = state_frame["total_loss"].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_interval(values)
            rows.append(
                {
                    "mode": mode,
                    "scenario": scenario,
                    "information_state": state,
                    "information_state_ru": STATE_LABEL_RU[state],
                    "num_trajectories": int(values.size),
                    "mean_loss": float(np.mean(values)),
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
    rows: list[dict[str, object]] = []
    for (mode, scenario), frame in _mode_scenario_groups(losses):
        pivot = frame.pivot_table(
            index=["mode", "scenario", "observation_seed"],
            columns="information_state",
            values="total_loss",
            aggfunc="first",
        )
        for left, right, label in PAIRWISE_COMPARISONS:
            if left not in pivot.columns or right not in pivot.columns:
                continue
            comparison = compare_paired_losses(
                left_name=left,
                right_name=right,
                left_losses=pivot[left].to_numpy(dtype=float),
                right_losses=pivot[right].to_numpy(dtype=float),
                tie_eps=1e-10,
            )
            rows.append(
                {
                    "mode": mode,
                    "scenario": scenario,
                    "comparison": f"{left}_minus_{right}",
                    "comparison_ru": label,
                    "left": left,
                    "right": right,
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


def _convergence_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (mode, state), frame in diagnostics.groupby(["mode", "information_state"], sort=False):
        rows.append(
            {
                "mode": mode,
                "information_state": state,
                "information_state_ru": STATE_LABEL_RU[state],
                "num_trajectories": int(frame.shape[0]),
                "convergence_failure_rate": float(1.0 - frame["converged"].mean()),
                "mean_iterations": float(frame["iterations"].mean()),
                "mean_rate_update_norm": float(frame["rate_update_norm"].mean()),
                "mean_state_update_norm": float(frame["state_update_norm"].mean()),
                "mean_rate_inversion_residual": float(frame["rate_inversion_residual"].mean()),
                "max_rate_inversion_residual": float(frame["rate_inversion_residual"].max()),
                "rate_inversion_condition_number": float(frame["rate_inversion_condition_number"].max()),
                "ridge_used": float(frame["ridge_used"].iloc[0]),
                "mean_stability_penalty": float(frame["stability_penalty"].mean()),
                "mean_convergence_penalty": float(frame["convergence_penalty"].mean()),
                "fallback_effects": ",".join(sorted(set(",".join(frame["fallback_effects"]).split(",")) - {""})),
            }
        )
    return pd.DataFrame(rows)


def _mode_scenario_groups(losses: pd.DataFrame):
    for key, frame in losses.groupby(["mode", "scenario"], sort=False):
        yield key, frame
    for mode, frame in losses.groupby("mode", sort=False):
        yield (mode, "all"), frame


def _write_latex(frame: pd.DataFrame, path: Path) -> None:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    for column in numeric:
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6g}")
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_report(
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    convergence: pd.DataFrame,
    diagnostics: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Closed-loop local SSJ evaluation",
        "",
        "Правила заморожены после основного прогона с совместным фильтром. ",
        "В closed-loop режиме ставка, контрфактическое состояние, наблюдения и фильтрованные признаки пересчитываются итеративно.",
        "",
    ]
    for mode in summary["mode"].drop_duplicates():
        block = summary[(summary["mode"] == mode) & (summary["scenario"] == "all")].sort_values("mean_loss")
        lines.extend([f"## {mode}", ""])
        for _, row in block.head(6).iterrows():
            lines.append(f"- {row['information_state_ru']}: {row['mean_loss']:.6g}")
        main_pair = pairwise[
            (pairwise["mode"] == mode)
            & (pairwise["scenario"] == "all")
            & (pairwise["comparison"] == "filtered_distribution_minus_filtered_aggregates")
        ]
        if not main_pair.empty:
            row = main_pair.iloc[0]
            lines.append(
                f"- Все распределительные статистики против агрегатов: снижение потерь {-row['mean_delta']:.6g}, "
                f"ДИ [{-row['ci_high']:.6g}, {-row['ci_low']:.6g}], доля выигрышей {row['win_rate']:.3g}."
            )
        lines.append("")
    failure_rate = diagnostics.groupby("mode")["converged"].mean().map(lambda value: 1.0 - value)
    lines.extend(["## Сходимость", ""])
    for mode, rate in failure_rate.items():
        lines.append(f"- {mode}: доля несошедшихся траекторий {rate:.3g}.")
    inversion = diagnostics.groupby("mode", sort=False).agg(
        mean_rate_inversion_residual=("rate_inversion_residual", "mean"),
        max_rate_inversion_residual=("rate_inversion_residual", "max"),
        rate_inversion_condition_number=("rate_inversion_condition_number", "max"),
        ridge_used=("ridge_used", "first"),
    )
    lines.extend(["", "## Обратное отображение ставки", ""])
    for mode, row in inversion.iterrows():
        lines.append(
            f"- {mode}: mean residual {row['mean_rate_inversion_residual']:.3g}, "
            f"max residual {row['max_rate_inversion_residual']:.3g}, "
            f"cond(J_i) {row['rate_inversion_condition_number']:.3g}, ridge {row['ridge_used']:.1e}."
        )
    fallback = sorted(set(",".join(diagnostics["fallback_effects"]).split(",")) - {""})
    if fallback:
        lines.extend(
            [
                "",
                "## Ограничение текущей closed-loop проверки",
                "",
                (
                    "В текущем `jacobians.npz` нет прямых SSJ-якобианов для части распределительных статистик. "
                    "Для них используется локальная регрессионная проекция на агрегатные SSJ-эффекты: "
                    + ", ".join(fallback)
                    + "."
                ),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
