from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.steady_state import solve_steady_state
from hank_full_baseline.transition import solve_transition
from hank_learning_policy_baseline.policies import BasePolicy, ClassicalFilteredRulePolicy, FullInformationRulePolicy
from hank_partial_info_baseline.state_space import fit_reduced_state_space
from regime_switching_baseline.regime_model import build_regime_switching_model

from .config import RegimeLearningConfig, RegimeLearningVariant
from .core_matrix import SCENARIO_LABELS
from .environment import RegimeSwitchingPolicyEnvironment, build_scenario_spec
from .evaluation import evaluate_policy_trace, simulate_policy_episode
from .tuning import default_universal_candidate_lookup, extreme_sticky_regime_config


SCENARIO_NAMES = (
    "macro_core_moderate_gap",
    "macro_core_strong_gap",
    "thin_information_moderate_gap",
    "thin_information_strong_gap",
)


@dataclass(frozen=True)
class LinearRuleParameters:
    phi_pi: float
    phi_y: float
    rho_i: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class HistoryRuleParameters:
    phi_pi: float
    phi_y: float
    rho_i: float
    alpha: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class OptimizedEstimatedStateRulePolicy(BasePolicy):
    """Taylor-type linear rule selected on validation paths and applied to the filtered state."""

    def __init__(self, params: LinearRuleParameters) -> None:
        self.params = params

    def rate(self, observation: np.ndarray, info: dict) -> float:
        state_names = tuple(info["state_names"])
        state = np.asarray(info["filtered_state"], dtype=float)
        previous_rate = float(info["current_rate"])
        lower, upper = info["rate_bounds"]
        idx_rstar = state_names.index("rstar_gap")
        idx_pi = state_names.index("inflation_gap")
        idx_output = state_names.index("output_gap")
        target = (
            state[idx_rstar]
            + self.params.phi_pi * state[idx_pi]
            + self.params.phi_y * state[idx_output]
        )
        rate = self.params.rho_i * previous_rate + (1.0 - self.params.rho_i) * target
        return float(np.clip(rate, lower, upper))


class HistoryObservationRulePolicy(BasePolicy):
    """Stateful linear rule that uses an exponentially smoothed history of observed releases."""

    def __init__(self, params: HistoryRuleParameters) -> None:
        self.params = params
        self._smoothed_pi = 0.0
        self._smoothed_output = 0.0
        self._initialized = False

    def reset(self) -> None:
        self._smoothed_pi = 0.0
        self._smoothed_output = 0.0
        self._initialized = False

    def rate(self, observation: np.ndarray, info: dict) -> float:
        observation_names = tuple(info["noisy_observation_names"])
        observations = np.asarray(info["current_observations"], dtype=float)
        previous_rate = float(info["current_rate"])
        lower, upper = info["rate_bounds"]
        obs_map = {name: float(value) for name, value in zip(observation_names, observations)}
        current_pi = obs_map.get("pi", obs_map.get("inflation_gap", 0.0))
        current_output = obs_map.get("output_gap", 0.0)
        if not self._initialized:
            self._smoothed_pi = current_pi
            self._smoothed_output = current_output
            self._initialized = True
        else:
            alpha = self.params.alpha
            self._smoothed_pi = alpha * self._smoothed_pi + (1.0 - alpha) * current_pi
            self._smoothed_output = alpha * self._smoothed_output + (1.0 - alpha) * current_output
        target = self.params.phi_pi * self._smoothed_pi + self.params.phi_y * self._smoothed_output
        rate = self.params.rho_i * previous_rate + (1.0 - self.params.rho_i) * target
        return float(np.clip(rate, lower, upper))


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _write_extension_latex_tables(root: Path, comparison_summary: pd.DataFrame, selected_specs: pd.DataFrame) -> None:
    focus = comparison_summary[
        comparison_summary["comparison_name"].isin(
            ["linear_minus_classical", "history_minus_classical", "linear_minus_history"]
        )
    ].copy()
    lines = [
        "\\begin{tabular}{p{0.32\\linewidth}p{0.24\\linewidth}rrrr}",
        "\\toprule",
        "Сценарий & Сравнение & $\\Delta J$ & 95\\% ДИ & Доля побед & $N$ \\\\",
        "\\midrule",
    ]
    for _, row in focus.iterrows():
        ci = f"[{row['ci_lower']:.4f}; {row['ci_upper']:.4f}]"
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["scenario_label"])),
                    _latex_escape(str(row["comparison_label"])),
                    f"{float(row['mean_delta_cumulative_loss']):.4f}",
                    _latex_escape(ci),
                    f"{float(row['win_rate']):.2f}",
                    f"{int(row['num_test_trajectories'])}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_stage6_policy_extensions_comparison.tex").write_text("\n".join(lines), encoding="utf-8")

    specs = selected_specs.copy()
    lines = [
        "\\begin{tabular}{p{0.36\\linewidth}lrrrr}",
        "\\toprule",
        "Сценарий & Правило & $\\phi_\\pi$ & $\\phi_y$ & $\\rho_i$ & $\\alpha$ \\\\",
        "\\midrule",
    ]
    policy_labels = {
        "optimized_linear_estimated_state": "Линейное по оценке",
        "history_observables_rule": "Историческое по наблюдениям",
    }
    for _, row in specs.iterrows():
        alpha = "" if pd.isna(row.get("alpha", np.nan)) else f"{float(row['alpha']):.2f}"
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["scenario_label"])),
                    _latex_escape(policy_labels.get(str(row["policy_name"]), str(row["policy_name"]))),
                    f"{float(row['phi_pi']):.2f}",
                    f"{float(row['phi_y']):.3f}",
                    f"{float(row['rho_i']):.2f}",
                    alpha,
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_stage6_policy_extensions_selected_rules.tex").write_text("\n".join(lines), encoding="utf-8")


def _bootstrap_ci(values: np.ndarray, *, seed: int = 2026, draws: int = 4000) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _build_variant(*, scenario_name: str, input_mode: str) -> RegimeLearningVariant:
    suffix = "estimated_state" if input_mode == "belief_state" else "history_observables"
    return RegimeLearningVariant(
        name=f"{scenario_name}_{suffix}",
        scenario_name=scenario_name,
        scenario_label=SCENARIO_LABELS[scenario_name],
        input_mode=input_mode,
        include_distributional_state=True,
        description=(
            "Selected linear rule on the filtered state."
            if input_mode == "belief_state"
            else "History-based rule on observed macro releases."
        ),
    )


@lru_cache(maxsize=1)
def _base_reduced_objects():
    hank_config = default_calibration()
    regime_config = extreme_sticky_regime_config()
    bundle = solve_steady_state(hank_config)
    reduced_model = fit_reduced_state_space(bundle, hank_config, regime_config.partial_config)
    return hank_config, regime_config, reduced_model


def _build_objects(scenario_name: str, *, input_mode: str, validation_seeds: tuple[int, ...], test_seeds: tuple[int, ...]):
    candidate = default_universal_candidate_lookup()["larger_network"]
    hank_config, regime_config, reduced_model = _base_reduced_objects()
    config = RegimeLearningConfig(
        horizon=60,
        gamma=0.99,
        lambda_y=0.5,
        lambda_i=0.05,
        action_bound=candidate.action_bound,
        classical_policy_mode="switching",
        training_seeds=(11,),
        selection_seeds=validation_seeds,
        evaluation_seeds=test_seeds,
        regime_config=regime_config,
        ppo=candidate.ppo,
    )
    variant = _build_variant(scenario_name=scenario_name, input_mode=input_mode)
    scenario_spec = build_scenario_spec(config, variant)
    regime_model = build_regime_switching_model(reduced_model, regime_config, scenario_spec.gap_scale)

    def env_factory():
        return RegimeSwitchingPolicyEnvironment(
            model=regime_model,
            regime_config=regime_config,
            scenario_spec=scenario_spec,
            phi_pi=hank_config.phi_pi,
            phi_y=hank_config.phi_y,
            rho_i=hank_config.rho_i,
        )

    return hank_config, scenario_spec, env_factory


def _mean_cumulative_loss(*, env_factory, scenario_spec, policy: BasePolicy, seeds: Iterable[int]) -> tuple[float, float, int]:
    losses = []
    volatilities = []
    unstable = 0
    for seed in seeds:
        trace = simulate_policy_episode(
            env_factory=env_factory,
            policy=policy,
            scenario_spec=scenario_spec,
            evaluation_seed=int(seed),
            policy_name="candidate_policy",
            policy_label="Candidate policy",
            training_seed=None,
        )
        losses.append(float(trace["loss"].sum()))
        volatilities.append(float(np.std(trace["policy_rate"].to_numpy(dtype=float))))
        values = trace[["true_inflation_gap", "true_output_gap", "policy_rate"]].to_numpy(dtype=float)
        unstable += int((not np.isfinite(values).all()) or np.any(np.abs(values) > np.array([0.06, 0.12, 0.06])[None, :]))
    return float(np.mean(losses)), float(np.mean(volatilities)), int(unstable)


def _select_linear_rule(*, scenario_name: str, validation_seeds: tuple[int, ...]) -> tuple[LinearRuleParameters, pd.DataFrame]:
    _hank_config, scenario_spec, env_factory = _build_objects(
        scenario_name,
        input_mode="belief_state",
        validation_seeds=validation_seeds,
        test_seeds=validation_seeds,
    )
    rows = []
    for phi_pi in (1.0, 1.5, 2.0, 2.5, 3.0):
        for phi_y in (0.0, 0.125, 0.25, 0.5):
            for rho_i in (0.3, 0.5, 0.7, 0.85):
                params = LinearRuleParameters(phi_pi=phi_pi, phi_y=phi_y, rho_i=rho_i)
                loss, volatility, unstable = _mean_cumulative_loss(
                    env_factory=env_factory,
                    scenario_spec=scenario_spec,
                    policy=OptimizedEstimatedStateRulePolicy(params),
                    seeds=validation_seeds,
                )
                rows.append({
                    "scenario_name": scenario_name,
                    **params.to_dict(),
                    "validation_cumulative_loss": loss,
                    "validation_policy_volatility": volatility,
                    "validation_unstable_episodes": unstable,
                })
    grid = pd.DataFrame(rows).sort_values(
        ["validation_unstable_episodes", "validation_cumulative_loss", "validation_policy_volatility"]
    ).reset_index(drop=True)
    best = grid.iloc[0]
    return (
        LinearRuleParameters(phi_pi=float(best["phi_pi"]), phi_y=float(best["phi_y"]), rho_i=float(best["rho_i"])),
        grid,
    )


def _select_history_rule(*, scenario_name: str, validation_seeds: tuple[int, ...]) -> tuple[HistoryRuleParameters, pd.DataFrame]:
    _hank_config, scenario_spec, env_factory = _build_objects(
        scenario_name,
        input_mode="raw_observations",
        validation_seeds=validation_seeds,
        test_seeds=validation_seeds,
    )
    output_grid = (0.0, 0.125, 0.25) if "macro_core" in scenario_name else (0.0,)
    rows = []
    for phi_pi in (0.5, 1.0, 1.5, 2.0, 2.5):
        for phi_y in output_grid:
            for rho_i in (0.3, 0.5, 0.7, 0.85):
                for alpha in (0.2, 0.5, 0.8):
                    params = HistoryRuleParameters(phi_pi=phi_pi, phi_y=phi_y, rho_i=rho_i, alpha=alpha)
                    loss, volatility, unstable = _mean_cumulative_loss(
                        env_factory=env_factory,
                        scenario_spec=scenario_spec,
                        policy=HistoryObservationRulePolicy(params),
                        seeds=validation_seeds,
                    )
                    rows.append({
                        "scenario_name": scenario_name,
                        **params.to_dict(),
                        "validation_cumulative_loss": loss,
                        "validation_policy_volatility": volatility,
                        "validation_unstable_episodes": unstable,
                    })
    grid = pd.DataFrame(rows).sort_values(
        ["validation_unstable_episodes", "validation_cumulative_loss", "validation_policy_volatility"]
    ).reset_index(drop=True)
    best = grid.iloc[0]
    return (
        HistoryRuleParameters(
            phi_pi=float(best["phi_pi"]),
            phi_y=float(best["phi_y"]),
            rho_i=float(best["rho_i"]),
            alpha=float(best["alpha"]),
        ),
        grid,
    )


def _evaluate_selected_rules(
    *,
    scenario_name: str,
    validation_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
    linear_params: LinearRuleParameters,
    history_params: HistoryRuleParameters,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _hank_config, belief_spec, belief_env_factory = _build_objects(
        scenario_name,
        input_mode="belief_state",
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )
    _hank_config, raw_spec, raw_env_factory = _build_objects(
        scenario_name,
        input_mode="raw_observations",
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )

    policy_metric_rows = []
    policy_path_frames = []
    classical_policy = ClassicalFilteredRulePolicy(action_bound=belief_spec.action_bound)
    full_policy = FullInformationRulePolicy(action_bound=belief_spec.action_bound)
    linear_policy = OptimizedEstimatedStateRulePolicy(linear_params)
    history_policy = HistoryObservationRulePolicy(history_params)

    for seed in test_seeds:
        full_trace = simulate_policy_episode(
            env_factory=belief_env_factory,
            policy=full_policy,
            scenario_spec=belief_spec,
            evaluation_seed=int(seed),
            policy_name="full_information_rule",
            policy_label="Правило при полной информации",
            training_seed=None,
        )
        policies = (
            (
                belief_env_factory,
                belief_spec,
                classical_policy,
                "classical_filtered_rule",
                "Классическое правило по оценённому состоянию",
            ),
            (
                belief_env_factory,
                belief_spec,
                linear_policy,
                "optimized_linear_estimated_state",
                "Оптимизированное линейное правило по оценённому состоянию",
            ),
            (
                raw_env_factory,
                raw_spec,
                history_policy,
                "history_observables_rule",
                "Историческое правило по наблюдаемым переменным",
            ),
            (
                belief_env_factory,
                belief_spec,
                full_policy,
                "full_information_rule",
                "Правило при полной информации",
            ),
        )
        for env_factory, scenario_spec, policy, policy_name, policy_label in policies:
            trace = full_trace if policy_name == "full_information_rule" else simulate_policy_episode(
                env_factory=env_factory,
                policy=policy,
                scenario_spec=scenario_spec,
                evaluation_seed=int(seed),
                policy_name=policy_name,
                policy_label=policy_label,
                training_seed=None,
            )
            metrics, path_frame = evaluate_policy_trace(
                policy_trace=trace,
                reference_trace=full_trace,
                scenario_spec=scenario_spec,
            )
            metrics["scenario_name"] = scenario_name
            metrics["scenario_label"] = SCENARIO_LABELS[scenario_name]
            policy_metric_rows.append(metrics)
            path_frame["scenario_name"] = scenario_name
            path_frame["scenario_label"] = SCENARIO_LABELS[scenario_name]
            policy_path_frames.append(path_frame)

    return pd.DataFrame(policy_metric_rows), pd.concat(policy_path_frames, ignore_index=True)


def _comparison_summary(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparison_specs = {
        "linear_minus_classical": ("optimized_linear_estimated_state", "classical_filtered_rule"),
        "history_minus_classical": ("history_observables_rule", "classical_filtered_rule"),
        "linear_minus_history": ("optimized_linear_estimated_state", "history_observables_rule"),
        "linear_minus_full_information": ("optimized_linear_estimated_state", "full_information_rule"),
    }
    labels = {
        "linear_minus_classical": "Оптимизированное линейное минус классическое",
        "history_minus_classical": "Историческое по наблюдаемым минус классическое",
        "linear_minus_history": "Оценённое состояние минус история наблюдений",
        "linear_minus_full_information": "Оптимизированное линейное минус полная информация",
    }
    for scenario_name, frame in policy_metrics.groupby("scenario_name"):
        pivot = frame.pivot_table(index="evaluation_seed", columns="policy_name", values="cumulative_policy_loss", aggfunc="first")
        for comparison_name, (left, right) in comparison_specs.items():
            if left not in pivot or right not in pivot:
                continue
            deltas = pivot[left].to_numpy(dtype=float) - pivot[right].to_numpy(dtype=float)
            benchmark = pivot[right].to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_ci(deltas)
            rows.append({
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "comparison_name": comparison_name,
                "comparison_label": labels[comparison_name],
                "mean_delta_cumulative_loss": float(deltas.mean()),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "win_rate": float(np.mean(deltas < 0.0)),
                "probability_of_degradation": float(np.mean(deltas > 0.0)),
                "relative_improvement_pct": float(100.0 * np.mean(-deltas / benchmark)),
                "num_test_trajectories": int(deltas.size),
            })
    return pd.DataFrame(rows).sort_values(["scenario_name", "comparison_name"]).reset_index(drop=True)


def _component_decomposition(policy_paths: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        policy_paths.groupby(["scenario_name", "scenario_label", "evaluation_seed", "policy_name"])[
            ["policy_loss", "inflation_gap", "output_gap", "policy_rate"]
        ]
        .agg(
            cumulative_policy_loss=("policy_loss", "sum"),
            inflation_component=("inflation_gap", lambda x: float(np.sum(np.square(x.to_numpy(dtype=float))))),
            output_component=("output_gap", lambda x: float(0.5 * np.sum(np.square(x.to_numpy(dtype=float))))),
            rate_component=("policy_rate", lambda x: float(0.05 * np.sum(np.square(np.diff(x.to_numpy(dtype=float), prepend=0.0))))),
        )
        .reset_index()
    )
    rows = []
    comparisons = {
        "linear_vs_classical": ("optimized_linear_estimated_state", "classical_filtered_rule"),
        "linear_vs_history": ("optimized_linear_estimated_state", "history_observables_rule"),
    }
    for scenario_name, frame in grouped.groupby("scenario_name"):
        pivot = frame.pivot_table(
            index="evaluation_seed",
            columns="policy_name",
            values=["inflation_component", "output_component", "rate_component"],
            aggfunc="first",
        )
        for comparison_name, (left, right) in comparisons.items():
            if ("inflation_component", left) not in pivot or ("inflation_component", right) not in pivot:
                continue
            rows.append({
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "comparison_name": comparison_name,
                "delta_inflation_component": float((pivot[("inflation_component", left)] - pivot[("inflation_component", right)]).mean()),
                "delta_output_component": float((pivot[("output_component", left)] - pivot[("output_component", right)]).mean()),
                "delta_rate_component": float((pivot[("rate_component", left)] - pivot[("rate_component", right)]).mean()),
            })
    return pd.DataFrame(rows).sort_values(["scenario_name", "comparison_name"]).reset_index(drop=True)


def _plot_delta_intervals(summary: pd.DataFrame, path: Path) -> None:
    data = summary[
        summary["comparison_name"].isin(("linear_minus_classical", "history_minus_classical", "linear_minus_history"))
    ].copy()
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    order = [scenario for scenario in SCENARIO_NAMES if scenario in set(data["scenario_name"])]
    comparisons = [
        ("linear_minus_classical", "Линейное по оценке vs классическое", "#0b6e4f", -0.2),
        ("history_minus_classical", "История наблюдений vs классическое", "#3a86ff", 0.0),
        ("linear_minus_history", "Оценка состояния vs история", "#ca6702", 0.2),
    ]
    x = np.arange(len(order))
    for comparison_name, label, color, offset in comparisons:
        available = [scenario for scenario in order if scenario in set(data.loc[data["comparison_name"] == comparison_name, "scenario_name"])]
        if not available:
            continue
        frame = data[data["comparison_name"] == comparison_name].set_index("scenario_name").loc[available]
        positions = np.array([order.index(scenario) for scenario in available], dtype=float)
        means = frame["mean_delta_cumulative_loss"].to_numpy(dtype=float)
        lower = means - frame["ci_lower"].to_numpy(dtype=float)
        upper = frame["ci_upper"].to_numpy(dtype=float) - means
        ax.errorbar(positions + offset, means, yerr=np.vstack([lower, upper]), fmt="o", capsize=4, color=color, label=label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    short_labels = {
        "macro_core_moderate_gap": "Базовый\nмакронабор,\nумеренная",
        "macro_core_strong_gap": "Базовый\nмакронабор,\nвысокая",
        "thin_information_moderate_gap": "Ограниченный\nнабор,\nумеренная",
        "thin_information_strong_gap": "Ограниченный\nнабор,\nвысокая",
    }
    ax.set_xticklabels([short_labels[scenario] for scenario in order])
    ax.set_ylabel("$\\Delta J$; ниже нуля означает выигрыш")
    ax.set_title("Расширенные сравнения правил на тестовых траекториях")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_component_decomposition(components: pd.DataFrame, path: Path) -> None:
    data = components[components["comparison_name"] == "linear_vs_classical"].copy()
    order = [scenario for scenario in SCENARIO_NAMES if scenario in set(data["scenario_name"])]
    data = data.set_index("scenario_name").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    x = np.arange(len(data))
    positive_bottom = np.zeros(len(data))
    negative_bottom = np.zeros(len(data))
    specs = [
        ("delta_inflation_component", "Инфляция", "#ca6702"),
        ("delta_output_component", "Разрыв выпуска", "#0b6e4f"),
        ("delta_rate_component", "Сглаживание ставки", "#4361ee"),
    ]
    for column, label, color in specs:
        values = data[column].to_numpy(dtype=float)
        bottoms = np.where(values >= 0.0, positive_bottom, negative_bottom)
        ax.bar(x, values, bottom=bottoms, color=color, width=0.58, label=label)
        positive_bottom += np.where(values >= 0.0, values, 0.0)
        negative_bottom += np.where(values < 0.0, values, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    short_labels = {
        "macro_core_moderate_gap": "Базовый\nмакронабор,\nумеренная",
        "macro_core_strong_gap": "Базовый\nмакронабор,\nвысокая",
        "thin_information_moderate_gap": "Ограниченный\nнабор,\nумеренная",
        "thin_information_strong_gap": "Ограниченный\nнабор,\nвысокая",
    }
    ax.set_xticklabels([short_labels[scenario] for scenario in order])
    ax.set_ylabel("Вклад в $\\Delta J$")
    ax.set_title("Разложение выигрыша оптимизированного линейного правила")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _run_full_hank_projection(
    *,
    policy_paths: pd.DataFrame,
    output_dir: Path,
    scenario_names: tuple[str, ...],
    policy_names: tuple[str, ...],
    shock_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    rows = []
    path_rows = []
    for scenario_name in scenario_names:
        for policy_name in policy_names:
            subset = policy_paths[
                (policy_paths["scenario_name"] == scenario_name)
                & (policy_paths["policy_name"] == policy_name)
            ].copy()
            if subset.empty:
                continue
            shock_path = (
                subset.groupby("period")["policy_rate"].mean().sort_index().to_numpy(dtype=float)
            )
            shock_path = shock_path[: hank_config.shock_T]
            if shock_path.size < hank_config.shock_T:
                shock_path = np.pad(shock_path, (0, hank_config.shock_T - shock_path.size))
            transition = None
            scale_used = math.nan
            solver_error = ""
            for scale in shock_scales:
                try:
                    transition = solve_transition(bundle, {"monetary_policy_shock": scale * shock_path})
                    scale_used = float(scale)
                    break
                except Exception as exc:  # pragma: no cover - records numerical solver failures.
                    solver_error = f"{type(exc).__name__}: {exc}"
                    transition = None
            if transition is None:
                rows.append({
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": policy_name,
                    "solver_success": 0,
                    "scale_used": math.nan,
                    "solver_error": solver_error,
                    "mean_shock_abs": float(np.mean(np.abs(shock_path))),
                    "peak_shock_abs": float(np.max(np.abs(shock_path))),
                    "full_hank_cumulative_loss": math.nan,
                    "peak_inflation_abs": math.nan,
                    "peak_output_gap_abs": math.nan,
                    "peak_consumption_abs": math.nan,
                    "peak_rate_abs": math.nan,
                })
                continue
            pi = transition["pi"]
            output = transition["output_gap"]
            rate = transition["i"]
            loss = pi**2 + 0.5 * output**2 + 0.05 * np.square(np.diff(rate, prepend=0.0))
            rows.append({
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "policy_name": policy_name,
                "solver_success": 1,
                "scale_used": scale_used,
                "solver_error": "",
                "mean_shock_abs": float(np.mean(np.abs(shock_path))),
                "peak_shock_abs": float(np.max(np.abs(shock_path))),
                "full_hank_cumulative_loss": float(np.sum(loss)),
                "peak_inflation_abs": float(np.max(np.abs(pi))),
                "peak_output_gap_abs": float(np.max(np.abs(output))),
                "peak_consumption_abs": float(np.max(np.abs(transition["C"]))),
                "peak_rate_abs": float(np.max(np.abs(rate))),
            })
            for period in range(len(pi)):
                path_rows.append({
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": policy_name,
                    "period": int(period),
                    "mean_policy_rate_path_used_as_shock": float(scale_used * shock_path[period]),
                    "projection_scale_used": scale_used,
                    "inflation_gap": float(pi[period]),
                    "output_gap": float(output[period]),
                    "policy_rate": float(rate[period]),
                    "consumption": float(transition["C"][period]),
                    "period_loss": float(loss[period]),
                })
    metrics = pd.DataFrame(rows)
    paths = pd.DataFrame(path_rows)
    metrics.to_csv(output_dir / "full_hank_projection_metrics.csv", index=False)
    paths.to_csv(output_dir / "full_hank_projection_paths.csv", index=False)
    return metrics, paths


def run_full_hank_projection_from_policy_paths(
    *,
    input_dir: str = "outputs/hank_regime_learning_stage6_policy_extensions",
    output_dir: str | None = None,
    scenario_names: tuple[str, ...] = ("thin_information_strong_gap",),
    policy_names: tuple[str, ...] = (
        "classical_filtered_rule",
        "optimized_linear_estimated_state",
        "history_observables_rule",
    ),
) -> dict[str, pd.DataFrame]:
    root = Path(input_dir)
    out = Path(output_dir) if output_dir is not None else root
    out.mkdir(parents=True, exist_ok=True)
    policy_paths = pd.read_csv(root / "policy_paths.csv")
    metrics, paths = _run_full_hank_projection(
        policy_paths=policy_paths,
        output_dir=out,
        scenario_names=scenario_names,
        policy_names=policy_names,
    )
    lines = [
        "# Full-HANK projection для правил этапа 6",
        "",
        "Средние тестовые траектории ставки из reduced-state экспериментов передаются в полную HANK как траектории monetary-policy shock. Это не является полной оптимизацией правила в full HANK; это проверка согласованности направления результатов при пропуске выбранных траекторий через full-HANK transition solver.",
        "",
        "Если full-scale траектория не сходится в nonlinear solver, используется последовательное уменьшение амплитуды. Поэтому поле `scale_used` важно для интерпретации: значение ниже единицы означает, что full-HANK solver принимает только локальную версию соответствующей policy path.",
        "",
        "## Результаты",
        "",
    ]
    for row in metrics.to_dict(orient="records"):
        if int(row["solver_success"]) == 1:
            lines.append(
                f"- {row['scenario_label']}, `{row['policy_name']}`: scale `{row['scale_used']:.2f}`, "
                f"full-HANK cumulative loss `{row['full_hank_cumulative_loss']:.4e}`, "
                f"peak inflation `{row['peak_inflation_abs']:.4e}`, peak output gap `{row['peak_output_gap_abs']:.4e}`."
            )
        else:
            lines.append(
                f"- {row['scenario_label']}, `{row['policy_name']}`: solver не сошелся; последняя ошибка `{row['solver_error']}`."
            )
    (out / "report_full_hank_projection.md").write_text("\n".join(lines), encoding="utf-8")
    return {"full_hank_metrics": metrics, "full_hank_paths": paths}


def run_policy_extension_experiments(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_policy_extensions",
    validation_seeds: tuple[int, ...] = tuple(range(500, 510)),
    test_seeds: tuple[int, ...] = tuple(range(900, 950)),
    scenario_names: tuple[str, ...] = SCENARIO_NAMES,
    run_full_hank_projection: bool = False,
    full_hank_scenarios: tuple[str, ...] = ("thin_information_strong_gap",),
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    figures_dir = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_json(
        root / "stage6_policy_extensions_spec.json",
        {
            "validation_seeds": list(validation_seeds),
            "test_seeds": list(test_seeds),
            "scenario_names": list(scenario_names),
            "run_full_hank_projection": run_full_hank_projection,
            "full_hank_scenarios": list(full_hank_scenarios),
            "note": (
                "Rules are selected on validation trajectories and reported on held-out test trajectories. "
                "The history-based rule is recurrent through an exponentially smoothed observation state."
            ),
        },
    )

    grid_frames = []
    selected_rows = []
    metrics_frames = []
    path_frames = []
    for scenario_name in scenario_names:
        linear_params, linear_grid = _select_linear_rule(scenario_name=scenario_name, validation_seeds=validation_seeds)
        history_params, history_grid = _select_history_rule(scenario_name=scenario_name, validation_seeds=validation_seeds)
        linear_grid["rule_family"] = "optimized_linear_estimated_state"
        history_grid["rule_family"] = "history_observables_rule"
        grid_frames.extend([linear_grid, history_grid])
        selected_rows.extend(
            [
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": "optimized_linear_estimated_state",
                    **linear_params.to_dict(),
                },
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": "history_observables_rule",
                    **history_params.to_dict(),
                },
            ]
        )
        metrics, paths = _evaluate_selected_rules(
            scenario_name=scenario_name,
            validation_seeds=validation_seeds,
            test_seeds=test_seeds,
            linear_params=linear_params,
            history_params=history_params,
        )
        metrics_frames.append(metrics)
        path_frames.append(paths)

    selection_grid = pd.concat(grid_frames, ignore_index=True)
    selected_specs = pd.DataFrame(selected_rows)
    policy_metrics = pd.concat(metrics_frames, ignore_index=True)
    policy_paths = pd.concat(path_frames, ignore_index=True)
    comparison_summary = _comparison_summary(policy_metrics)
    components = _component_decomposition(policy_paths)

    selection_grid.to_csv(root / "selection_grid_results.csv", index=False)
    selected_specs.to_csv(root / "selected_rule_specs.csv", index=False)
    policy_metrics.to_csv(root / "policy_metrics.csv", index=False)
    policy_paths.to_csv(root / "policy_paths.csv", index=False)
    comparison_summary.to_csv(root / "comparison_summary.csv", index=False)
    components.to_csv(root / "component_decomposition.csv", index=False)
    _write_extension_latex_tables(root, comparison_summary, selected_specs)

    _plot_delta_intervals(comparison_summary, figures_dir / "fig_01_extended_delta_intervals")
    _plot_component_decomposition(components, figures_dir / "fig_02_extended_component_decomposition")

    full_hank_metrics = pd.DataFrame()
    full_hank_paths = pd.DataFrame()
    if run_full_hank_projection:
        full_hank_metrics, full_hank_paths = _run_full_hank_projection(
            policy_paths=policy_paths,
            output_dir=root,
            scenario_names=full_hank_scenarios,
            policy_names=(
                "classical_filtered_rule",
                "optimized_linear_estimated_state",
                "history_observables_rule",
            ),
        )

    report_lines = [
        "# Расширенные эксперименты этапа 6",
        "",
        "## Что добавлено",
        "",
        "- Оптимизированное линейное правило по оценённому состоянию.",
        "- Историческое правило по наблюдаемым переменным с внутренним сглаженным состоянием.",
        f"- Отбор выполнен на validation seeds `{validation_seeds[0]}`--`{validation_seeds[-1]}`.",
        f"- Итоговая проверка выполнена на `{len(test_seeds)}` независимых test trajectories.",
        "",
        "## Главный смысл",
        "",
        "Эти эксперименты отделяют эффект формы правила от эффекта информации. Если оптимизированное линейное правило по оценённому состоянию устойчиво выигрывает у классического правила, это означает, что важна не только фильтрация, но и настройка самой формы правила. Если историческое правило по наблюдениям догоняет правило по оценённому состоянию, значит явная фильтрация менее критична; если не догоняет, это поддерживает тезис о ценности оценки скрытого состояния.",
        "",
        "## Попарные сравнения",
        "",
    ]
    for row in comparison_summary.to_dict(orient="records"):
        if row["comparison_name"] in {"linear_minus_classical", "history_minus_classical", "linear_minus_history"}:
            report_lines.append(
                f"- {row['scenario_label']}, {row['comparison_label']}: "
                f"delta `{row['mean_delta_cumulative_loss']:.4e}`, "
                f"95% ДИ `[{row['ci_lower']:.4e}; {row['ci_upper']:.4e}]`, "
                f"доля побед `{row['win_rate']:.2f}`."
            )
    if run_full_hank_projection:
        report_lines.extend(
            [
                "",
                "## Full-HANK projection",
                "",
                "Средние тестовые траектории ставки из reduced-state экспериментов дополнительно переданы в полную HANK как экзогенные траектории monetary-policy shock. Это не заменяет полную оптимизацию в HANK, но служит проверкой того, не исчезает ли различие между правилами при пропуске через full-HANK transition solver.",
            ]
        )
    (root / "report_stage6_policy_extensions.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "selection_grid": selection_grid,
        "selected_specs": selected_specs,
        "policy_metrics": policy_metrics,
        "policy_paths": policy_paths,
        "comparison_summary": comparison_summary,
        "components": components,
        "full_hank_metrics": full_hank_metrics,
        "full_hank_paths": full_hank_paths,
    }
