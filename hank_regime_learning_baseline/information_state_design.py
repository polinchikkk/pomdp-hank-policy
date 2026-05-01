from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.steady_state import solve_steady_state
from hank_learning_policy_baseline.policies import BasePolicy
from hank_partial_info_baseline.state_space import fit_reduced_state_space
from regime_switching_baseline.regime_model import build_regime_switching_model

from .config import RegimeLearningConfig, RegimeLearningVariant
from .environment import RegimeSwitchingPolicyEnvironment, build_scenario_spec
from .evaluation import simulate_policy_episode
from .regime_config import extreme_sticky_regime_config
from .scenario_catalog import (
    information_state_design_scenario_label,
    information_state_design_scenario_names,
)


BASE_STATE_NAMES = (
    "rstar_gap",
    "productivity_gap",
    "fiscal_gap",
    "inflation_gap",
    "output_gap",
)

DIST_STATE_NAMES = BASE_STATE_NAMES + (
    "low_liquidity_gap",
    "mean_mpc_gap",
)

RULE_ORDER = (
    "history_rule",
    "posterior_mean_rule",
    "belief_interaction_rule",
    "distribution_belief_rule",
)

RULE_LABELS_RU = {
    "history_rule": "История наблюдений",
    "posterior_mean_rule": "Апостериорное среднее",
    "belief_interaction_rule": "Среднее и режимное расхождение",
    "distribution_belief_rule": "Распределительно расширенное состояние",
}

COMPARISON_SPECS = (
    ("mean_minus_history", "posterior_mean_rule", "history_rule", "Среднее минус история"),
    ("interaction_minus_mean", "belief_interaction_rule", "posterior_mean_rule", "Режимное расхождение минус среднее"),
    ("dist_minus_interaction", "distribution_belief_rule", "belief_interaction_rule", "Распределение минус режимное расхождение"),
)

TIE_EPS = 1e-12


@dataclass(frozen=True)
class LinearInformationRuleParameters:
    rule_name: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "rule_name": self.rule_name,
            "feature_names": "|".join(self.feature_names),
        }
        for feature_name, value in zip(self.feature_names, self.coefficients):
            payload[f"weight_{feature_name}"] = float(value)
        return payload

    def coefficient_map(self) -> dict[str, float]:
        return {
            feature_name: float(value)
            for feature_name, value in zip(self.feature_names, self.coefficients)
        }


class LinearInformationStatePolicy(BasePolicy):
    def __init__(self, params: LinearInformationRuleParameters) -> None:
        self.params = params

    @staticmethod
    def _state_map(info: dict, key: str) -> dict[str, float]:
        names = tuple(info["state_names"])
        values = np.asarray(info[key], dtype=float)
        return {name: float(values[index]) for index, name in enumerate(names)}

    @staticmethod
    def _signal_map(info: dict) -> dict[str, float]:
        signals: dict[str, float] = {"constant": 1.0}
        current = np.asarray(info["current_observations"], dtype=float)
        lagged = np.asarray(info["lagged_observations"], dtype=float)
        observation_names = tuple(info["noisy_observation_names"])
        for index, name in enumerate(observation_names):
            signals[f"observed_{name}"] = float(current[index])
            signals[f"lagged_observed_{name}"] = float(lagged[index])

        filtered = LinearInformationStatePolicy._state_map(info, "filtered_state")
        interaction = LinearInformationStatePolicy._state_map(info, "stress_interaction_state")
        for name in BASE_STATE_NAMES:
            signals[f"mu_{name}"] = filtered[name]
            signals[f"pdelta_{name}"] = interaction[name]
        for name in DIST_STATE_NAMES:
            signals[f"mudist_{name}"] = filtered[name]
            signals[f"pdelta_dist_{name}"] = interaction[name]

        signals["previous_rate"] = float(info["current_rate"])
        return signals

    def rate(self, observation: np.ndarray, info: dict) -> float:
        signals = self._signal_map(info)
        rate = 0.0
        for feature_name, coefficient in zip(self.params.feature_names, self.params.coefficients):
            rate += float(coefficient) * float(signals[feature_name])
        lower, upper = info["rate_bounds"]
        return float(np.clip(rate, lower, upper))


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _feature_names_for_rule(rule_name: str, observation_names: tuple[str, ...]) -> tuple[str, ...]:
    if rule_name == "history_rule":
        names: list[str] = ["constant"]
        for observation_name in observation_names:
            names.append(f"observed_{observation_name}")
            names.append(f"lagged_observed_{observation_name}")
        names.append("previous_rate")
        return tuple(names)
    if rule_name == "posterior_mean_rule":
        return ("constant",) + tuple(f"mu_{name}" for name in BASE_STATE_NAMES) + ("previous_rate",)
    if rule_name == "belief_interaction_rule":
        return (
            ("constant",)
            + tuple(f"mu_{name}" for name in BASE_STATE_NAMES)
            + tuple(f"pdelta_{name}" for name in BASE_STATE_NAMES)
            + ("previous_rate",)
        )
    if rule_name == "distribution_belief_rule":
        return (
            ("constant",)
            + tuple(f"mudist_{name}" for name in DIST_STATE_NAMES)
            + tuple(f"pdelta_dist_{name}" for name in DIST_STATE_NAMES)
            + ("previous_rate",)
        )
    raise ValueError(f"Неизвестное правило: {rule_name}")


def _initial_coefficient(feature_name: str) -> float:
    if feature_name == "constant":
        return 0.0
    if feature_name.endswith("rstar_gap"):
        return 1.0
    if feature_name.endswith("inflation_gap") or feature_name == "observed_pi":
        return 1.5
    if feature_name.endswith("output_gap") or feature_name == "observed_output_gap":
        return 0.25
    if feature_name == "previous_rate":
        return 0.6
    return 0.0


def _candidate_values(feature_name: str) -> tuple[float, ...]:
    if feature_name == "constant":
        return (-0.001, -0.0005, 0.0, 0.0005, 0.001)
    if feature_name == "previous_rate":
        return (0.0, 0.3, 0.6, 0.85)
    if feature_name.endswith("rstar_gap"):
        return (0.0, 0.5, 1.0, 1.5)
    if feature_name.endswith("inflation_gap") or feature_name == "observed_pi":
        return (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    if feature_name.endswith("output_gap") or feature_name == "observed_output_gap":
        return (0.0, 0.125, 0.25, 0.5)
    if feature_name.endswith("low_liquidity_gap") or feature_name.endswith("mean_mpc_gap"):
        return (-0.5, -0.25, 0.0, 0.25, 0.5)
    if feature_name.startswith("pdelta"):
        return (-1.0, -0.5, 0.0, 0.5, 1.0)
    return (-0.5, -0.25, 0.0, 0.25, 0.5)


def _bootstrap_ci(values: np.ndarray, *, seed: int = 2027, draws: int = 2000) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _make_env_factory(
    *,
    scenario_name: str,
    action_bound: float,
    horizon: int,
    noise_scale_multiplier: float,
):
    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    regime_config = extreme_sticky_regime_config()
    reduced_model = fit_reduced_state_space(bundle, hank_config, regime_config.partial_config)
    variant = RegimeLearningVariant(
        name=f"{scenario_name}_information_state_design",
        scenario_name=scenario_name,
        scenario_label=information_state_design_scenario_label(scenario_name),
        input_mode="observed_information_state",
        include_distributional_state=True,
        description="Сравнение сжатых информационных состояний для правила ставки.",
    )
    config = RegimeLearningConfig(
        action_bound=action_bound,
        horizon=horizon,
        regime_config=regime_config,
        training_seeds=(),
        selection_seeds=(),
        evaluation_seeds=(),
    )
    scenario_spec = build_scenario_spec(config, variant)
    if noise_scale_multiplier != 1.0:
        from dataclasses import replace

        scenario_spec = replace(
            scenario_spec,
            noise_scale=float(scenario_spec.noise_scale * noise_scale_multiplier),
            scenario_name=f"{scenario_name}_noise_{noise_scale_multiplier:g}",
            scenario_label=f"{scenario_spec.scenario_label} × шум {noise_scale_multiplier:g}",
        )
    model = build_regime_switching_model(reduced_model, regime_config, scenario_spec.gap_scale)

    def factory() -> RegimeSwitchingPolicyEnvironment:
        return RegimeSwitchingPolicyEnvironment(
            model=model,
            regime_config=regime_config,
            scenario_spec=scenario_spec,
            phi_pi=hank_config.phi_pi,
            phi_y=hank_config.phi_y,
            rho_i=hank_config.rho_i,
        )

    return factory, scenario_spec


def _evaluate_params(
    *,
    env_factory,
    scenario_spec,
    params: LinearInformationRuleParameters,
    seeds: tuple[int, ...],
) -> tuple[float, pd.DataFrame]:
    policy = LinearInformationStatePolicy(params)
    traces = []
    losses = []
    for seed in seeds:
        trace = simulate_policy_episode(
            env_factory=env_factory,
            policy=policy,
            scenario_spec=scenario_spec,
            evaluation_seed=int(seed),
            policy_name=params.rule_name,
            policy_label=RULE_LABELS_RU[params.rule_name],
            training_seed=None,
        )
        traces.append(trace)
        losses.append(float(trace["loss"].sum()))
    return float(np.mean(losses)), pd.concat(traces, ignore_index=True)


def _select_rule(
    *,
    env_factory,
    scenario_spec,
    rule_name: str,
    validation_seeds: tuple[int, ...],
    max_rounds: int,
) -> tuple[LinearInformationRuleParameters, pd.DataFrame]:
    observation_names = tuple(scenario_spec.noisy_observations)
    feature_names = _feature_names_for_rule(rule_name, observation_names)
    coefficients = {
        feature_name: _initial_coefficient(feature_name)
        for feature_name in feature_names
    }
    rows = []
    best_loss, _ = _evaluate_params(
        env_factory=env_factory,
        scenario_spec=scenario_spec,
        params=LinearInformationRuleParameters(
            rule_name=rule_name,
            feature_names=feature_names,
            coefficients=tuple(coefficients[name] for name in feature_names),
        ),
        seeds=validation_seeds,
    )
    rows.append({"rule_name": rule_name, "round": 0, "feature_name": "initial", "candidate_value": math.nan, "validation_loss": best_loss})

    for round_index in range(1, max_rounds + 1):
        improved = False
        for feature_name in feature_names:
            feature_best = coefficients[feature_name]
            feature_best_loss = best_loss
            for candidate_value in _candidate_values(feature_name):
                trial = dict(coefficients)
                trial[feature_name] = float(candidate_value)
                loss, _ = _evaluate_params(
                    env_factory=env_factory,
                    scenario_spec=scenario_spec,
                    params=LinearInformationRuleParameters(
                        rule_name=rule_name,
                        feature_names=feature_names,
                        coefficients=tuple(trial[name] for name in feature_names),
                    ),
                    seeds=validation_seeds,
                )
                rows.append(
                    {
                        "rule_name": rule_name,
                        "round": round_index,
                        "feature_name": feature_name,
                        "candidate_value": float(candidate_value),
                        "validation_loss": loss,
                    }
                )
                if loss < feature_best_loss:
                    feature_best_loss = loss
                    feature_best = float(candidate_value)
            if feature_best_loss < best_loss:
                coefficients[feature_name] = feature_best
                best_loss = feature_best_loss
                improved = True
        if not improved:
            break

    params = LinearInformationRuleParameters(
        rule_name=rule_name,
        feature_names=feature_names,
        coefficients=tuple(coefficients[name] for name in feature_names),
    )
    return params, pd.DataFrame(rows)


def _summarize_metrics(paths: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario_name, scenario_label, rule_name, rule_label, seed), frame in paths.groupby(
        ["scenario_name", "scenario_label", "policy_name", "policy_label", "evaluation_seed"]
    ):
        rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": scenario_label,
                "rule_name": rule_name,
                "rule_label": rule_label,
                "evaluation_seed": int(seed),
                "cumulative_loss": float(frame["loss"].sum()),
                "inflation_loss": float(frame["inflation_loss"].sum()),
                "output_gap_loss": float(frame["output_gap_loss"].sum()),
                "rate_change_loss": float(frame["rate_change_loss"].sum()),
                "mean_stress_probability": float(frame["stress_probability"].mean()),
                "median_stress_probability": float(frame["stress_probability"].median()),
                "p10_stress_probability": float(frame["stress_probability"].quantile(0.1)),
                "p90_stress_probability": float(frame["stress_probability"].quantile(0.9)),
                "ambiguous_regime_share": float(frame["stress_probability"].between(0.2, 0.8).mean()),
                "mean_delta_norm": float(frame["regime_mean_delta_norm"].mean()),
                "median_delta_norm": float(frame["regime_mean_delta_norm"].median()),
                "p90_delta_norm": float(frame["regime_mean_delta_norm"].quantile(0.9)),
            }
        )
    return pd.DataFrame(rows)


def _pairwise(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot = metrics.pivot_table(
        index=["scenario_name", "scenario_label", "evaluation_seed"],
        columns="rule_name",
        values="cumulative_loss",
        aggfunc="first",
    )
    for comparison_name, left, right, label in COMPARISON_SPECS:
        if left not in pivot.columns or right not in pivot.columns:
            continue
        deltas = (pivot[left] - pivot[right]).dropna().to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(deltas)
        grouped = (pivot[left] - pivot[right]).dropna().reset_index(name="delta")
        for (scenario_name, scenario_label), frame in grouped.groupby(["scenario_name", "scenario_label"]):
            scenario_deltas = frame["delta"].to_numpy(dtype=float)
            scenario_ci_low, scenario_ci_high = _bootstrap_ci(scenario_deltas)
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": scenario_label,
                    "comparison_name": comparison_name,
                    "comparison_label": label,
                    "left_rule": left,
                    "right_rule": right,
                    "mean_delta": float(np.mean(scenario_deltas)),
                    "ci_low": scenario_ci_low,
                    "ci_high": scenario_ci_high,
                    "win_rate": float(np.mean(scenario_deltas < -TIE_EPS)),
                    "tie_rate": float(np.mean(np.abs(scenario_deltas) <= TIE_EPS)),
                    "loss_rate": float(np.mean(scenario_deltas > TIE_EPS)),
                    "num_test_trajectories": int(scenario_deltas.size),
                }
            )
        rows.append(
            {
                "scenario_name": "all_scenarios",
                "scenario_label": "Все сценарии",
                "comparison_name": comparison_name,
                "comparison_label": label,
                "left_rule": left,
                "right_rule": right,
                "mean_delta": float(np.mean(deltas)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "win_rate": float(np.mean(deltas < -TIE_EPS)),
                "tie_rate": float(np.mean(np.abs(deltas) <= TIE_EPS)),
                "loss_rate": float(np.mean(deltas > TIE_EPS)),
                "num_test_trajectories": int(deltas.size),
            }
        )
    return pd.DataFrame(rows)


def _component_decomposition(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    value_columns = (
        "cumulative_loss",
        "inflation_loss",
        "output_gap_loss",
        "rate_change_loss",
    )
    for comparison_name, left, right, label in COMPARISON_SPECS:
        for (scenario_name, scenario_label), scenario_metrics in metrics.groupby(["scenario_name", "scenario_label"]):
            row = {
                "scenario_name": scenario_name,
                "scenario_label": scenario_label,
                "comparison_name": comparison_name,
                "comparison_label": label,
                "left_rule": left,
                "right_rule": right,
            }
            complete = True
            for value_column in value_columns:
                pivot = scenario_metrics.pivot_table(
                    index="evaluation_seed",
                    columns="rule_name",
                    values=value_column,
                    aggfunc="first",
                )
                if left not in pivot.columns or right not in pivot.columns:
                    complete = False
                    break
                deltas = (pivot[left] - pivot[right]).dropna().to_numpy(dtype=float)
                if deltas.size == 0:
                    complete = False
                    break
                row[f"mean_delta_{value_column}"] = float(np.mean(deltas))
                row[f"num_{value_column}"] = int(deltas.size)
            if complete:
                rows.append(row)
    return pd.DataFrame(rows)


def _levels(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["scenario_name", "scenario_label", "rule_name", "rule_label"], as_index=False)
        .agg(
            mean_cumulative_loss=("cumulative_loss", "mean"),
            std_cumulative_loss=("cumulative_loss", "std"),
            mean_inflation_loss=("inflation_loss", "mean"),
            mean_output_gap_loss=("output_gap_loss", "mean"),
            mean_rate_change_loss=("rate_change_loss", "mean"),
            mean_stress_probability=("mean_stress_probability", "mean"),
            median_stress_probability=("median_stress_probability", "mean"),
            p10_stress_probability=("p10_stress_probability", "mean"),
            p90_stress_probability=("p90_stress_probability", "mean"),
            ambiguous_regime_share=("ambiguous_regime_share", "mean"),
            mean_delta_norm=("mean_delta_norm", "mean"),
            median_delta_norm=("median_delta_norm", "mean"),
            p90_delta_norm=("p90_delta_norm", "mean"),
            num_test_trajectories=("evaluation_seed", "nunique"),
        )
        .sort_values(["scenario_name", "mean_cumulative_loss"])
        .reset_index(drop=True)
    )


def _write_table(df: pd.DataFrame, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path_without_suffix.with_suffix(".csv"), index=False)
    try:
        df.to_latex(
            path_without_suffix.with_suffix(".tex"),
            index=False,
            escape=False,
        )
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        path_without_suffix.with_suffix(".tex.error.txt").write_text(str(exc), encoding="utf-8")


def _write_article_tables(
    *,
    root: Path,
    levels: pd.DataFrame,
    pairwise: pd.DataFrame,
    decomposition: pd.DataFrame,
) -> None:
    tables_dir = root / "tables"
    state_components = pd.DataFrame(
        [
            {"Компонента": r"$r_t^*$", "Тип": "скрытая", "Смысл": "естественная ставка", "Где используется": "базовое и расширенное состояние"},
            {"Компонента": r"$z_t$", "Тип": "скрытая", "Смысл": "фактор производительности", "Где используется": "базовое и расширенное состояние"},
            {"Компонента": r"$f_t$", "Тип": "скрытая", "Смысл": "фискальный фактор", "Где используется": "базовое и расширенное состояние"},
            {"Компонента": r"$\pi_t^{gap}$", "Тип": "макроэкономическая", "Смысл": "инфляционный разрыв", "Где используется": "состояние и функция потерь"},
            {"Компонента": r"$y_t^{gap}$", "Тип": "макроэкономическая", "Смысл": "разрыв выпуска", "Где используется": "состояние и функция потерь"},
            {"Компонента": r"$s_t^{liq}$", "Тип": "распределительная", "Смысл": "доля низколиквидных домохозяйств", "Где используется": "распределительное расширение"},
            {"Компонента": r"$m_t^{mpc}$", "Тип": "распределительная", "Смысл": "средняя предельная склонность к потреблению", "Где используется": "распределительное расширение"},
            {"Компонента": r"$r_t$", "Тип": "скрытый режим", "Смысл": "нормальный или стрессовый режим", "Где используется": "динамика состояния"},
            {"Компонента": r"$p_t$", "Тип": "оценка фильтра", "Смысл": "вероятность стрессового режима", "Где используется": "режимное расхождение"},
        ]
    )
    rule_descriptions = pd.DataFrame(
        [
            {
                "Правило": RULE_LABELS_RU["history_rule"],
                "Вход": r"$(y_t^{obs}, y_{t-1}^{obs}, i_{t-1})$",
                "Что проверяет": "достаточно ли короткой истории наблюдений без явного фильтра",
            },
            {
                "Правило": RULE_LABELS_RU["posterior_mean_rule"],
                "Вход": r"$(\mu_t, i_{t-1})$",
                "Что проверяет": "ценность фильтра как сжатия истории наблюдений",
            },
            {
                "Правило": RULE_LABELS_RU["belief_interaction_rule"],
                "Вход": r"$(\mu_t, p_t\delta_t, i_{t-1})$",
                "Что проверяет": "ценность режимного расхождения условных оценок",
            },
            {
                "Правило": RULE_LABELS_RU["distribution_belief_rule"],
                "Вход": r"$(\mu_t^{dist}, p_t\delta_t^{dist}, i_{t-1})$",
                "Что проверяет": "ценность распределительных характеристик",
            },
        ]
    )
    main_results = levels[
        [
            "scenario_label",
            "rule_label",
            "mean_cumulative_loss",
            "std_cumulative_loss",
            "num_test_trajectories",
        ]
    ].rename(
        columns={
            "scenario_label": "Сценарий",
            "rule_label": "Правило",
            "mean_cumulative_loss": "Средняя накопленная потеря",
            "std_cumulative_loss": "Стандартное отклонение",
            "num_test_trajectories": "Число траекторий",
        }
    )
    pairwise_results = pairwise[
        [
            "scenario_label",
            "comparison_label",
            "mean_delta",
            "ci_low",
            "ci_high",
            "win_rate",
            "tie_rate",
            "loss_rate",
            "num_test_trajectories",
        ]
    ].rename(
        columns={
            "scenario_label": "Сценарий",
            "comparison_label": "Сравнение",
            "mean_delta": "Средняя разность",
            "ci_low": "Нижняя граница",
            "ci_high": "Верхняя граница",
            "win_rate": "Доля выигрышей",
            "tie_rate": "Доля совпадений",
            "loss_rate": "Доля проигрышей",
            "num_test_trajectories": "Число траекторий",
        }
    )
    loss_decomposition = decomposition[
        [
            "scenario_label",
            "comparison_label",
            "mean_delta_inflation_loss",
            "mean_delta_output_gap_loss",
            "mean_delta_rate_change_loss",
            "mean_delta_cumulative_loss",
        ]
    ].rename(
        columns={
            "scenario_label": "Сценарий",
            "comparison_label": "Сравнение",
            "mean_delta_inflation_loss": "Инфляция",
            "mean_delta_output_gap_loss": "Разрыв выпуска",
            "mean_delta_rate_change_loss": "Изменение ставки",
            "mean_delta_cumulative_loss": "Итого",
        }
    )
    _write_table(state_components, tables_dir / "table_state_components")
    _write_table(rule_descriptions, tables_dir / "table_policy_rules")
    _write_table(main_results, tables_dir / "table_main_results")
    _write_table(pairwise_results, tables_dir / "table_pairwise_results")
    _write_table(loss_decomposition, tables_dir / "table_loss_decomposition")


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def _write_figures(
    *,
    root: Path,
    levels: pd.DataFrame,
    policy_paths: pd.DataFrame,
    decomposition: pd.DataFrame,
) -> None:
    plt = _load_pyplot()
    if plt is None:
        (root / "figures_unavailable.txt").write_text(
            "matplotlib недоступен, рисунки не построены.",
            encoding="utf-8",
        )
        return

    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    ordered_levels = levels.copy()
    ordered_levels["rule_label"] = pd.Categorical(
        ordered_levels["rule_label"],
        categories=[RULE_LABELS_RU[name] for name in RULE_ORDER],
        ordered=True,
    )
    pivot = ordered_levels.pivot_table(
        index="scenario_label",
        columns="rule_label",
        values="mean_cumulative_loss",
        aggfunc="first",
        observed=False,
    )
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(9.5, max(4.0, 0.7 * len(pivot.index))))
        pivot.plot(kind="barh", ax=ax)
        ax.set_xlabel("Средняя накопленная потеря")
        ax.set_ylabel("Сценарий")
        ax.set_title("Сравнение информационных состояний")
        ax.legend(title="Правило", fontsize=8)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(figures_dir / "fig_main_comparison.pdf")
        fig.savefig(figures_dir / "fig_main_comparison.png", dpi=180)
        plt.close(fig)

    if not policy_paths.empty:
        diagnostic_paths = policy_paths[policy_paths["policy_name"] == "posterior_mean_rule"]
        if diagnostic_paths.empty:
            diagnostic_paths = policy_paths
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.hist(diagnostic_paths["stress_probability"].dropna(), bins=30, color="#356859", alpha=0.85)
        ax.set_xlabel("Вероятность стрессового режима")
        ax.set_ylabel("Число наблюдений")
        ax.set_title("Диагностика режимной вероятности")
        fig.tight_layout()
        fig.savefig(figures_dir / "fig_stress_probability_histogram.pdf")
        fig.savefig(figures_dir / "fig_stress_probability_histogram.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        ax.hist(diagnostic_paths["regime_mean_delta_norm"].dropna(), bins=30, color="#8a5a44", alpha=0.85)
        ax.set_xlabel("Норма режимного расхождения")
        ax.set_ylabel("Число наблюдений")
        ax.set_title("Диагностика расхождения условных оценок")
        fig.tight_layout()
        fig.savefig(figures_dir / "fig_regime_delta_histogram.pdf")
        fig.savefig(figures_dir / "fig_regime_delta_histogram.png", dpi=180)
        plt.close(fig)

    if not decomposition.empty:
        component_columns = [
            "mean_delta_inflation_loss",
            "mean_delta_output_gap_loss",
            "mean_delta_rate_change_loss",
        ]
        aggregate = decomposition.groupby("comparison_label", as_index=True)[component_columns].mean()
        if not aggregate.empty:
            aggregate = aggregate.rename(
                columns={
                    "mean_delta_inflation_loss": "Инфляция",
                    "mean_delta_output_gap_loss": "Разрыв выпуска",
                    "mean_delta_rate_change_loss": "Изменение ставки",
                }
            )
            fig, ax = plt.subplots(figsize=(8.5, 4.5))
            aggregate.plot(kind="bar", ax=ax)
            ax.axhline(0.0, color="black", linewidth=0.8)
            ax.set_xlabel("Сравнение")
            ax.set_ylabel("Вклад в разность потерь")
            ax.set_title("Разложение разности потерь")
            ax.legend(title="Компонента", fontsize=8)
            fig.tight_layout()
            fig.savefig(figures_dir / "fig_loss_decomposition.pdf")
            fig.savefig(figures_dir / "fig_loss_decomposition.png", dpi=180)
            plt.close(fig)


def run_information_state_design(
    *,
    output_dir: str = "outputs/information_state_design_main",
    scenario_names: tuple[str, ...] = information_state_design_scenario_names(),
    validation_seeds: tuple[int, ...] = tuple(range(500, 510)),
    test_seeds: tuple[int, ...] = tuple(range(900, 950)),
    action_bound: float = 0.0015,
    horizon: int = 60,
    max_rounds: int = 2,
    noise_scale_multiplier: float = 1.0,
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _save_json(
        root / "information_state_design_spec.json",
        {
            "scenario_names": list(scenario_names),
            "validation_seeds": list(validation_seeds),
            "test_seeds": list(test_seeds),
            "rules": [
                {"name": rule_name, "label_ru": RULE_LABELS_RU[rule_name]}
                for rule_name in RULE_ORDER
            ],
            "base_state_names": list(BASE_STATE_NAMES),
            "distribution_state_names": list(DIST_STATE_NAMES),
            "rule_selection": "Все линейные правила подбираются на валидационных траекториях и затем фиксируются.",
            "comparisons": [
                {"name": name, "left": left, "right": right, "label_ru": label}
                for name, left, right, label in COMPARISON_SPECS
            ],
            "action_bound": float(action_bound),
            "horizon": int(horizon),
            "max_rounds": int(max_rounds),
            "noise_scale_multiplier": float(noise_scale_multiplier),
        },
    )

    selection_frames = []
    selected_rows = []
    path_frames = []
    for scenario_name in scenario_names:
        env_factory, scenario_spec = _make_env_factory(
            scenario_name=scenario_name,
            action_bound=action_bound,
            horizon=horizon,
            noise_scale_multiplier=noise_scale_multiplier,
        )
        for rule_name in RULE_ORDER:
            params, selection = _select_rule(
                env_factory=env_factory,
                scenario_spec=scenario_spec,
                rule_name=rule_name,
                validation_seeds=validation_seeds,
                max_rounds=max_rounds,
            )
            selection.insert(0, "scenario_name", scenario_spec.scenario_name)
            selection.insert(1, "scenario_label", scenario_spec.scenario_label)
            selection_frames.append(selection)
            selected_rows.append(
                {
                    "scenario_name": scenario_spec.scenario_name,
                    "scenario_label": scenario_spec.scenario_label,
                    "rule_name": rule_name,
                    "rule_label": RULE_LABELS_RU[rule_name],
                    **params.to_dict(),
                }
            )
            _, test_paths = _evaluate_params(
                env_factory=env_factory,
                scenario_spec=scenario_spec,
                params=params,
                seeds=test_seeds,
            )
            path_frames.append(test_paths)

    selection_grid = pd.concat(selection_frames, ignore_index=True)
    selected_rules = pd.DataFrame(selected_rows)
    policy_paths = pd.concat(path_frames, ignore_index=True)
    seed_metrics = _summarize_metrics(policy_paths)
    levels = _levels(seed_metrics)
    pairwise = _pairwise(seed_metrics)
    decomposition = _component_decomposition(seed_metrics)

    selection_grid.to_csv(root / "selection_grid.csv", index=False)
    selected_rules.to_csv(root / "selected_rules.csv", index=False)
    policy_paths.to_csv(root / "policy_paths.csv", index=False)
    seed_metrics.to_csv(root / "seed_level_metrics.csv", index=False)
    levels.to_csv(root / "information_state_levels.csv", index=False)
    pairwise.to_csv(root / "information_state_pairwise.csv", index=False)
    decomposition.to_csv(root / "loss_component_decomposition.csv", index=False)
    _write_article_tables(root=root, levels=levels, pairwise=pairwise, decomposition=decomposition)
    _write_figures(root=root, levels=levels, policy_paths=policy_paths, decomposition=decomposition)

    lines = [
        "# Дизайн информационного состояния",
        "",
        "Сравниваются четыре линейных правила: история наблюдений, апостериорное среднее, среднее с режимным расхождением и распределительно расширенное состояние.",
        "",
        "Все правила сначала подбираются на валидационных траекториях, затем фиксируются и оцениваются на независимых тестовых траекториях.",
        "",
        "## Главные сравнения",
        "",
    ]
    for row in pairwise[pairwise["scenario_name"] == "all_scenarios"].to_dict(orient="records"):
        lines.append(
            f"- {row['comparison_label']}: средняя разность {row['mean_delta']:.4e}, "
            f"95%-й интервал [{row['ci_low']:.4e}, {row['ci_high']:.4e}], "
            f"доли выигрыш/совпадение/проигрыш "
            f"{row['win_rate']:.2f}/{row['tie_rate']:.2f}/{row['loss_rate']:.2f}."
        )
    lines.extend(
        [
            "",
            "## Диагностика фильтра",
            "",
            "В таблице `information_state_levels.csv` сохранены средняя вероятность стрессового режима, доля неопределённых периодов и норма режимного расхождения.",
            "",
            "Готовые таблицы для текста лежат в папке `tables`, рисунки — в папке `figures`.",
        ]
    )
    (root / "report_information_state_design.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "selection_grid": selection_grid,
        "selected_rules": selected_rules,
        "policy_paths": policy_paths,
        "seed_level_metrics": seed_metrics,
        "information_state_levels": levels,
        "information_state_pairwise": pairwise,
        "loss_component_decomposition": decomposition,
    }
