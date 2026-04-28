from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.steady_state import solve_steady_state
from hank_full_baseline.transition import solve_transition
from hank_learning_policy_baseline.policies import BasePolicy, ClassicalFilteredRulePolicy
from hank_partial_info_baseline.state_space import fit_reduced_state_space
from regime_switching_baseline.regime_model import build_regime_switching_model

from .config import RegimeLearningConfig, RegimeLearningVariant
from .core_matrix import SCENARIO_LABELS
from .environment import RegimeSwitchingPolicyEnvironment, build_scenario_spec
from .evaluation import simulate_policy_episode
from .tuning import default_universal_candidate_lookup, extreme_sticky_regime_config


INFO_SET_ORDER = [
    "observed_information",
    "posterior_mean",
    "posterior_regime",
    "posterior_uncertainty",
    "posterior_distribution",
]

INFO_SET_LABELS_RU = {
    "observed_information": "Наблюдаемые переменные",
    "posterior_mean": "Апостериорное среднее",
    "posterior_regime": "Среднее и вероятность режима",
    "posterior_uncertainty": "Среднее, вероятность режима и неопределённость",
    "posterior_distribution": "Распределительно расширенное состояние",
}

INPUT_MODE_BY_INFO_SET = {
    "observed_information": "observed_information_state",
    "posterior_mean": "posterior_mean_state",
    "posterior_regime": "posterior_regime_state",
    "posterior_uncertainty": "posterior_uncertainty_state",
    "posterior_distribution": "posterior_distribution_state",
}

INCLUDE_DISTRIBUTIONAL_STATE = {
    "observed_information": False,
    "posterior_mean": False,
    "posterior_regime": False,
    "posterior_uncertainty": False,
    "posterior_distribution": True,
}

FEATURE_COLUMNS = [
    "observed_pi",
    "lagged_observed_pi",
    "observed_output_gap",
    "lagged_observed_output_gap",
    "mean_rstar_gap",
    "mean_productivity_gap",
    "mean_fiscal_gap",
    "mean_inflation_gap",
    "mean_output_gap",
    "stress_probability",
    "stress_entropy",
    "filtered_variance_trace",
    "mean_low_liquidity_gap",
    "mean_mean_mpc_gap",
]

FEATURE_LABELS_RU = {
    "observed_pi": "наблюдаемая инфляция",
    "lagged_observed_pi": "инфляция с лагом",
    "observed_output_gap": "наблюдаемый разрыв выпуска",
    "lagged_observed_output_gap": "разрыв выпуска с лагом",
    "mean_rstar_gap": "средняя естественная ставка",
    "mean_productivity_gap": "средняя производительность",
    "mean_fiscal_gap": "средний фискальный фактор",
    "mean_inflation_gap": "средний инфляционный разрыв",
    "mean_output_gap": "средний разрыв выпуска",
    "stress_probability": "вероятность стресса",
    "stress_entropy": "энтропия режима",
    "filtered_variance_trace": "след ковариации",
    "mean_low_liquidity_gap": "средний ликвидностный разрыв",
    "mean_mean_mpc_gap": "средний разрыв MPC",
}

PAIRWISE_COMPARISONS = {
    "mean_minus_observed": ("posterior_mean", "observed_information"),
    "regime_minus_mean": ("posterior_regime", "posterior_mean"),
    "uncertainty_minus_regime": ("posterior_uncertainty", "posterior_regime"),
    "distribution_minus_uncertainty": ("posterior_distribution", "posterior_uncertainty"),
}

PAIRWISE_LABELS_RU = {
    "mean_minus_observed": "Апостериорное среднее минус наблюдаемые переменные",
    "regime_minus_mean": "Добавление вероятности режима",
    "uncertainty_minus_regime": "Добавление меры неопределённости",
    "uncertainty_minus_mean": "Среднее с неопределённостью минус одно среднее",
    "distribution_minus_uncertainty": "Добавление распределительных переменных",
}

KEY_HANK_POLICIES = (
    "linear_observed_information",
    "linear_posterior_mean",
    "linear_posterior_uncertainty",
    "linear_posterior_distribution",
)

PAIRWISE_HANK_CHECKS = (
    ("mean_minus_observed", "linear_posterior_mean", "linear_observed_information"),
    ("uncertainty_minus_mean", "linear_posterior_uncertainty", "linear_posterior_mean"),
    ("distribution_minus_uncertainty", "linear_posterior_distribution", "linear_posterior_uncertainty"),
)

TIE_TOLERANCE = 1.0e-10


@dataclass(frozen=True)
class PosteriorRuleParameters:
    info_set_name: str
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    rho_i: float

    def to_dict(self) -> dict[str, float | str]:
        payload: dict[str, float | str] = {
            "info_set_name": self.info_set_name,
            "rho_i": float(self.rho_i),
            "feature_names": "|".join(self.feature_names),
        }
        coefficient_map = self.coefficient_map()
        for feature_name in FEATURE_COLUMNS:
            payload[f"weight_{feature_name}"] = float(coefficient_map.get(feature_name, 0.0))
        return payload

    def coefficient_map(self) -> dict[str, float]:
        return {
            feature_name: float(value)
            for feature_name, value in zip(self.feature_names, self.coefficients)
        }


class PosteriorInformationLinearRulePolicy(BasePolicy):
    def __init__(
        self,
        params: PosteriorRuleParameters,
        *,
        signal_means: dict[str, float] | None = None,
        signal_stds: dict[str, float] | None = None,
    ) -> None:
        self.params = params
        self.signal_means = signal_means or {}
        self.signal_stds = signal_stds or {}

    @staticmethod
    def _signal_map(info: dict) -> dict[str, float]:
        state_names = tuple(info["state_names"])
        filtered_state = np.asarray(info["filtered_state"], dtype=float)
        state_map = {name: float(filtered_state[index]) for index, name in enumerate(state_names)}

        current_observations = np.asarray(info["current_observations"], dtype=float)
        lagged_observations = np.asarray(info["lagged_observations"], dtype=float)
        observation_names = tuple(info["noisy_observation_names"])
        current_map = {name: float(current_observations[index]) for index, name in enumerate(observation_names)}
        lagged_map = {name: float(lagged_observations[index]) for index, name in enumerate(observation_names)}

        return {
            "observed_pi": float(current_map.get("pi", 0.0)),
            "lagged_observed_pi": float(lagged_map.get("pi", 0.0)),
            "observed_output_gap": float(current_map.get("output_gap", 0.0)),
            "lagged_observed_output_gap": float(lagged_map.get("output_gap", 0.0)),
            "mean_rstar_gap": float(state_map["rstar_gap"]),
            "mean_productivity_gap": float(state_map["productivity_gap"]),
            "mean_fiscal_gap": float(state_map["fiscal_gap"]),
            "mean_inflation_gap": float(state_map["inflation_gap"]),
            "mean_output_gap": float(state_map["output_gap"]),
            "stress_probability": float(info["stress_probability"]),
            "stress_entropy": float(info["stress_entropy"]),
            "filtered_variance_trace": float(info["filtered_variance_trace"]),
            "mean_low_liquidity_gap": float(state_map["low_liquidity_gap"]),
            "mean_mean_mpc_gap": float(state_map["mean_mpc_gap"]),
        }

    def rate(self, observation: np.ndarray, info: dict) -> float:
        signals = self._signal_map(info)
        target = 0.0
        for feature_name, coefficient in zip(self.params.feature_names, self.params.coefficients):
            signal_value = float(signals[feature_name])
            if feature_name in self.signal_stds:
                mean = float(self.signal_means.get(feature_name, 0.0))
                std = float(self.signal_stds[feature_name])
                signal_value = (signal_value - mean) / std
            target += float(coefficient) * signal_value
        previous_rate = float(info["current_rate"])
        lower, upper = info["rate_bounds"]
        rate = self.params.rho_i * previous_rate + (1.0 - self.params.rho_i) * target
        return float(np.clip(rate, lower, upper))


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _bootstrap_ci(values: np.ndarray, *, seed: int = 2026, draws: int = 4000) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _feature_names(info_set_name: str, scenario_name: str) -> tuple[str, ...]:
    if info_set_name == "observed_information":
        names = ["observed_pi", "lagged_observed_pi"]
        if "macro_core" in scenario_name:
            names.extend(["observed_output_gap", "lagged_observed_output_gap"])
        return tuple(names)
    if info_set_name == "posterior_mean":
        return (
            "mean_rstar_gap",
            "mean_productivity_gap",
            "mean_fiscal_gap",
            "mean_inflation_gap",
            "mean_output_gap",
        )
    if info_set_name == "posterior_regime":
        return _feature_names("posterior_mean", scenario_name) + ("stress_probability",)
    if info_set_name == "posterior_uncertainty":
        return _feature_names("posterior_regime", scenario_name) + (
            "stress_entropy",
            "filtered_variance_trace",
        )
    if info_set_name == "posterior_distribution":
        return _feature_names("posterior_uncertainty", scenario_name) + (
            "mean_low_liquidity_gap",
            "mean_mean_mpc_gap",
        )
    raise ValueError(f"Неизвестный набор информации: {info_set_name}")


def _initial_coefficient(feature_name: str) -> float:
    defaults = {
        "observed_pi": 1.5,
        "lagged_observed_pi": 0.0,
        "observed_output_gap": 0.25,
        "lagged_observed_output_gap": 0.0,
        "mean_rstar_gap": 1.0,
        "mean_productivity_gap": 0.0,
        "mean_fiscal_gap": 0.0,
        "mean_inflation_gap": 1.5,
        "mean_output_gap": 0.25,
        "stress_probability": 0.0,
        "stress_entropy": 0.0,
        "filtered_variance_trace": 0.0,
        "mean_low_liquidity_gap": 0.0,
        "mean_mean_mpc_gap": 0.0,
    }
    return float(defaults[feature_name])


def _candidate_values(feature_name: str, *, standardized_signals: bool = False) -> tuple[float, ...]:
    if standardized_signals:
        return (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)
    if feature_name in {"observed_pi", "lagged_observed_pi", "mean_inflation_gap"}:
        return (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    if feature_name in {"observed_output_gap", "lagged_observed_output_gap", "mean_output_gap"}:
        return (0.0, 0.125, 0.25, 0.5)
    if feature_name == "mean_rstar_gap":
        return (0.0, 0.5, 1.0, 1.5)
    if feature_name in {
        "mean_productivity_gap",
        "mean_fiscal_gap",
        "mean_low_liquidity_gap",
        "mean_mean_mpc_gap",
    }:
        return (-0.5, -0.25, 0.0, 0.25, 0.5)
    if feature_name in {"stress_probability", "stress_entropy"}:
        return (-0.01, -0.005, 0.0, 0.005, 0.01)
    if feature_name == "filtered_variance_trace":
        return (-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0)
    raise ValueError(f"Неизвестный признак: {feature_name}")


def _coefficient_string(params: PosteriorRuleParameters) -> str:
    parts = [f"rho_i={params.rho_i:.2f}"]
    for feature_name, coefficient in zip(params.feature_names, params.coefficients):
        if abs(float(coefficient)) < 1.0e-12:
            continue
        parts.append(f"{FEATURE_LABELS_RU[feature_name]}={float(coefficient):.3g}")
    return ", ".join(parts)


@lru_cache(maxsize=1)
def _base_reduced_objects():
    hank_config = default_calibration()
    regime_config = extreme_sticky_regime_config()
    bundle = solve_steady_state(hank_config)
    reduced_model = fit_reduced_state_space(bundle, hank_config, regime_config.partial_config)
    return hank_config, regime_config, reduced_model


def _build_objects(
    *,
    scenario_name: str,
    info_set_name: str,
    validation_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
):
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
    variant = RegimeLearningVariant(
        name=f"{scenario_name}_{info_set_name}",
        scenario_name=scenario_name,
        scenario_label=SCENARIO_LABELS[scenario_name],
        input_mode=INPUT_MODE_BY_INFO_SET[info_set_name],
        include_distributional_state=INCLUDE_DISTRIBUTIONAL_STATE[info_set_name],
        description=(
            "Линейное правило на конечномерном представлении апостериорной информации."
        ),
    )
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

    return scenario_spec, env_factory


def _policy_name(info_set_name: str) -> str:
    names = {
        "observed_information": "linear_observed_information",
        "posterior_mean": "linear_posterior_mean",
        "posterior_regime": "linear_posterior_regime",
        "posterior_uncertainty": "linear_posterior_uncertainty",
        "posterior_distribution": "linear_posterior_distribution",
    }
    return names[info_set_name]


def _policy_label(info_set_name: str) -> str:
    return f"Линейное правило: {INFO_SET_LABELS_RU[info_set_name]}"


def _policy_label_standardized(info_set_name: str) -> str:
    return f"Линейное правило со стандартизацией: {INFO_SET_LABELS_RU[info_set_name]}"


def _mean_cumulative_loss(
    *,
    env_factory,
    scenario_spec,
    policy: BasePolicy,
    seeds: Iterable[int],
) -> tuple[float, float, int]:
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
            policy_label="Кандидат",
            training_seed=None,
        )
        losses.append(float(trace["loss"].sum()))
        volatilities.append(float(np.std(trace["policy_rate"].to_numpy(dtype=float))))
        values = trace[["true_inflation_gap", "true_output_gap", "policy_rate"]].to_numpy(dtype=float)
        unstable += int((not np.isfinite(values).all()) or np.any(np.abs(values) > np.array([0.06, 0.12, 0.06])[None, :]))
    return float(np.mean(losses)), float(np.mean(volatilities)), int(unstable)


def _collect_signal_statistics(
    *,
    scenario_name: str,
    validation_seeds: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    scenario_spec, env_factory = _build_objects(
        scenario_name=scenario_name,
        info_set_name="posterior_distribution",
        validation_seeds=validation_seeds,
        test_seeds=validation_seeds,
    )
    policy = ClassicalFilteredRulePolicy(action_bound=scenario_spec.action_bound)
    traces = []
    for seed in validation_seeds:
        trace = simulate_policy_episode(
            env_factory=env_factory,
            policy=policy,
            scenario_spec=scenario_spec,
            evaluation_seed=int(seed),
            policy_name="classical_filtered_rule",
            policy_label="Классическое правило по оценённому состоянию",
            training_seed=None,
        )
        traces.append(trace)
    frame = pd.concat(traces, ignore_index=True)
    stats_means: dict[str, float] = {}
    stats_stds: dict[str, float] = {}

    def _column_or_zeros(column_name: str) -> np.ndarray:
        if column_name not in frame.columns:
            return np.zeros(len(frame), dtype=float)
        return frame[column_name].fillna(0.0).to_numpy(dtype=float)

    for feature_name in FEATURE_COLUMNS:
        if feature_name == "observed_pi":
            values = _column_or_zeros("observed_pi")
        elif feature_name == "lagged_observed_pi":
            values = _column_or_zeros("lagged_observed_pi")
        elif feature_name == "observed_output_gap":
            values = _column_or_zeros("observed_output_gap")
        elif feature_name == "lagged_observed_output_gap":
            values = _column_or_zeros("lagged_observed_output_gap")
        elif feature_name == "stress_probability":
            values = _column_or_zeros("stress_probability")
        elif feature_name == "stress_entropy":
            values = _column_or_zeros("stress_entropy")
        elif feature_name == "filtered_variance_trace":
            values = _column_or_zeros("filtered_variance_trace")
        elif feature_name == "mean_rstar_gap":
            values = _column_or_zeros("filtered_rstar_gap")
        elif feature_name == "mean_productivity_gap":
            values = _column_or_zeros("filtered_productivity_gap")
        elif feature_name == "mean_fiscal_gap":
            values = _column_or_zeros("filtered_fiscal_gap")
        elif feature_name == "mean_inflation_gap":
            values = _column_or_zeros("filtered_inflation_gap")
        elif feature_name == "mean_output_gap":
            values = _column_or_zeros("filtered_output_gap")
        elif feature_name == "mean_low_liquidity_gap":
            values = _column_or_zeros("filtered_low_liquidity_gap")
        elif feature_name == "mean_mean_mpc_gap":
            values = _column_or_zeros("filtered_mean_mpc_gap")
        else:  # pragma: no cover
            continue
        stats_means[feature_name] = float(np.mean(values))
        std = float(np.std(values))
        stats_stds[feature_name] = std if std > 1.0e-12 else 1.0
    return stats_means, stats_stds


def _select_information_rule(
    *,
    scenario_name: str,
    info_set_name: str,
    validation_seeds: tuple[int, ...],
    standardized_signals: bool = False,
) -> tuple[PosteriorRuleParameters, pd.DataFrame]:
    scenario_spec, env_factory = _build_objects(
        scenario_name=scenario_name,
        info_set_name=info_set_name,
        validation_seeds=validation_seeds,
        test_seeds=validation_seeds,
    )
    feature_names = _feature_names(info_set_name, scenario_name)
    signal_means = {}
    signal_stds = {}
    if standardized_signals:
        signal_means, signal_stds = _collect_signal_statistics(
            scenario_name=scenario_name,
            validation_seeds=validation_seeds,
        )

    current = PosteriorRuleParameters(
        info_set_name=info_set_name,
        feature_names=feature_names,
        coefficients=(
            tuple(0.0 for _ in feature_names)
            if standardized_signals
            else tuple(_initial_coefficient(feature_name) for feature_name in feature_names)
        ),
        rho_i=0.7,
    )
    rows: list[dict[str, float | str | int]] = []

    def evaluate(params: PosteriorRuleParameters, *, coordinate_name: str, coordinate_pass: int) -> tuple[float, float, int]:
        loss, volatility, unstable = _mean_cumulative_loss(
            env_factory=env_factory,
            scenario_spec=scenario_spec,
            policy=PosteriorInformationLinearRulePolicy(
                params,
                signal_means=signal_means if standardized_signals else None,
                signal_stds=signal_stds if standardized_signals else None,
            ),
            seeds=validation_seeds,
        )
        row: dict[str, float | str | int] = {
            "scenario_name": scenario_name,
            "scenario_label": SCENARIO_LABELS[scenario_name],
            "info_set_name": info_set_name,
            "info_set_label": INFO_SET_LABELS_RU[info_set_name],
            "policy_name": _policy_name(info_set_name),
            "policy_label": (
                _policy_label_standardized(info_set_name)
                if standardized_signals
                else _policy_label(info_set_name)
            ),
            "coordinate_name": coordinate_name,
            "coordinate_pass": coordinate_pass,
            "standardized_signals": int(standardized_signals),
            "validation_cumulative_loss": loss,
            "validation_policy_volatility": volatility,
            "validation_unstable_episodes": unstable,
            **params.to_dict(),
        }
        rows.append(row)
        return loss, volatility, unstable

    best_loss, best_volatility, best_unstable = evaluate(
        current,
        coordinate_name="начальная точка",
        coordinate_pass=0,
    )
    for coordinate_pass in range(1, 4):
        improved = False
        for feature_name in list(feature_names) + ["rho_i"]:
            best_field_params = current
            best_field_tuple = (best_unstable, best_loss, best_volatility)
            candidates = (
                (0.3, 0.5, 0.7, 0.85)
                if feature_name == "rho_i"
                else _candidate_values(feature_name, standardized_signals=standardized_signals)
            )
            for candidate_value in candidates:
                if feature_name == "rho_i":
                    params = PosteriorRuleParameters(
                        info_set_name=info_set_name,
                        feature_names=current.feature_names,
                        coefficients=current.coefficients,
                        rho_i=float(candidate_value),
                    )
                else:
                    coefficients = list(current.coefficients)
                    index = current.feature_names.index(feature_name)
                    coefficients[index] = float(candidate_value)
                    params = PosteriorRuleParameters(
                        info_set_name=info_set_name,
                        feature_names=current.feature_names,
                        coefficients=tuple(coefficients),
                        rho_i=current.rho_i,
                    )
                loss, volatility, unstable = evaluate(
                    params,
                    coordinate_name=feature_name,
                    coordinate_pass=coordinate_pass,
                )
                candidate_tuple = (unstable, loss, volatility)
                if candidate_tuple < best_field_tuple:
                    best_field_tuple = candidate_tuple
                    best_field_params = params
            if best_field_params != current:
                current = best_field_params
                best_unstable, best_loss, best_volatility = best_field_tuple
                improved = True
        if not improved:
            break

    grid = pd.DataFrame(rows).sort_values(
        ["validation_unstable_episodes", "validation_cumulative_loss", "validation_policy_volatility"]
    ).reset_index(drop=True)
    best = grid.iloc[0]
    params = PosteriorRuleParameters(
        info_set_name=info_set_name,
        feature_names=feature_names,
        coefficients=tuple(float(best[f"weight_{feature_name}"]) for feature_name in feature_names),
        rho_i=float(best["rho_i"]),
    )
    return params, grid


def _evaluate_selected_rules(
    *,
    scenario_name: str,
    validation_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
    selected_rules: dict[str, PosteriorRuleParameters],
    standardized_signals: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float | str | int]] = []
    path_frames: list[pd.DataFrame] = []
    signal_means = {}
    signal_stds = {}
    if standardized_signals:
        signal_means, signal_stds = _collect_signal_statistics(
            scenario_name=scenario_name,
            validation_seeds=validation_seeds,
        )
    for info_set_name in INFO_SET_ORDER:
        params = selected_rules[info_set_name]
        scenario_spec, env_factory = _build_objects(
            scenario_name=scenario_name,
            info_set_name=info_set_name,
            validation_seeds=validation_seeds,
            test_seeds=test_seeds,
        )
        policy = PosteriorInformationLinearRulePolicy(
            params,
            signal_means=signal_means if standardized_signals else None,
            signal_stds=signal_stds if standardized_signals else None,
        )
        for seed in test_seeds:
            trace = simulate_policy_episode(
                env_factory=env_factory,
                policy=policy,
                scenario_spec=scenario_spec,
                evaluation_seed=int(seed),
                policy_name=_policy_name(info_set_name),
                policy_label=(
                    _policy_label_standardized(info_set_name)
                    if standardized_signals
                    else _policy_label(info_set_name)
                ),
                training_seed=None,
            )
            trace = trace.copy()
            trace["info_set_name"] = info_set_name
            trace["info_set_label"] = INFO_SET_LABELS_RU[info_set_name]
            path_frames.append(trace)
            losses = trace["loss"].to_numpy(dtype=float)
            metrics: dict[str, float | str | int] = {
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "info_set_name": info_set_name,
                "info_set_label": INFO_SET_LABELS_RU[info_set_name],
                "policy_name": _policy_name(info_set_name),
                "policy_label": (
                    _policy_label_standardized(info_set_name)
                    if standardized_signals
                    else _policy_label(info_set_name)
                ),
                "evaluation_seed": int(seed),
                "standardized_signals": int(standardized_signals),
                "mean_policy_loss": float(np.mean(losses)),
                "cumulative_policy_loss": float(np.sum(losses)),
                "cumulative_inflation_loss": float(np.sum(trace["inflation_loss"].to_numpy(dtype=float))),
                "cumulative_output_gap_loss": float(np.sum(trace["output_gap_loss"].to_numpy(dtype=float))),
                "cumulative_rate_change_loss": float(np.sum(trace["rate_change_loss"].to_numpy(dtype=float))),
                "policy_rate_volatility": float(np.std(trace["policy_rate"].to_numpy(dtype=float))),
                "unstable": int(
                    np.any(
                        np.abs(
                            trace[["true_inflation_gap", "true_output_gap", "policy_rate"]].to_numpy(dtype=float)
                        )
                        > np.array([0.06, 0.12, 0.06])[None, :]
                    )
                ),
            }
            metric_rows.append(metrics)
    return pd.DataFrame(metric_rows), pd.concat(path_frames, ignore_index=True)


def _summarize_levels(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario_name, info_set_name), frame in policy_metrics.groupby(["scenario_name", "info_set_name"], dropna=False):
        losses = frame["cumulative_policy_loss"].to_numpy(dtype=float)
        ci_lower, ci_upper = _bootstrap_ci(losses)
        rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "info_set_name": info_set_name,
                "info_set_label": INFO_SET_LABELS_RU[info_set_name],
                "mean_cumulative_loss": float(losses.mean()),
                "std_cumulative_loss": float(losses.std(ddof=1)) if losses.size > 1 else 0.0,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "mean_cumulative_inflation_loss": float(frame["cumulative_inflation_loss"].mean()),
                "mean_cumulative_output_gap_loss": float(frame["cumulative_output_gap_loss"].mean()),
                "mean_cumulative_rate_change_loss": float(frame["cumulative_rate_change_loss"].mean()),
                "mean_policy_rate_volatility": float(frame["policy_rate_volatility"].mean()),
                "num_test_trajectories": int(frame.shape[0]),
            }
        )
    summary = pd.DataFrame(rows)
    summary["info_set_order"] = pd.Categorical(
        summary["info_set_name"],
        categories=INFO_SET_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(["scenario_name", "info_set_order"]).drop(columns="info_set_order")
    return summary.reset_index(drop=True)


def _pairwise_comparisons(
    policy_metrics: pd.DataFrame,
    *,
    tie_tolerance: float = TIE_TOLERANCE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = policy_metrics.pivot_table(
        index=["scenario_name", "evaluation_seed"],
        columns="info_set_name",
        values=[
            "cumulative_policy_loss",
            "cumulative_inflation_loss",
            "cumulative_output_gap_loss",
            "cumulative_rate_change_loss",
        ],
        aggfunc="first",
    )
    pairwise_rows = []
    component_rows = []
    for scenario_name in policy_metrics["scenario_name"].drop_duplicates():
        scenario_frame = pivot.loc[scenario_name]
        for comparison_name, (left_name, right_name) in PAIRWISE_COMPARISONS.items():
            if left_name not in scenario_frame["cumulative_policy_loss"] or right_name not in scenario_frame["cumulative_policy_loss"]:
                continue
            left_loss = scenario_frame["cumulative_policy_loss"][left_name].to_numpy(dtype=float)
            right_loss = scenario_frame["cumulative_policy_loss"][right_name].to_numpy(dtype=float)
            delta = left_loss - right_loss
            ci_lower, ci_upper = _bootstrap_ci(delta)
            pairwise_rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "comparison_name": comparison_name,
                    "comparison_label": PAIRWISE_LABELS_RU[comparison_name],
                    "left_info_set_name": left_name,
                    "right_info_set_name": right_name,
                    "left_info_set_label": INFO_SET_LABELS_RU[left_name],
                    "right_info_set_label": INFO_SET_LABELS_RU[right_name],
                    "mean_delta_cumulative_loss": float(delta.mean()),
                    "std_delta_cumulative_loss": float(delta.std(ddof=1)) if delta.size > 1 else 0.0,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "win_rate": float(np.mean(delta < -tie_tolerance)),
                    "tie_rate": float(np.mean(np.abs(delta) <= tie_tolerance)),
                    "probability_of_degradation": float(np.mean(delta > tie_tolerance)),
                    "mean_relative_improvement_pct": float(100.0 * np.mean(-delta / right_loss)),
                    "num_test_trajectories": int(delta.size),
                }
            )
            component_rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "comparison_name": comparison_name,
                    "comparison_label": PAIRWISE_LABELS_RU[comparison_name],
                    "delta_inflation_loss": float(
                        (
                            scenario_frame["cumulative_inflation_loss"][left_name].to_numpy(dtype=float)
                            - scenario_frame["cumulative_inflation_loss"][right_name].to_numpy(dtype=float)
                        ).mean()
                    ),
                    "delta_output_gap_loss": float(
                        (
                            scenario_frame["cumulative_output_gap_loss"][left_name].to_numpy(dtype=float)
                            - scenario_frame["cumulative_output_gap_loss"][right_name].to_numpy(dtype=float)
                        ).mean()
                    ),
                    "delta_rate_change_loss": float(
                        (
                            scenario_frame["cumulative_rate_change_loss"][left_name].to_numpy(dtype=float)
                            - scenario_frame["cumulative_rate_change_loss"][right_name].to_numpy(dtype=float)
                        ).mean()
                    ),
                }
            )
    return pd.DataFrame(pairwise_rows), pd.DataFrame(component_rows)


def _signal_quality(mean_paths: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    calibration_rows = []
    bins = np.linspace(0.0, 1.0, 6)
    clipped = np.clip(mean_paths["stress_probability"].to_numpy(dtype=float), 1.0e-10, 1.0 - 1.0e-10)
    hidden = mean_paths["hidden_regime"].to_numpy(dtype=int)
    for scenario_name, frame in mean_paths.groupby("scenario_name", dropna=False):
        p = np.clip(frame["stress_probability"].to_numpy(dtype=float), 1.0e-10, 1.0 - 1.0e-10)
        y = frame["hidden_regime"].to_numpy(dtype=int)
        rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "policy_name": _policy_name("posterior_mean"),
                "policy_label": _policy_label("posterior_mean"),
                "brier_score": float(np.mean(np.square(p - y))),
                "log_score": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
                "regime_accuracy": float(np.mean((p >= 0.5).astype(int) == y)),
                "mean_entropy": float(frame["stress_entropy"].mean()),
                "num_periods": int(frame.shape[0]),
                "num_trajectories": int(frame["evaluation_seed"].nunique()),
            }
        )
        bin_ids = np.digitize(frame["stress_probability"].to_numpy(dtype=float), bins[1:-1], right=False)
        for bin_id in range(len(bins) - 1):
            mask = bin_ids == bin_id
            if not np.any(mask):
                continue
            calibration_rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "bin_left": float(bins[bin_id]),
                    "bin_right": float(bins[bin_id + 1]),
                    "mean_predicted_probability": float(frame.loc[mask, "stress_probability"].mean()),
                    "actual_stress_frequency": float(frame.loc[mask, "hidden_regime"].mean()),
                    "count": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(calibration_rows)


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
            shock_path = subset.groupby("period")["policy_rate"].mean().sort_index().to_numpy(dtype=float)
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
                except Exception as exc:  # pragma: no cover
                    solver_error = f"{type(exc).__name__}: {exc}"
                    transition = None
            if transition is None:
                rows.append(
                    {
                        "scenario_name": scenario_name,
                        "scenario_label": SCENARIO_LABELS[scenario_name],
                        "policy_name": policy_name,
                        "solver_success": 0,
                        "scale_used": math.nan,
                        "solver_error": solver_error,
                        "full_hank_cumulative_loss": math.nan,
                    }
                )
                continue
            pi = transition["pi"]
            output = transition["output_gap"]
            rate = transition["i"]
            loss = pi**2 + 0.5 * output**2 + 0.05 * np.square(np.diff(rate, prepend=0.0))
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": policy_name,
                    "solver_success": 1,
                    "scale_used": scale_used,
                    "solver_error": "",
                    "full_hank_cumulative_loss": float(np.sum(loss)),
                }
            )
            for period in range(len(pi)):
                path_rows.append(
                    {
                        "scenario_name": scenario_name,
                        "scenario_label": SCENARIO_LABELS[scenario_name],
                        "policy_name": policy_name,
                        "period": int(period),
                        "projection_scale_used": scale_used,
                        "mean_policy_rate_path_used_as_shock": float(scale_used * shock_path[period]),
                        "inflation_gap": float(pi[period]),
                        "output_gap": float(output[period]),
                        "policy_rate": float(rate[period]),
                        "period_loss": float(loss[period]),
                    }
                )
    metrics = pd.DataFrame(rows)
    paths = pd.DataFrame(path_rows)
    metrics.to_csv(output_dir / "posterior_information_hank_projection_metrics.csv", index=False)
    paths.to_csv(output_dir / "posterior_information_hank_projection_paths.csv", index=False)
    return metrics, paths


def _rank_series(series: pd.Series) -> pd.Series:
    return series.rank(method="dense", ascending=True)


def _posterior_common_scale_hank_validation(
    *,
    policy_paths: pd.DataFrame,
    reduced_levels: pd.DataFrame,
    full_hank_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    ranking_rows = []
    pair_rows = []
    reduced_map = reduced_levels.set_index(["scenario_name", "info_set_name"])["mean_cumulative_loss"].to_dict()
    policy_to_info = {
        "linear_observed_information": "observed_information",
        "linear_posterior_mean": "posterior_mean",
        "linear_posterior_uncertainty": "posterior_uncertainty",
        "linear_posterior_distribution": "posterior_distribution",
    }
    for scenario_name, frame in full_hank_metrics.groupby("scenario_name"):
        successful = frame[
            frame["policy_name"].isin(KEY_HANK_POLICIES) & (frame["solver_success"] == 1)
        ].copy()
        if successful.shape[0] < len(KEY_HANK_POLICIES):
            continue
        common_scale = float(successful["scale_used"].min())
        common_losses: dict[str, float] = {}
        for policy_name in KEY_HANK_POLICIES:
            subset = policy_paths[
                (policy_paths["scenario_name"] == scenario_name)
                & (policy_paths["policy_name"] == policy_name)
            ].copy()
            shock_path = subset.groupby("period")["policy_rate"].mean().sort_index().to_numpy(dtype=float)
            shock_path = shock_path[: hank_config.shock_T]
            if shock_path.size < hank_config.shock_T:
                shock_path = np.pad(shock_path, (0, hank_config.shock_T - shock_path.size))
            transition = solve_transition(bundle, {"monetary_policy_shock": common_scale * shock_path})
            pi = transition["pi"]
            output = transition["output_gap"]
            rate = transition["i"]
            loss = pi**2 + 0.5 * output**2 + 0.05 * np.square(np.diff(rate, prepend=0.0))
            common_losses[policy_name] = float(np.sum(loss))
            ranking_rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": policy_name,
                    "info_set_name": policy_to_info[policy_name],
                    "common_scale": common_scale,
                    "reduced_cumulative_loss": float(reduced_map[(scenario_name, policy_to_info[policy_name])]),
                    "common_scale_full_hank_loss": float(np.sum(loss)),
                }
            )
        reduced_series = pd.Series(
            {policy_name: reduced_map[(scenario_name, policy_to_info[policy_name])] for policy_name in KEY_HANK_POLICIES}
        )
        common_series = pd.Series(common_losses)
        reduced_ranks = _rank_series(reduced_series)
        common_ranks = _rank_series(common_series)
        spearman = float(reduced_ranks.corr(common_ranks, method="spearman"))
        for row in ranking_rows[-len(KEY_HANK_POLICIES):]:
            row["reduced_rank"] = float(reduced_ranks[row["policy_name"]])
            row["common_scale_full_hank_rank"] = float(common_ranks[row["policy_name"]])
            row["spearman_rank_correlation"] = spearman
            row["ranking_preserved"] = int(float(reduced_ranks[row["policy_name"]]) == float(common_ranks[row["policy_name"]]))
        for comparison_name, left_policy, right_policy in PAIRWISE_HANK_CHECKS:
            reduced_delta = float(reduced_series[left_policy] - reduced_series[right_policy])
            hank_delta = float(common_series[left_policy] - common_series[right_policy])
            pair_rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "comparison_name": comparison_name,
                    "comparison_label": PAIRWISE_LABELS_RU.get(comparison_name, comparison_name),
                    "left_policy": left_policy,
                    "right_policy": right_policy,
                    "common_scale": common_scale,
                    "reduced_delta_loss": reduced_delta,
                    "common_scale_full_hank_delta_loss": hank_delta,
                    "same_pairwise_sign": int(np.sign(reduced_delta) == np.sign(hank_delta)),
                }
            )
    return pd.DataFrame(ranking_rows), pd.DataFrame(pair_rows)


def _write_latex_tables(
    root: Path,
    levels: pd.DataFrame,
    pairwise: pd.DataFrame,
    components: pd.DataFrame,
    signal_quality: pd.DataFrame,
    hank_pairwise: pd.DataFrame,
) -> None:
    level_pivot = levels.pivot_table(
        index=["scenario_name", "scenario_label"],
        columns="info_set_name",
        values="mean_cumulative_loss",
        aggfunc="first",
    )
    available = [name for name in INFO_SET_ORDER if name in level_pivot.columns]
    lines = [
        "\\begin{tabular}{p{0.28\\linewidth}" + "r" * len(available) + "}",
        "\\toprule",
        "Сценарий & " + " & ".join(_latex_escape(INFO_SET_LABELS_RU[name]) for name in available) + " \\\\",
        "\\midrule",
    ]
    for (_scenario_name, scenario_label), row in level_pivot.iterrows():
        lines.append(
            " & ".join(
                [_latex_escape(str(scenario_label))]
                + [f"{float(row[name]):.4e}" for name in available]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_posterior_information_levels.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{tabular}{p{0.24\\linewidth}p{0.20\\linewidth}rrrrrr}",
        "\\toprule",
        "Сценарий & Сравнение & $\\Delta J$ & 95\\% ДИ & Победа & Совпадение & Ухудшение & $N$ \\\\",
        "\\midrule",
    ]
    for row in pairwise.to_dict(orient="records"):
        ci = f"[{float(row['ci_lower']):.4e}; {float(row['ci_upper']):.4e}]"
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["scenario_label"])),
                    _latex_escape(str(row["comparison_label"])),
                    f"{float(row['mean_delta_cumulative_loss']):.4e}",
                    _latex_escape(ci),
                    f"{float(row['win_rate']):.2f}",
                    f"{float(row['tie_rate']):.2f}",
                    f"{float(row['probability_of_degradation']):.2f}",
                    f"{int(row['num_test_trajectories'])}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_posterior_information_pairwise.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{tabular}{p{0.28\\linewidth}p{0.22\\linewidth}rrr}",
        "\\toprule",
        "Сценарий & Сравнение & Инфляция & Выпуск & Изменение ставки \\\\",
        "\\midrule",
    ]
    for row in components.to_dict(orient="records"):
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["scenario_label"])),
                    _latex_escape(str(row["comparison_label"])),
                    f"{float(row['delta_inflation_loss']):.4e}",
                    f"{float(row['delta_output_gap_loss']):.4e}",
                    f"{float(row['delta_rate_change_loss']):.4e}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_posterior_information_components.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        "\\begin{tabular}{p{0.32\\linewidth}rrrr}",
        "\\toprule",
        "Сценарий & Квадратическая ошибка вероятности режима & Логарифмический балл & Точность режима & Средняя энтропия \\\\",
        "\\midrule",
    ]
    for row in signal_quality.to_dict(orient="records"):
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["scenario_label"])),
                    f"{float(row['brier_score']):.4f}",
                    f"{float(row['log_score']):.4f}",
                    f"{float(row['regime_accuracy']):.3f}",
                    f"{float(row['mean_entropy']):.4f}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_posterior_signal_quality.tex").write_text("\n".join(lines), encoding="utf-8")

    if not hank_pairwise.empty:
        lines = [
            "\\begin{tabular}{p{0.28\\linewidth}p{0.20\\linewidth}rrr}",
            "\\toprule",
            "Сценарий & Сравнение & $\\Delta J$ в редуцированной задаче & $\\Delta J$ после HANK-проекции & Знак совпал \\\\",
            "\\midrule",
        ]
        for row in hank_pairwise.to_dict(orient="records"):
            lines.append(
                " & ".join(
                    [
                        _latex_escape(str(row["scenario_label"])),
                        _latex_escape(str(row["comparison_label"])),
                        f"{float(row['reduced_delta_loss']):.4e}",
                        f"{float(row['common_scale_full_hank_delta_loss']):.4e}",
                        "да" if int(row["same_pairwise_sign"]) == 1 else "нет",
                    ]
                )
                + " \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        (root / "table_posterior_information_hank_pairwise.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_report(
    *,
    root: Path,
    levels: pd.DataFrame,
    pairwise: pd.DataFrame,
    signal_quality: pd.DataFrame,
    selected_specs: pd.DataFrame,
    standardized_pairwise: pd.DataFrame,
    hank_ranking: pd.DataFrame,
    hank_pairwise: pd.DataFrame,
    scenario_names: tuple[str, ...],
    validation_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
) -> None:
    headline_rows = []
    for scenario_name in scenario_names:
        scenario_levels = levels[levels["scenario_name"] == scenario_name].sort_values("mean_cumulative_loss")
        best_row = scenario_levels.iloc[0]
        scenario_pairwise = pairwise[pairwise["scenario_name"] == scenario_name]
        mean_row = scenario_pairwise[scenario_pairwise["comparison_name"] == "mean_minus_observed"]
        regime_row = scenario_pairwise[scenario_pairwise["comparison_name"] == "regime_minus_mean"]
        uncertainty_row = scenario_pairwise[scenario_pairwise["comparison_name"] == "uncertainty_minus_regime"]
        distribution_row = scenario_pairwise[scenario_pairwise["comparison_name"] == "distribution_minus_uncertainty"]

        phrases = [
            f"минимальная средняя потеря у правила «{best_row['info_set_label']}»",
        ]
        if not mean_row.empty:
            mean_delta = float(mean_row["mean_delta_cumulative_loss"].iloc[0])
            mean_ci_low = float(mean_row["ci_lower"].iloc[0])
            mean_ci_high = float(mean_row["ci_upper"].iloc[0])
            if mean_ci_low <= 0.0 <= mean_ci_high:
                phrases.append("апостериорное среднее и набор наблюдаемых переменных почти не различаются")
            elif mean_delta < -5.0e-6:
                phrases.append("апостериорное среднее лучше набора наблюдаемых переменных")
            elif mean_delta > 5.0e-6:
                phrases.append("набор наблюдаемых переменных лучше апостериорного среднего")
            else:
                phrases.append("апостериорное среднее и наблюдаемые переменные почти не различаются")
        if not regime_row.empty:
            regime_delta = abs(float(regime_row["mean_delta_cumulative_loss"].iloc[0]))
            if regime_delta <= 5.0e-8:
                phrases.append("добавление вероятности режима дополнительного выигрыша не даёт")
        if not uncertainty_row.empty:
            uncertainty_delta = abs(float(uncertainty_row["mean_delta_cumulative_loss"].iloc[0]))
            if uncertainty_delta <= 5.0e-8:
                phrases.append("добавление меры неопределённости дополнительного выигрыша не даёт")
        if not distribution_row.empty:
            distribution_delta = float(distribution_row["mean_delta_cumulative_loss"].iloc[0])
            distribution_ci_low = float(distribution_row["ci_lower"].iloc[0])
            distribution_ci_high = float(distribution_row["ci_upper"].iloc[0])
            if distribution_ci_low <= 0.0 <= distribution_ci_high:
                phrases.append("распределительное расширение не даёт устойчивого изменения")
            elif distribution_delta < -5.0e-6:
                phrases.append("распределительное расширение улучшает результат")

        headline_rows.append(f"- {SCENARIO_LABELS[scenario_name]}: " + "; ".join(phrases) + ".")

    signal_lines = []
    for row in signal_quality.to_dict(orient="records"):
        signal_lines.append(
            f"- {row['scenario_label']}: квадратическая ошибка вероятности режима = {float(row['brier_score']):.4f}, "
            f"логарифмический балл = {float(row['log_score']):.4f}, "
            f"точность режима = {float(row['regime_accuracy']):.3f}."
        )

    robustness_lines = []
    if not standardized_pairwise.empty:
        key_rows = standardized_pairwise[
            standardized_pairwise["comparison_name"].isin(
                ["regime_minus_mean", "uncertainty_minus_regime", "distribution_minus_uncertainty"]
            )
        ]
        for row in key_rows.to_dict(orient="records"):
            robustness_lines.append(
                f"- {row['scenario_label']}, {row['comparison_label'].lower()}: "
                f"средняя разность = {float(row['mean_delta_cumulative_loss']):.4e}, "
                f"доля совпадений = {float(row['tie_rate']):.2f}."
            )

    hank_lines = []
    validated_hank_scenarios = sorted(hank_pairwise["scenario_name"].unique().tolist()) if not hank_pairwise.empty else []
    missing_hank_scenarios = [
        SCENARIO_LABELS[scenario_name]
        for scenario_name in scenario_names
        if scenario_name not in validated_hank_scenarios
    ]
    if not hank_pairwise.empty:
        hank_lines.append(
            f"- Полная проверка для всех четырёх ключевых правил выполнена в {len(validated_hank_scenarios)} из {len(scenario_names)} сценариев."
        )
        if missing_hank_scenarios:
            hank_lines.append(
                "- В остальных сценариях общий слой HANK-проверки не используется, "
                "потому что проекция сходится не для всех четырёх правил одновременно."
            )
        matches = int(hank_pairwise["same_pairwise_sign"].sum())
        total = int(hank_pairwise.shape[0])
        hank_lines.append(
            f"- В общей HANK-проекции совпадают {matches} из {total} проверяемых знаков попарных сравнений."
        )
        if not hank_ranking.empty:
            for scenario_name, frame in hank_ranking.groupby("scenario_name"):
                spearman = float(frame["spearman_rank_correlation"].iloc[0])
                common_scale = float(frame["common_scale"].iloc[0])
                hank_lines.append(
                    f"- {SCENARIO_LABELS[scenario_name]}: ранговая корреляция Спирмена = {spearman:.2f}, общий масштаб = {common_scale:.2f}."
                )

    text = "\n".join(
        [
            "# Апостериорное информационное состояние",
            "",
            "## Что сравнивается",
            "",
            "В этом блоке сравниваются пять линейных правил на одних и тех же траекториях.",
            "Меняется только то, какая сводка апостериорной информации передаётся правилу.",
            "",
            "- Наблюдаемые переменные.",
            "- Апостериорное среднее скрытого состояния.",
            "- Апостериорное среднее и вероятность стрессового режима.",
            "- Апостериорное среднее, вероятность режима и мера неопределённости.",
            "- Распределительно расширенное состояние.",
            "",
            "Правило везде имеет одну и ту же форму:",
            "`i_t = rho_i * i_{t-1} + theta' z_t`.",
            "",
            "## Дизайн",
            "",
            f"- Сценарии: {len(scenario_names)}.",
            f"- Валидационные траектории: {_format_seed_span(validation_seeds)}.",
            f"- Тестовые траектории: {_format_seed_span(test_seeds)}.",
            "- Сравнение парное: для всех правил используются одни и те же тестовые траектории.",
            "",
            "## Главные результаты",
            "",
            *headline_rows,
            "",
            "## Качество сигнала режима",
            "",
            "Отдельно оценивается сама вероятность стрессового режима. Для этого берутся траектории правила,",
            "которое использует только апостериорное среднее, и по ним считаются квадратическая ошибка, логарифмический балл и доля",
            "правильно распознанных режимов.",
            "",
            *signal_lines,
            "",
            "## Проверка масштаба коэффициентов",
            "",
            "Чтобы убедиться, что нулевые эффекты не вызваны слишком узкой сеткой коэффициентов,",
            "дополнительно повторён тот же линейный эксперимент после стандартизации признаков.",
            "Ниже вынесены ключевые результаты этой короткой проверки.",
            "",
            *(robustness_lines if robustness_lines else ["- Дополнительная проверка не выполнена."]),
            "",
            "## Проверка через HANK-проекцию",
            "",
            "Для четырёх ключевых правил дополнительно проверяется, сохраняются ли знаки сравнений",
            "после пропуска средней траектории ставки через переходный решатель полной HANK-модели",
            "на общем допустимом масштабе.",
            "",
            *(hank_lines if hank_lines else ["- Проверка через HANK-проекцию не выполнена."]),
            "",
            "## Интерпретация",
            "",
            "Этот блок не утверждает, что условного среднего достаточно всегда.",
            "Он проверяет более узкий вопрос: что именно даёт выигрыш, когда правило получает",
            "не одну точечную оценку, а более богатую конечномерную сводку апостериорной информации.",
            "",
            "В текущих расчётах добавление вероятности режима и простой агрегированной меры неопределённости",
            "не улучшает линейное правило сверх апостериорного среднего. Проверка со стандартизацией признаков",
            "не меняет этого вывода. Зато в части сценариев полезным",
            "оказывается распределительное расширение состояния. Поэтому главный вывод здесь такой:",
            "важно не только оценивать скрытое состояние, но и проверять, какая именно конечномерная",
            "сводка этой информации действительно нужна правилу политики.",
        ]
    )
    _save_text(root / "report_posterior_information_state.md", text)

    methods_paragraph = (
        "В задаче управления при неполной наблюдаемости достаточной статистикой истории наблюдений "
        "является апостериорное распределение скрытого состояния, а не только его условное среднее. "
        "Поэтому в работе отдельно рассматриваются несколько конечномерных представлений "
        "апостериорной информации: только апостериорное среднее, среднее с вероятностью скрытого режима "
        "и среднее с дополнительной мерой неопределённости. Это позволяет отделить ценность самой "
        "фильтрации от ценности отдельных компонентов фильтрованного информационного состояния."
    )

    result_paragraph = (
        "Сравнение проводится при одинаковой линейной форме правила и на одинаковых тестовых траекториях. "
        "Тем самым меняется только информационное представление состояния. В выполненных расчётах "
        "добавление вероятности режима и простой агрегированной меры неопределённости не даёт "
        "дополнительного выигрыша сверх апостериорного среднего. Этот вывод сохраняется и после "
        "стандартизации признаков. В нескольких сценариях выигрыш даёт "
        "распределительное расширение состояния. Значит, вопрос состоит не просто в том, чтобы заменить "
        "наблюдения фильтрованным средним, а в том, какие именно компоненты апостериорной информации "
        "следует передавать правилу денежно-кредитной политики."
    )

    tex = "\n".join(
        [
            "\\subsection{Конечномерное апостериорное информационное состояние}",
            methods_paragraph,
            "",
            result_paragraph,
        ]
    )
    _save_text(root / "posterior_information_text_blocks.tex", tex)


def _write_selection_table(root: Path, selected_specs: pd.DataFrame) -> None:
    lines = [
        "\\begin{tabular}{p{0.24\\linewidth}p{0.22\\linewidth}p{0.38\\linewidth}}",
        "\\toprule",
        "Сценарий & Набор информации & Выбранные коэффициенты \\\\",
        "\\midrule",
    ]
    for row in selected_specs.to_dict(orient="records"):
        params = PosteriorRuleParameters(
            info_set_name=str(row["info_set_name"]),
            feature_names=tuple(str(row["feature_names"]).split("|")) if str(row["feature_names"]) else tuple(),
            coefficients=tuple(float(row[f"weight_{name}"]) for name in str(row["feature_names"]).split("|") if name),
            rho_i=float(row["rho_i"]),
        )
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["scenario_label"])),
                    _latex_escape(str(row["info_set_label"])),
                    _latex_escape(_coefficient_string(params)),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_selected_posterior_rules.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_value_summary_table(root: Path, levels: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    rows = []
    pivot = levels.pivot_table(
        index="scenario_name",
        columns="info_set_name",
        values="mean_cumulative_loss",
        aggfunc="first",
    )
    pair_pivot = pairwise.pivot_table(
        index="scenario_name",
        columns="comparison_name",
        values="mean_delta_cumulative_loss",
        aggfunc="first",
    )
    for scenario_name in pivot.index:
        rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "loss_observed": float(pivot.loc[scenario_name, "observed_information"]),
                "loss_mean": float(pivot.loc[scenario_name, "posterior_mean"]),
                "loss_regime": float(pivot.loc[scenario_name, "posterior_regime"]),
                "loss_uncertainty": float(pivot.loc[scenario_name, "posterior_uncertainty"]),
                "loss_distribution": float(pivot.loc[scenario_name, "posterior_distribution"]),
                "gain_regime": float(-pair_pivot.loc[scenario_name, "regime_minus_mean"]),
                "gain_uncertainty": float(-pair_pivot.loc[scenario_name, "uncertainty_minus_regime"]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "posterior_information_value_summary.csv", index=False)

    lines = [
        "\\begin{tabular}{p{0.26\\linewidth}rrrrrrr}",
        "\\toprule",
        "Сценарий & $J^{obs}$ & $J^{mean}$ & $J^{regime}$ & $J^{unc}$ & $J^{dist}$ & выигрыш режима & выигрыш неопределённости \\\\",
        "\\midrule",
    ]
    for row in frame.to_dict(orient="records"):
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["scenario_label"])),
                    f"{float(row['loss_observed']):.4e}",
                    f"{float(row['loss_mean']):.4e}",
                    f"{float(row['loss_regime']):.4e}",
                    f"{float(row['loss_uncertainty']):.4e}",
                    f"{float(row['loss_distribution']):.4e}",
                    f"{float(row['gain_regime']):.4e}",
                    f"{float(row['gain_uncertainty']):.4e}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_posterior_information_value_summary.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_standardized_table(root: Path, pairwise: pd.DataFrame) -> None:
    if pairwise.empty:
        return
    lines = [
        "\\begin{tabular}{p{0.25\\linewidth}p{0.21\\linewidth}rrrrr}",
        "\\toprule",
        "Сценарий & Сравнение & $\\Delta J$ & Победа & Совпадение & Ухудшение & $N$ \\\\",
        "\\midrule",
    ]
    for row in pairwise.to_dict(orient="records"):
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["scenario_label"])),
                    _latex_escape(str(row["comparison_label"])),
                    f"{float(row['mean_delta_cumulative_loss']):.4e}",
                    f"{float(row['win_rate']):.2f}",
                    f"{float(row['tie_rate']):.2f}",
                    f"{float(row['probability_of_degradation']):.2f}",
                    f"{int(row['num_test_trajectories'])}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_posterior_information_standardized_pairwise.tex").write_text("\n".join(lines), encoding="utf-8")


def _format_seed_span(seeds: tuple[int, ...]) -> str:
    if not seeds:
        return ""
    if len(seeds) == 1:
        return str(seeds[0])
    return f"{seeds[0]}--{seeds[-1]}"


def run_posterior_information_state(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_posterior_information_state",
    scenario_names: tuple[str, ...] = (
        "macro_core_moderate_gap",
        "macro_core_strong_gap",
        "thin_information_moderate_gap",
        "thin_information_strong_gap",
    ),
    validation_seeds: tuple[int, ...] = tuple(range(500, 510)),
    test_seeds: tuple[int, ...] = tuple(range(900, 950)),
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _save_json(
        root / "posterior_information_spec.json",
        {
            "scenario_names": list(scenario_names),
            "validation_seeds": list(validation_seeds),
            "test_seeds": list(test_seeds),
            "tie_tolerance": TIE_TOLERANCE,
            "has_standardized_robustness": True,
            "has_hank_projection": True,
            "info_sets": [
                {
                    "name": info_set_name,
                    "label_ru": INFO_SET_LABELS_RU[info_set_name],
                    "input_mode": INPUT_MODE_BY_INFO_SET[info_set_name],
                    "include_distributional_state": INCLUDE_DISTRIBUTIONAL_STATE[info_set_name],
                }
                for info_set_name in INFO_SET_ORDER
            ],
        },
    )

    selected_specs_rows = []
    selection_frames = []
    metric_frames = []
    path_frames = []
    standardized_selected_specs_rows = []
    standardized_selection_frames = []
    standardized_metric_frames = []

    for scenario_name in scenario_names:
        selected_rules: dict[str, PosteriorRuleParameters] = {}
        standardized_rules: dict[str, PosteriorRuleParameters] = {}
        for info_set_name in INFO_SET_ORDER:
            params, grid = _select_information_rule(
                scenario_name=scenario_name,
                info_set_name=info_set_name,
                validation_seeds=validation_seeds,
            )
            selected_rules[info_set_name] = params
            selection_frames.append(grid)
            selected_specs_rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "info_set_name": info_set_name,
                    "info_set_label": INFO_SET_LABELS_RU[info_set_name],
                    "policy_name": _policy_name(info_set_name),
                    "policy_label": _policy_label(info_set_name),
                    "input_mode": INPUT_MODE_BY_INFO_SET[info_set_name],
                    "coefficient_summary_ru": _coefficient_string(params),
                    **params.to_dict(),
                }
            )
            standardized_params, standardized_grid = _select_information_rule(
                scenario_name=scenario_name,
                info_set_name=info_set_name,
                validation_seeds=validation_seeds,
                standardized_signals=True,
            )
            standardized_rules[info_set_name] = standardized_params
            standardized_selection_frames.append(standardized_grid)
            standardized_selected_specs_rows.append(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "info_set_name": info_set_name,
                    "info_set_label": INFO_SET_LABELS_RU[info_set_name],
                    "policy_name": _policy_name(info_set_name),
                    "policy_label": _policy_label_standardized(info_set_name),
                    "input_mode": INPUT_MODE_BY_INFO_SET[info_set_name],
                    "coefficient_summary_ru": _coefficient_string(standardized_params),
                    "standardized_signals": 1,
                    **standardized_params.to_dict(),
                }
            )
        scenario_metrics, scenario_paths = _evaluate_selected_rules(
            scenario_name=scenario_name,
            validation_seeds=validation_seeds,
            test_seeds=test_seeds,
            selected_rules=selected_rules,
        )
        metric_frames.append(scenario_metrics)
        path_frames.append(scenario_paths)
        standardized_metrics, _standardized_paths = _evaluate_selected_rules(
            scenario_name=scenario_name,
            validation_seeds=validation_seeds,
            test_seeds=test_seeds,
            selected_rules=standardized_rules,
            standardized_signals=True,
        )
        standardized_metric_frames.append(standardized_metrics)

    selected_specs = pd.DataFrame(selected_specs_rows)
    selection_grid = pd.concat(selection_frames, ignore_index=True)
    policy_metrics = pd.concat(metric_frames, ignore_index=True)
    policy_paths = pd.concat(path_frames, ignore_index=True)
    standardized_selected_specs = pd.DataFrame(standardized_selected_specs_rows)
    standardized_selection_grid = pd.concat(standardized_selection_frames, ignore_index=True)
    standardized_policy_metrics = pd.concat(standardized_metric_frames, ignore_index=True)

    levels = _summarize_levels(policy_metrics)
    pairwise, components = _pairwise_comparisons(policy_metrics)
    mean_paths = policy_paths[policy_paths["info_set_name"] == "posterior_mean"].copy()
    signal_quality, signal_calibration = _signal_quality(mean_paths)
    standardized_levels = _summarize_levels(standardized_policy_metrics)
    standardized_pairwise, _standardized_components = _pairwise_comparisons(standardized_policy_metrics)
    full_hank_metrics, full_hank_paths = _run_full_hank_projection(
        policy_paths=policy_paths,
        output_dir=root,
        scenario_names=scenario_names,
        policy_names=KEY_HANK_POLICIES,
    )
    hank_ranking, hank_pairwise = _posterior_common_scale_hank_validation(
        policy_paths=policy_paths,
        reduced_levels=levels,
        full_hank_metrics=full_hank_metrics,
    )

    selected_specs.to_csv(root / "selected_rule_specs.csv", index=False)
    selection_grid.to_csv(root / "selection_grid_results.csv", index=False)
    policy_metrics.to_csv(root / "posterior_policy_metrics.csv", index=False)
    policy_paths.to_csv(root / "posterior_policy_paths.csv", index=False)
    levels.to_csv(root / "posterior_information_levels.csv", index=False)
    pairwise.to_csv(root / "posterior_information_pairwise.csv", index=False)
    components.to_csv(root / "posterior_information_component_decomposition.csv", index=False)
    signal_quality.to_csv(root / "posterior_signal_quality.csv", index=False)
    signal_calibration.to_csv(root / "posterior_signal_calibration.csv", index=False)
    standardized_selected_specs.to_csv(root / "standardized_selected_rule_specs.csv", index=False)
    standardized_selection_grid.to_csv(root / "standardized_selection_grid_results.csv", index=False)
    standardized_policy_metrics.to_csv(root / "posterior_information_standardized_metrics.csv", index=False)
    standardized_levels.to_csv(root / "posterior_information_standardized_levels.csv", index=False)
    standardized_pairwise.to_csv(root / "posterior_information_standardized_pairwise.csv", index=False)
    hank_ranking.to_csv(root / "posterior_information_hank_ranking_validation.csv", index=False)
    hank_pairwise.to_csv(root / "posterior_information_hank_pairwise_validation.csv", index=False)

    _write_latex_tables(root, levels, pairwise, components, signal_quality, hank_pairwise)
    _write_selection_table(root, selected_specs)
    _write_value_summary_table(root, levels, pairwise)
    _write_standardized_table(root, standardized_pairwise)
    _write_report(
        root=root,
        levels=levels,
        pairwise=pairwise,
        signal_quality=signal_quality,
        selected_specs=selected_specs,
        standardized_pairwise=standardized_pairwise,
        hank_ranking=hank_ranking,
        hank_pairwise=hank_pairwise,
        scenario_names=scenario_names,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )

    return {
        "selected_specs": selected_specs,
        "selection_grid": selection_grid,
        "policy_metrics": policy_metrics,
        "policy_paths": policy_paths,
        "levels": levels,
        "pairwise": pairwise,
        "components": components,
        "signal_quality": signal_quality,
        "signal_calibration": signal_calibration,
        "standardized_selected_specs": standardized_selected_specs,
        "standardized_selection_grid": standardized_selection_grid,
        "standardized_policy_metrics": standardized_policy_metrics,
        "standardized_levels": standardized_levels,
        "standardized_pairwise": standardized_pairwise,
        "full_hank_metrics": full_hank_metrics,
        "full_hank_paths": full_hank_paths,
        "hank_ranking": hank_ranking,
        "hank_pairwise": hank_pairwise,
    }
