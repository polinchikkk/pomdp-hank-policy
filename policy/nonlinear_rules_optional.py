from __future__ import annotations

"""Дополнительные нелинейные правила для проверок устойчивости.

В основной постановке используются линейные правила. Этот модуль нужен только
для проверки: сохраняется ли ранжирование информационных состояний, если
разрешить простую нелинейность.
"""

from dataclasses import dataclass

import numpy as np

from .linear_rules import LinearRuleSpec, rule_spec_for_information_state


@dataclass(frozen=True)
class QuadraticRule:
    spec: LinearRuleSpec
    intercept: float
    linear_coefficients: tuple[float, ...]
    squared_coefficients: tuple[float, ...]
    lagged_rate_weight: float

    def rate(self, features: dict[str, float], lagged_rate: float) -> float:
        value = float(self.intercept)
        for name, coefficient in zip(self.spec.feature_names, self.linear_coefficients):
            feature = float(features[name])
            value += float(coefficient) * feature
        for name, coefficient in zip(self.spec.feature_names, self.squared_coefficients):
            feature = float(features[name])
            value += float(coefficient) * feature * feature
        value += float(self.lagged_rate_weight) * float(lagged_rate)
        return float(value)


@dataclass(frozen=True)
class FittedQuadraticRule:
    rule: QuadraticRule
    validation_loss: float
    num_candidates: int
    feature_scales: dict[str, float]


def zero_quadratic_rule(information_state: str) -> QuadraticRule:
    spec = rule_spec_for_information_state(information_state)
    zeros = tuple(0.0 for _ in spec.feature_names)
    return QuadraticRule(
        spec=spec,
        intercept=0.0,
        linear_coefficients=zeros,
        squared_coefficients=zeros,
        lagged_rate_weight=0.0,
    )


def fit_quadratic_rule(
    *,
    environment,
    information_state: str,
    validation_seeds: list[int],
    num_candidates: int = 250,
    seed: int = 811,
    extra_candidates: list[QuadraticRule] | None = None,
) -> FittedQuadraticRule:
    base_policy = zero_quadratic_rule(information_state)
    feature_scales = environment.feature_scales(
        policy=base_policy,
        information_state=information_state,
        seeds=validation_seeds,
    )
    candidates = _candidate_rules(
        information_state=information_state,
        feature_scales=feature_scales,
        num_candidates=num_candidates,
        seed=seed,
    )
    if extra_candidates:
        candidates = [*extra_candidates, *candidates]

    best_rule = candidates[0]
    best_loss = float("inf")
    for candidate in candidates:
        losses = [
            environment.simulate(policy=candidate, information_state=information_state, seed=sim_seed).total_loss
            for sim_seed in validation_seeds
        ]
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss:
            best_loss = mean_loss
            best_rule = candidate
    return FittedQuadraticRule(
        rule=best_rule,
        validation_loss=best_loss,
        num_candidates=len(candidates),
        feature_scales=feature_scales,
    )


def project_quadratic_rule_to_information_state(source: QuadraticRule, target_information_state: str) -> QuadraticRule:
    target_spec = rule_spec_for_information_state(target_information_state)
    source_linear = {
        _core_feature_name(name): coefficient
        for name, coefficient in zip(source.spec.feature_names, source.linear_coefficients)
    }
    source_squared = {
        _core_feature_name(name): coefficient
        for name, coefficient in zip(source.spec.feature_names, source.squared_coefficients)
    }
    return QuadraticRule(
        spec=target_spec,
        intercept=source.intercept,
        linear_coefficients=tuple(float(source_linear.get(_core_feature_name(name), 0.0)) for name in target_spec.feature_names),
        squared_coefficients=tuple(float(source_squared.get(_core_feature_name(name), 0.0)) for name in target_spec.feature_names),
        lagged_rate_weight=source.lagged_rate_weight,
    )


def _candidate_rules(
    *,
    information_state: str,
    feature_scales: dict[str, float],
    num_candidates: int,
    seed: int,
) -> list[QuadraticRule]:
    spec = rule_spec_for_information_state(information_state)
    rng = np.random.default_rng(seed)
    linear_grid = np.array([-0.014, -0.008, -0.004, 0.0, 0.004, 0.008, 0.014])
    squared_grid = np.array([-0.004, -0.002, 0.0, 0.002, 0.004])
    lagged_grid = np.array([0.0, 0.35, 0.60, 0.80])
    intercept_grid = np.array([-0.0015, -0.0005, 0.0, 0.0005, 0.0015])

    candidates = [zero_quadratic_rule(information_state)]
    while len(candidates) < max(num_candidates, 1):
        standardized_linear = rng.choice(linear_grid, size=len(spec.feature_names), replace=True)
        standardized_squared = rng.choice(squared_grid, size=len(spec.feature_names), replace=True)
        if rng.random() < 0.35:
            standardized_squared = np.zeros_like(standardized_squared)
        candidates.append(
            QuadraticRule(
                spec=spec,
                intercept=float(rng.choice(intercept_grid)),
                linear_coefficients=tuple(
                    float(value)
                    for value in _raw_coefficients(standardized_linear, spec.feature_names, feature_scales)
                ),
                squared_coefficients=tuple(
                    float(value)
                    for value in _raw_squared_coefficients(standardized_squared, spec.feature_names, feature_scales)
                ),
                lagged_rate_weight=float(rng.choice(lagged_grid)),
            )
        )
    return candidates[:num_candidates]


def _raw_coefficients(
    standardized: np.ndarray,
    feature_names: tuple[str, ...],
    feature_scales: dict[str, float],
) -> np.ndarray:
    return np.asarray(
        [
            value / max(float(feature_scales.get(name, 1.0)), 1e-5)
            for value, name in zip(standardized, feature_names)
        ],
        dtype=float,
    )


def _raw_squared_coefficients(
    standardized: np.ndarray,
    feature_names: tuple[str, ...],
    feature_scales: dict[str, float],
) -> np.ndarray:
    return np.asarray(
        [
            value / max(float(feature_scales.get(name, 1.0)) ** 2, 1e-8)
            for value, name in zip(standardized, feature_names)
        ],
        dtype=float,
    )


def _core_feature_name(name: str) -> str:
    for prefix in ("observed_", "filtered_", "true_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name
