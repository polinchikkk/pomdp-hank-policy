from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LinearRuleSpec:
    information_state: str
    feature_names: tuple[str, ...]
    includes_lagged_rate: bool = True


@dataclass(frozen=True)
class LinearRule:
    spec: LinearRuleSpec
    intercept: float
    coefficients: tuple[float, ...]
    lagged_rate_weight: float

    def rate(self, features: dict[str, float], lagged_rate: float) -> float:
        value = float(self.intercept)
        for name, coefficient in zip(self.spec.feature_names, self.coefficients):
            value += float(coefficient) * float(features[name])
        if self.spec.includes_lagged_rate:
            value += float(self.lagged_rate_weight) * float(lagged_rate)
        return float(value)


def rule_spec_for_information_state(name: str) -> LinearRuleSpec:
    if name == "aggregate_only":
        return LinearRuleSpec(
            information_state=name,
            feature_names=("observed_inflation_gap", "observed_output_gap"),
        )
    if name == "filtered_aggregates":
        return LinearRuleSpec(
            information_state=name,
            feature_names=("filtered_inflation_gap", "filtered_output_gap", "filtered_natural_rate_gap"),
        )
    if name == "distributional":
        return LinearRuleSpec(
            information_state=name,
            feature_names=(
                "filtered_inflation_gap",
                "filtered_output_gap",
                "filtered_natural_rate_gap",
                "filtered_mean_mpc",
                "filtered_low_liquidity_share",
            ),
        )
    if name == "distributional_mpc":
        return LinearRuleSpec(
            information_state=name,
            feature_names=(
                "filtered_inflation_gap",
                "filtered_output_gap",
                "filtered_natural_rate_gap",
                "filtered_mean_mpc",
            ),
        )
    if name == "distributional_liquidity":
        return LinearRuleSpec(
            information_state=name,
            feature_names=(
                "filtered_inflation_gap",
                "filtered_output_gap",
                "filtered_natural_rate_gap",
                "filtered_low_liquidity_share",
            ),
        )
    if name == "full_information":
        return LinearRuleSpec(
            information_state=name,
            feature_names=(
                "true_inflation_gap",
                "true_output_gap",
                "true_natural_rate_gap",
                "true_mean_mpc",
                "true_low_liquidity_share",
            ),
        )
    raise ValueError(f"Unknown information state: {name}")


def coefficient_vector(rule: LinearRule) -> np.ndarray:
    return np.asarray((rule.intercept, *rule.coefficients, rule.lagged_rate_weight), dtype=float)
