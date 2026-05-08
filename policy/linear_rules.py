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
            feature_names=("pi_obs", "Y_obs"),
        )
    if name == "aggregate_history":
        return LinearRuleSpec(
            information_state=name,
            feature_names=("pi_obs", "Y_obs", "pi_obs_lag", "Y_obs_lag"),
        )
    if name == "filtered_aggregates":
        return LinearRuleSpec(
            information_state=name,
            feature_names=("E_pi", "E_Y", "E_C"),
        )
    if name == "observed_distribution":
        return LinearRuleSpec(
            information_state=name,
            feature_names=(
                "pi_obs",
                "Y_obs",
                "C_obs",
                "mean_mpc_obs",
                "low_liquidity_share_obs",
                "interest_exposure_obs",
            ),
        )
    if name == "filtered_distribution":
        return LinearRuleSpec(
            information_state=name,
            feature_names=(
                "E_pi",
                "E_Y",
                "E_C",
                "E_mean_mpc",
                "E_low_liquidity_share",
                "E_interest_exposure",
            ),
        )
    if name == "filtered_distribution_mpc":
        return LinearRuleSpec(
            information_state=name,
            feature_names=(
                "E_pi",
                "E_Y",
                "E_C",
                "E_mean_mpc",
            ),
        )
    if name == "filtered_distribution_liquidity":
        return LinearRuleSpec(
            information_state=name,
            feature_names=(
                "E_pi",
                "E_Y",
                "E_C",
                "E_low_liquidity_share",
            ),
        )
    if name == "filtered_distribution_exposure":
        return LinearRuleSpec(
            information_state=name,
            feature_names=(
                "E_pi",
                "E_Y",
                "E_C",
                "E_interest_exposure",
            ),
        )
    if name == "full_information":
        return LinearRuleSpec(
            information_state=name,
            feature_names=(
                "pi",
                "Y",
                "C",
                "mean_mpc",
                "low_liquidity_share",
                "interest_exposure",
            ),
        )
    raise ValueError(f"Unknown information state: {name}")


def coefficient_vector(rule: LinearRule) -> np.ndarray:
    return np.asarray((rule.intercept, *rule.coefficients, rule.lagged_rate_weight), dtype=float)
