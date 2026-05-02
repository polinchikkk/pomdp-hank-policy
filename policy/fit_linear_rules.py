from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linear_rules import LinearRule, rule_spec_for_information_state


@dataclass(frozen=True)
class FittedRule:
    rule: LinearRule
    validation_loss: float
    num_candidates: int
    feature_scales: dict[str, float]


def zero_rule(information_state: str) -> LinearRule:
    spec = rule_spec_for_information_state(information_state)
    return LinearRule(
        spec=spec,
        intercept=0.0,
        coefficients=tuple(0.0 for _ in spec.feature_names),
        lagged_rate_weight=0.0,
    )


def fit_linear_rule(
    *,
    environment,
    information_state: str,
    validation_seeds: list[int],
    num_candidates: int = 500,
    seed: int = 2027,
    extra_candidates: list[LinearRule] | None = None,
) -> FittedRule:
    spec = rule_spec_for_information_state(information_state)
    base_policy = zero_rule(information_state)
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
    return FittedRule(
        rule=best_rule,
        validation_loss=best_loss,
        num_candidates=len(candidates),
        feature_scales=feature_scales,
    )


def project_rule_to_information_state(source: LinearRule, target_information_state: str) -> LinearRule:
    target_spec = rule_spec_for_information_state(target_information_state)
    source_by_core_name = {
        _core_feature_name(name): coefficient
        for name, coefficient in zip(source.spec.feature_names, source.coefficients)
    }
    coefficients = tuple(
        float(source_by_core_name.get(_core_feature_name(name), 0.0))
        for name in target_spec.feature_names
    )
    return LinearRule(
        spec=target_spec,
        intercept=source.intercept,
        coefficients=coefficients,
        lagged_rate_weight=source.lagged_rate_weight,
    )


def _candidate_rules(
    *,
    information_state: str,
    feature_scales: dict[str, float],
    num_candidates: int,
    seed: int,
) -> list[LinearRule]:
    spec = rule_spec_for_information_state(information_state)
    rng = np.random.default_rng(seed)
    standardized_grid = np.array([-0.018, -0.012, -0.006, -0.003, 0.0, 0.003, 0.006, 0.012, 0.018])
    lagged_grid = np.array([0.0, 0.35, 0.60, 0.80, 0.90])
    intercept_grid = np.array([-0.002, -0.001, 0.0, 0.001, 0.002])

    candidates = [zero_rule(information_state)]
    candidates.extend(_structured_candidates(information_state, feature_scales))

    while len(candidates) < max(num_candidates, 1):
        standardized = rng.choice(standardized_grid, size=len(spec.feature_names), replace=True)
        if rng.random() < 0.25:
            mask = rng.random(len(spec.feature_names)) < 0.5
            standardized = standardized * mask
        coefficients = _raw_coefficients(standardized, spec.feature_names, feature_scales)
        candidates.append(
            LinearRule(
                spec=spec,
                intercept=float(rng.choice(intercept_grid)),
                coefficients=tuple(float(value) for value in coefficients),
                lagged_rate_weight=float(rng.choice(lagged_grid)),
            )
        )
    return candidates[:num_candidates]


def _structured_candidates(information_state: str, feature_scales: dict[str, float]) -> list[LinearRule]:
    spec = rule_spec_for_information_state(information_state)
    candidates: list[LinearRule] = []
    for lagged_weight in (0.35, 0.60, 0.80):
        for response in (0.004, 0.008, 0.012):
            standardized = np.zeros(len(spec.feature_names), dtype=float)
            for index, name in enumerate(spec.feature_names):
                if "inflation" in name:
                    standardized[index] = response
                elif "output" in name:
                    standardized[index] = 0.65 * response
                elif "natural_rate" in name:
                    standardized[index] = 0.50 * response
                elif "mean_mpc" in name or "low_liquidity" in name:
                    standardized[index] = -0.35 * response
            coefficients = _raw_coefficients(standardized, spec.feature_names, feature_scales)
            candidates.append(
                LinearRule(
                    spec=spec,
                    intercept=0.0,
                    coefficients=tuple(float(value) for value in coefficients),
                    lagged_rate_weight=lagged_weight,
                )
            )
    return candidates


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


def _core_feature_name(name: str) -> str:
    for prefix in ("observed_", "filtered_", "true_"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name
