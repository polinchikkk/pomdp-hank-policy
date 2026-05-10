from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from .fit_linear_rules import FittedRule
from .linear_rules import LinearRule, rule_spec_for_information_state


@dataclass(frozen=True)
class LinearRuleOptimizationBounds:
    intercept_abs_bound: float = 0.01
    standardized_coefficient_abs_bound: float = 0.05
    lagged_rate_min: float = 0.0
    lagged_rate_max: float = 0.99


@dataclass(frozen=True)
class ContinuousLinearRuleFit:
    rule: LinearRule
    validation_loss: float
    feature_scales: dict[str, float]
    seed: int
    num_starts: int
    methods: tuple[str, ...]
    best_method: str
    best_start_index: int
    converged: bool
    num_function_evaluations: int
    message: str
    regularization_strength: float = 0.0
    penalized_validation_loss: float | None = None


def fit_linear_rule_continuous(
    *,
    environment,
    information_state: str,
    validation_seeds: list[int],
    feature_scales: dict[str, float],
    initial_rules: Iterable[LinearRule] = (),
    seed: int = 2027,
    num_starts: int = 8,
    methods: tuple[str, ...] = ("L-BFGS-B", "Powell", "Nelder-Mead"),
    bounds: LinearRuleOptimizationBounds | None = None,
    maxiter: int = 120,
    regularization_strength: float = 0.0,
    sign_constraints: dict[str, str] | None = None,
) -> ContinuousLinearRuleFit:
    """Fit an interpretable linear rule by continuous validation-loss minimization.

    Coefficients are optimized in standardized units. This keeps the numerical
    problem well scaled while the returned rule is expressed in raw feature units.
    """

    spec = rule_spec_for_information_state(information_state)
    bounds = LinearRuleOptimizationBounds() if bounds is None else bounds
    rng = np.random.default_rng(seed)
    starts = _initial_points(
        information_state=information_state,
        feature_scales=feature_scales,
        initial_rules=tuple(initial_rules),
        seed_rng=rng,
        num_starts=num_starts,
        bounds=bounds,
    )
    scipy_bounds = _scipy_bounds(
        feature_names=spec.feature_names,
        bounds=bounds,
        sign_constraints=sign_constraints,
    )

    best_x = starts[0]
    best_objective = float("inf")
    best_method = methods[0] if methods else "none"
    best_start_index = 0
    best_success = False
    best_message = ""
    total_evaluations = 0

    def unpenalized_loss(vector: np.ndarray) -> float:
        clipped = _clip_vector(vector, scipy_bounds)
        rule = _vector_to_rule(clipped, information_state, feature_scales)
        try:
            losses = [
                environment.simulate(policy=rule, information_state=information_state, seed=sim_seed).total_loss
                for sim_seed in validation_seeds
            ]
        except Exception:
            return 1e12
        value = float(np.mean(losses))
        if not np.isfinite(value):
            return 1e12
        return value

    def objective(vector: np.ndarray) -> float:
        clipped = _clip_vector(vector, scipy_bounds)
        value = unpenalized_loss(clipped)
        penalty = float(regularization_strength) * float(np.sum(clipped[1:-1] ** 2))
        return value + penalty

    for start_index, start in enumerate(starts):
        for method in methods:
            result = minimize(
                objective,
                start,
                method=method,
                bounds=scipy_bounds,
                options=_method_options(method, maxiter=maxiter),
            )
            total_evaluations += int(getattr(result, "nfev", 0))
            value = float(result.fun) if np.isfinite(result.fun) else float("inf")
            if value < best_objective:
                best_objective = value
                best_x = _clip_vector(np.asarray(result.x, dtype=float), scipy_bounds)
                best_method = method
                best_start_index = start_index
                best_success = bool(result.success)
                best_message = str(result.message)

    best_unpenalized = unpenalized_loss(best_x)

    return ContinuousLinearRuleFit(
        rule=_vector_to_rule(best_x, information_state, feature_scales),
        validation_loss=best_unpenalized,
        feature_scales=feature_scales,
        seed=int(seed),
        num_starts=int(len(starts)),
        methods=tuple(methods),
        best_method=best_method,
        best_start_index=int(best_start_index),
        converged=bool(best_success),
        num_function_evaluations=int(total_evaluations),
        message=best_message,
        regularization_strength=float(regularization_strength),
        penalized_validation_loss=float(best_objective),
    )


def fitted_rule_as_continuous_like(
    *,
    fit: FittedRule,
    seed: int,
    mode: str,
) -> ContinuousLinearRuleFit:
    return ContinuousLinearRuleFit(
        rule=fit.rule,
        validation_loss=float(fit.validation_loss),
        feature_scales=fit.feature_scales,
        seed=int(seed),
        num_starts=0,
        methods=(mode,),
        best_method=mode,
        best_start_index=0,
        converged=True,
        num_function_evaluations=int(fit.num_candidates),
        message="candidate search",
        regularization_strength=0.0,
        penalized_validation_loss=float(fit.validation_loss),
    )


def _initial_points(
    *,
    information_state: str,
    feature_scales: dict[str, float],
    initial_rules: tuple[LinearRule, ...],
    seed_rng: np.random.Generator,
    num_starts: int,
    bounds: LinearRuleOptimizationBounds,
) -> list[np.ndarray]:
    spec = rule_spec_for_information_state(information_state)
    points: list[np.ndarray] = []
    for rule in initial_rules:
        if rule.spec.information_state != information_state:
            continue
        points.append(_rule_to_vector(rule, feature_scales))
    points.append(np.zeros(len(spec.feature_names) + 2, dtype=float))
    while len(points) < max(num_starts, 1):
        point = np.zeros(len(spec.feature_names) + 2, dtype=float)
        point[0] = seed_rng.uniform(-0.25, 0.25) * bounds.intercept_abs_bound
        point[1:-1] = seed_rng.uniform(
            -0.5 * bounds.standardized_coefficient_abs_bound,
            0.5 * bounds.standardized_coefficient_abs_bound,
            size=len(spec.feature_names),
        )
        point[-1] = seed_rng.uniform(bounds.lagged_rate_min, min(bounds.lagged_rate_max, 0.9))
        points.append(point)
    scipy_bounds = _scipy_bounds(feature_names=spec.feature_names, bounds=bounds)
    return [_clip_vector(point, scipy_bounds) for point in points[: max(num_starts, 1)]]


def _rule_to_vector(rule: LinearRule, feature_scales: dict[str, float]) -> np.ndarray:
    standardized = [
        float(coefficient) * max(float(feature_scales.get(name, 1.0)), 1e-5)
        for name, coefficient in zip(rule.spec.feature_names, rule.coefficients)
    ]
    return np.asarray((rule.intercept, *standardized, rule.lagged_rate_weight), dtype=float)


def _vector_to_rule(
    vector: np.ndarray,
    information_state: str,
    feature_scales: dict[str, float],
) -> LinearRule:
    spec = rule_spec_for_information_state(information_state)
    coefficients = [
        float(value) / max(float(feature_scales.get(name, 1.0)), 1e-5)
        for value, name in zip(vector[1:-1], spec.feature_names)
    ]
    return LinearRule(
        spec=spec,
        intercept=float(vector[0]),
        coefficients=tuple(coefficients),
        lagged_rate_weight=float(vector[-1]),
    )


def _scipy_bounds(
    *,
    feature_names: tuple[str, ...],
    bounds: LinearRuleOptimizationBounds,
    sign_constraints: dict[str, str] | None = None,
) -> list[tuple[float, float]]:
    feature_bounds = []
    for feature_name in feature_names:
        lower = -bounds.standardized_coefficient_abs_bound
        upper = bounds.standardized_coefficient_abs_bound
        sign = _feature_sign_constraint(feature_name, sign_constraints or {})
        if sign == "nonnegative":
            lower = max(lower, 0.0)
        elif sign == "nonpositive":
            upper = min(upper, 0.0)
        feature_bounds.append((lower, upper))
    return [
        (-bounds.intercept_abs_bound, bounds.intercept_abs_bound),
        *feature_bounds,
        (bounds.lagged_rate_min, bounds.lagged_rate_max),
    ]


def _feature_sign_constraint(feature_name: str, sign_constraints: dict[str, str]) -> str | None:
    canonical = _canonical_constraint_name(feature_name)
    value = sign_constraints.get(canonical)
    if value in {"nonnegative", "nonpositive"}:
        return str(value)
    return None


def _canonical_constraint_name(feature_name: str) -> str:
    if feature_name.endswith("_lag"):
        feature_name = feature_name[: -len("_lag")]
    if feature_name in {"pi", "pi_obs", "E_pi"}:
        return "inflation"
    if feature_name in {"Y", "Y_obs", "E_Y", "output_gap"}:
        return "output_gap"
    if feature_name in {"C", "C_obs", "E_C"}:
        return "consumption"
    if "mean_mpc" in feature_name:
        return "mpc"
    if "low_liquidity" in feature_name:
        return "low_liquidity"
    if "interest_exposure" in feature_name:
        return "interest_exposure"
    return feature_name


def _clip_vector(vector: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    clipped = np.asarray(vector, dtype=float).copy()
    for index, (lower, upper) in enumerate(bounds):
        clipped[index] = min(max(clipped[index], lower), upper)
    return clipped


def _method_options(method: str, *, maxiter: int) -> dict[str, float | int]:
    if method == "L-BFGS-B":
        return {"maxiter": int(maxiter), "ftol": 1e-12, "maxls": 20}
    if method == "Powell":
        return {"maxiter": int(maxiter), "xtol": 1e-5, "ftol": 1e-10}
    if method == "Nelder-Mead":
        return {"maxiter": int(maxiter), "xatol": 1e-5, "fatol": 1e-10}
    return {"maxiter": int(maxiter)}
