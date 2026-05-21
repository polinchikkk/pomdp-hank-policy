from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hank_ssj import HankSSJPolicyEnvironment, PolicyLossWeights, TrajectoryLoss
from policy.linear_rules import LinearRule, LinearRuleSpec, coefficient_vector
from policy.optimize_rules import compare_paired_losses


AGGREGATE_STATE = "filtered_aggregates"
DISTRIBUTION_STATE = "filtered_distribution"
AGGREGATE_CORE = ("E_pi", "E_Y", "E_C")
DISTRIBUTION_FEATURES = ("E_mean_mpc", "E_low_liquidity_share", "E_interest_exposure")


@dataclass(frozen=True)
class AlternativeRuleClass:
    name: str
    label_ru: str
    aggregate_features: tuple[str, ...]
    distribution_features: tuple[str, ...]
    fit_method: str
    sign_restricted: bool = False
    appendix_only: bool = False
    l1_penalty: float = 0.0
    l2_penalty: float = 0.0
    note: str = ""


RULE_CLASSES = (
    AlternativeRuleClass(
        name="taylor",
        label_ru="Правило Тейлора",
        aggregate_features=("E_pi", "E_Y"),
        distribution_features=("E_pi", "E_Y"),
        fit_method="continuous",
        sign_restricted=True,
        note="Одинаковая форма правила; распределительные признаки не входят напрямую.",
    ),
    AlternativeRuleClass(
        name="taylor_consumption",
        label_ru="Тейлор + потребление",
        aggregate_features=AGGREGATE_CORE,
        distribution_features=AGGREGATE_CORE,
        fit_method="continuous",
        sign_restricted=True,
        note="Добавляет потребление, но не распределительные признаки.",
    ),
    AlternativeRuleClass(
        name="taylor_distribution",
        label_ru="Тейлор + распределительные показатели",
        aggregate_features=AGGREGATE_CORE,
        distribution_features=(*AGGREGATE_CORE, *DISTRIBUTION_FEATURES),
        fit_method="continuous",
        sign_restricted=True,
        note="Прямо добавляет распределительные статистики к Taylor-like правилу.",
    ),
    AlternativeRuleClass(
        name="restricted_sign",
        label_ru="Правило с ограничениями на знаки",
        aggregate_features=AGGREGATE_CORE,
        distribution_features=(*AGGREGATE_CORE, *DISTRIBUTION_FEATURES),
        fit_method="continuous",
        sign_restricted=True,
        note="Положительная реакция на инфляцию и разрыв выпуска; знаки распределительных коэффициентов не фиксируются.",
    ),
    AlternativeRuleClass(
        name="optimized_linear",
        label_ru="Оптимизированное линейное правило",
        aggregate_features=AGGREGATE_CORE,
        distribution_features=(*AGGREGATE_CORE, *DISTRIBUTION_FEATURES),
        fit_method="continuous",
        sign_restricted=False,
        note="Непрерывная оптимизация validation loss без ограничений на знаки коэффициентов.",
    ),
    AlternativeRuleClass(
        name="ridge_regularized",
        label_ru="Ridge-регуляризация",
        aggregate_features=AGGREGATE_CORE,
        distribution_features=(*AGGREGATE_CORE, *DISTRIBUTION_FEATURES),
        fit_method="continuous_regularized",
        l2_penalty=1e-3,
        sign_restricted=False,
        note="Правило минимизирует validation loss с L2-регуляризацией стандартизированных коэффициентов.",
    ),
    AlternativeRuleClass(
        name="elastic_net",
        label_ru="LASSO/ElasticNet",
        aggregate_features=AGGREGATE_CORE,
        distribution_features=(*AGGREGATE_CORE, *DISTRIBUTION_FEATURES),
        fit_method="continuous_regularized",
        l1_penalty=2e-5,
        l2_penalty=5e-4,
        sign_restricted=False,
        note="Правило минимизирует validation loss с L1/L2-регуляризацией стандартизированных коэффициентов.",
    ),
    AlternativeRuleClass(
        name="small_neural",
        label_ru="Малое нелинейное правило",
        aggregate_features=AGGREGATE_CORE,
        distribution_features=(*AGGREGATE_CORE, *DISTRIBUTION_FEATURES),
        fit_method="neural",
        sign_restricted=False,
        appendix_only=True,
        note="Небольшая сеть используется только как проверка нелинейности, не как основной вклад.",
    ),
)


@dataclass(frozen=True)
class PolicyClassRobustnessSpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    output_dir: str
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    rule_classes: tuple[str, ...]
    maxiter: int
    candidate_seed: int
    note: str


@dataclass(frozen=True)
class FittedAlternativePolicy:
    policy: object
    fit_method: str
    validation_loss: float
    converged: bool
    message: str
    coefficient_vector: list[float] | None = None
    selected_nonzero_distribution_terms: int | None = None


class FeaturePolicy:
    def __init__(self, *, feature_names: tuple[str, ...], model, label: str) -> None:
        self.feature_names = feature_names
        self.model = model
        self.label = label

    def rate_path(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(features), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the value of distributional information across policy-rule classes.")
    parser.add_argument("--information-inputs", default="outputs/ssj/stochastic/state_space/information_inputs/information_state_inputs_long.csv")
    parser.add_argument("--hank-observables", default="outputs/ssj/stochastic/hank_observables.csv")
    parser.add_argument("--jacobians", default="outputs/ssj/jacobians.npz")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/policy_class_robustness")
    parser.add_argument("--validation-seeds", default="900:905")
    parser.add_argument("--test-seeds", default="906:911")
    parser.add_argument("--candidate-seed", type=int, default=7027)
    parser.add_argument("--maxiter", type=int, default=80)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_seeds = _parse_seed_range(args.validation_seeds)
    test_seeds = _parse_seed_range(args.test_seeds)

    environment = HankSSJPolicyEnvironment.from_files(
        information_inputs_csv=Path(args.information_inputs),
        hank_observables_csv=Path(args.hank_observables),
        jacobians_npz=Path(args.jacobians),
        loss_weights=PolicyLossWeights(),
    )

    summary_rows: list[dict[str, object]] = []
    loss_rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []
    for index, rule_class in enumerate(RULE_CLASSES):
        print(f"Policy class {rule_class.name} ({index + 1}/{len(RULE_CLASSES)})", flush=True)
        aggregate_fit = _fit_policy(
            environment=environment,
            rule_class=rule_class,
            information_state=AGGREGATE_STATE,
            feature_names=rule_class.aggregate_features,
            validation_seeds=validation_seeds,
            seed=args.candidate_seed + 10 * index,
            maxiter=args.maxiter,
        )
        distribution_fit = _fit_policy(
            environment=environment,
            rule_class=rule_class,
            information_state=DISTRIBUTION_STATE,
            feature_names=rule_class.distribution_features,
            validation_seeds=validation_seeds,
            seed=args.candidate_seed + 10 * index + 1,
            maxiter=args.maxiter,
        )
        losses = _evaluate_pair(
            environment=environment,
            rule_class=rule_class,
            aggregate_fit=aggregate_fit,
            distribution_fit=distribution_fit,
            test_seeds=test_seeds,
        )
        loss_rows.extend(losses.to_dict(orient="records"))
        summary_rows.append(_summary_row(rule_class, aggregate_fit, distribution_fit, losses))
        rule_rows.extend(_rule_rows(rule_class, AGGREGATE_STATE, aggregate_fit))
        rule_rows.extend(_rule_rows(rule_class, DISTRIBUTION_STATE, distribution_fit))

    summary = pd.DataFrame(summary_rows)
    losses = pd.DataFrame(loss_rows)
    rules = pd.DataFrame(rule_rows)
    summary.to_csv(output_dir / "policy_class_robustness_summary.csv", index=False)
    losses.to_csv(output_dir / "policy_class_robustness_trajectory_losses.csv", index=False)
    rules.to_csv(output_dir / "policy_class_fitted_rules.csv", index=False)
    _write_latex(summary, output_dir / "table_policy_class_robustness.tex")
    _write_report(summary, output_dir / "report_policy_class_robustness.md")
    _plot_policy_class_robustness(summary, Path("article/figures/fig_policy_class_robustness.pdf"))

    spec = PolicyClassRobustnessSpec(
        information_inputs=args.information_inputs,
        hank_observables=args.hank_observables,
        jacobians=args.jacobians,
        output_dir=args.output_dir,
        validation_seeds=tuple(validation_seeds),
        test_seeds=tuple(test_seeds),
        rule_classes=tuple(item.name for item in RULE_CLASSES),
        maxiter=int(args.maxiter),
        candidate_seed=int(args.candidate_seed),
        note=(
            "Проверка показывает, сохраняется ли предельная ценность распределительной информации "
            "при разумных альтернативных классах правил: Taylor-like, restricted sign, ridge, Elastic Net "
            "и малое нелинейное правило как appendix-проверка."
        ),
    )
    (output_dir / "policy_class_robustness_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'policy_class_robustness_summary.csv'}")
    print("Wrote article/figures/fig_policy_class_robustness.pdf")


def _fit_policy(
    *,
    environment: HankSSJPolicyEnvironment,
    rule_class: AlternativeRuleClass,
    information_state: str,
    feature_names: tuple[str, ...],
    validation_seeds: list[int],
    seed: int,
    maxiter: int,
) -> FittedAlternativePolicy:
    if rule_class.fit_method in {"continuous", "continuous_regularized"}:
        return _fit_continuous_policy(
            environment=environment,
            rule_class=rule_class,
            information_state=information_state,
            feature_names=feature_names,
            validation_seeds=validation_seeds,
            seed=seed,
            maxiter=maxiter,
        )
    if rule_class.fit_method == "ridge":
        return _fit_supervised_policy(
            environment=environment,
            rule_class=rule_class,
            information_state=information_state,
            feature_names=feature_names,
            validation_seeds=validation_seeds,
            model_type="ridge",
            seed=seed,
        )
    if rule_class.fit_method == "elastic_net":
        return _fit_supervised_policy(
            environment=environment,
            rule_class=rule_class,
            information_state=information_state,
            feature_names=feature_names,
            validation_seeds=validation_seeds,
            model_type="elastic_net",
            seed=seed,
        )
    if rule_class.fit_method == "neural":
        return _fit_neural_policy(
            environment=environment,
            information_state=information_state,
            feature_names=feature_names,
            validation_seeds=validation_seeds,
            seed=seed,
        )
    raise ValueError(f"Unknown fit method: {rule_class.fit_method}")


def _fit_continuous_policy(
    *,
    environment: HankSSJPolicyEnvironment,
    rule_class: AlternativeRuleClass,
    information_state: str,
    feature_names: tuple[str, ...],
    validation_seeds: list[int],
    seed: int,
    maxiter: int,
) -> FittedAlternativePolicy:
    scales = _feature_scales(environment, information_state, feature_names, validation_seeds)
    initial = _supervised_linear_initial_rule(
        environment=environment,
        information_state=information_state,
        feature_names=feature_names,
        validation_seeds=validation_seeds,
        scales=scales,
    )
    bounds = _bounds_for_features(feature_names, sign_restricted=rule_class.sign_restricted)

    def objective(vector: np.ndarray) -> float:
        rule = _vector_to_rule(vector, information_state, feature_names, scales)
        loss = _validation_loss(environment, rule, information_state, validation_seeds)
        standardized = np.asarray(vector[1:-1], dtype=float)
        l1 = rule_class.l1_penalty * float(np.sum(np.sqrt(standardized**2 + 1e-10)))
        l2 = rule_class.l2_penalty * float(np.sum(standardized**2))
        return loss + l1 + l2

    start = _clip_to_bounds(_rule_to_vector(initial, scales), bounds)
    result = minimize(
        objective,
        start,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-12, "maxls": 20},
    )
    best_vector = np.asarray(result.x, dtype=float)
    best_rule = _vector_to_rule(best_vector, information_state, feature_names, scales)
    validation_loss = _validation_loss(environment, best_rule, information_state, validation_seeds)
    return FittedAlternativePolicy(
        policy=best_rule,
        fit_method=rule_class.fit_method,
        validation_loss=validation_loss,
        converged=bool(result.success),
        message=str(result.message),
        coefficient_vector=coefficient_vector(best_rule).tolist(),
        selected_nonzero_distribution_terms=_count_distribution_terms(best_rule),
    )


def _fit_supervised_policy(
    *,
    environment: HankSSJPolicyEnvironment,
    rule_class: AlternativeRuleClass,
    information_state: str,
    feature_names: tuple[str, ...],
    validation_seeds: list[int],
    model_type: str,
    seed: int,
) -> FittedAlternativePolicy:
    x, y = _optimal_rate_training_data(environment, information_state, feature_names, validation_seeds)
    lagged_candidates = (0.0, 0.35, 0.60, 0.80)
    best_rule: LinearRule | None = None
    best_loss = float("inf")
    best_message = ""
    for lagged_weight in lagged_candidates:
        y_transformed = y - lagged_weight * _lag_by_path(environment, validation_seeds, y)
        if model_type == "ridge":
            model = make_pipeline(StandardScaler(), Ridge(alpha=1e-4, fit_intercept=True))
        else:
            model = make_pipeline(
                StandardScaler(),
                ElasticNet(alpha=1e-6, l1_ratio=0.55, fit_intercept=True, max_iter=5000, random_state=seed),
            )
        model.fit(x, y_transformed)
        linear = model.named_steps[list(model.named_steps.keys())[-1]]
        scaler = model.named_steps["standardscaler"]
        coefficients_scaled = linear.coef_ / scaler.scale_
        intercept = linear.intercept_ - np.sum(linear.coef_ * scaler.mean_ / scaler.scale_)
        rule = LinearRule(
            spec=LinearRuleSpec(information_state=information_state, feature_names=feature_names),
            intercept=float(intercept),
            coefficients=tuple(float(value) for value in coefficients_scaled),
            lagged_rate_weight=float(lagged_weight),
        )
        losses = [
            environment.simulate(policy=rule, information_state=information_state, seed=item).total_loss
            for item in validation_seeds
        ]
        loss = float(np.mean(losses))
        if loss < best_loss:
            best_rule = rule
            best_loss = loss
            best_message = f"{model_type}, lagged_rate={lagged_weight}"
    if best_rule is None:
        raise RuntimeError("Could not fit supervised policy.")
    return FittedAlternativePolicy(
        policy=best_rule,
        fit_method=model_type,
        validation_loss=best_loss,
        converged=True,
        message=best_message,
        coefficient_vector=coefficient_vector(best_rule).tolist(),
        selected_nonzero_distribution_terms=_count_distribution_terms(best_rule),
    )


def _fit_neural_policy(
    *,
    environment: HankSSJPolicyEnvironment,
    information_state: str,
    feature_names: tuple[str, ...],
    validation_seeds: list[int],
    seed: int,
) -> FittedAlternativePolicy:
    x, y = _optimal_rate_training_data(environment, information_state, feature_names, validation_seeds)
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(8,),
            activation="tanh",
            alpha=1e-3,
            random_state=seed,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.2,
        ),
    )
    model.fit(x, y)
    policy = FeaturePolicy(feature_names=feature_names, model=model, label="small_neural")
    losses = [
        _simulate_feature_policy(environment, policy, information_state=information_state, seed=item).total_loss
        for item in validation_seeds
    ]
    return FittedAlternativePolicy(
        policy=policy,
        fit_method="small_neural",
        validation_loss=float(np.mean(losses)),
        converged=True,
        message="small MLP trained on local SSJ-optimal rate",
        coefficient_vector=None,
        selected_nonzero_distribution_terms=None,
    )


def _validation_loss(
    environment: HankSSJPolicyEnvironment,
    rule: LinearRule,
    information_state: str,
    validation_seeds: list[int],
) -> float:
    losses = [
        environment.simulate(policy=rule, information_state=information_state, seed=item).total_loss
        for item in validation_seeds
    ]
    return float(np.mean(losses))


def _evaluate_pair(
    *,
    environment: HankSSJPolicyEnvironment,
    rule_class: AlternativeRuleClass,
    aggregate_fit: FittedAlternativePolicy,
    distribution_fit: FittedAlternativePolicy,
    test_seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in environment.scenarios:
        for seed in test_seeds:
            aggregate_loss = _simulate_policy(
                environment,
                aggregate_fit.policy,
                information_state=AGGREGATE_STATE,
                scenario=scenario,
                seed=seed,
            )
            distribution_loss = _simulate_policy(
                environment,
                distribution_fit.policy,
                information_state=DISTRIBUTION_STATE,
                scenario=scenario,
                seed=seed,
            )
            rows.append(
                {
                    "rule_class": rule_class.name,
                    "rule_class_ru": rule_class.label_ru,
                    "appendix_only": rule_class.appendix_only,
                    "scenario": scenario,
                    "observation_seed": int(seed),
                    "loss_filtered_aggregates": aggregate_loss.total_loss,
                    "loss_filtered_distribution": distribution_loss.total_loss,
                    "delta_distribution_minus_aggregates": distribution_loss.total_loss - aggregate_loss.total_loss,
                    "output_gap_delta": distribution_loss.output_gap_loss - aggregate_loss.output_gap_loss,
                    "inflation_delta": distribution_loss.inflation_loss - aggregate_loss.inflation_loss,
                    "consumption_delta": distribution_loss.consumption_loss - aggregate_loss.consumption_loss,
                    "rate_smoothing_delta": distribution_loss.rate_smoothing_loss - aggregate_loss.rate_smoothing_loss,
                }
            )
    return pd.DataFrame(rows)


def _simulate_policy(
    environment: HankSSJPolicyEnvironment,
    policy: object,
    *,
    information_state: str,
    scenario: str,
    seed: int,
) -> TrajectoryLoss:
    if isinstance(policy, LinearRule):
        return environment.simulate_scenario(
            policy=policy,
            information_state=information_state,
            scenario=scenario,
            seed=seed,
        )
    return _simulate_feature_policy(environment, policy, information_state=information_state, scenario=scenario, seed=seed)


def _simulate_feature_policy(
    environment: HankSSJPolicyEnvironment,
    policy: FeaturePolicy,
    *,
    information_state: str,
    seed: int,
    scenario: str | None = None,
) -> TrajectoryLoss:
    scenarios = environment.scenarios if scenario is None else (scenario,)
    losses = []
    for scenario_name in scenarios:
        feature_matrix = environment.feature_matrix(
            scenario=scenario_name,
            information_state=information_state,
            seed=seed,
            feature_names=policy.feature_names,
        )
        base = environment._observables[(scenario_name,)].sort_values("period").reset_index(drop=True)
        periods = min(feature_matrix.shape[0], len(base), environment.periods)
        feature_matrix = feature_matrix[:periods]
        base = base.iloc[:periods]
        policy_rate = policy.rate_path(feature_matrix)[:periods]
        losses.append(_loss_from_rate_path(environment, base, policy_rate))
    return _mean_losses(losses)


def _loss_from_rate_path(environment: HankSSJPolicyEnvironment, base: pd.DataFrame, policy_rate: np.ndarray) -> TrajectoryLoss:
    periods = len(base)
    baseline_rate = base["i"].to_numpy(dtype=float)
    rate_change = policy_rate - np.r_[0.0, policy_rate[:-1]]
    rate_deviation = policy_rate - baseline_rate
    pi = base["pi"].to_numpy(dtype=float) + environment._effects["pi"][:periods, :periods] @ rate_deviation
    y = base["output_gap"].to_numpy(dtype=float) + environment._effects["output_gap"][:periods, :periods] @ rate_deviation
    c = base["C"].to_numpy(dtype=float) + environment._effects["C"][:periods, :periods] @ rate_deviation
    discounts = environment.discount ** np.arange(periods)
    weights = environment.loss_weights
    inflation_loss = float(np.sum(discounts * weights.inflation * pi**2))
    output_gap_loss = float(np.sum(discounts * weights.output_gap * y**2))
    consumption_loss = float(np.sum(discounts * weights.consumption * c**2))
    rate_smoothing_loss = float(np.sum(discounts * weights.rate_smoothing * rate_change**2))
    stability_penalty = environment._stability_penalty(policy_rate, rate_change)
    return TrajectoryLoss(
        total_loss=inflation_loss + output_gap_loss + consumption_loss + rate_smoothing_loss + stability_penalty,
        inflation_loss=inflation_loss,
        output_gap_loss=output_gap_loss,
        consumption_loss=consumption_loss,
        rate_smoothing_loss=rate_smoothing_loss,
        stability_penalty=stability_penalty,
    )


def _summary_row(
    rule_class: AlternativeRuleClass,
    aggregate_fit: FittedAlternativePolicy,
    distribution_fit: FittedAlternativePolicy,
    losses: pd.DataFrame,
) -> dict[str, object]:
    comparison = compare_paired_losses(
        left_name="filtered_distribution",
        right_name="filtered_aggregates",
        left_losses=losses["loss_filtered_distribution"].to_numpy(dtype=float),
        right_losses=losses["loss_filtered_aggregates"].to_numpy(dtype=float),
        tie_eps=1e-10,
    )
    return {
        "rule_class": rule_class.name,
        "rule_class_ru": rule_class.label_ru,
        "appendix_only": rule_class.appendix_only,
        "fit_method": rule_class.fit_method,
        "sign_restricted": rule_class.sign_restricted,
        "aggregate_num_features": len(rule_class.aggregate_features),
        "distribution_num_features": len(rule_class.distribution_features),
        "loss_filtered_aggregates": float(losses["loss_filtered_aggregates"].mean()),
        "loss_filtered_distribution": float(losses["loss_filtered_distribution"].mean()),
        "mean_delta": comparison.mean_delta,
        "loss_reduction": -comparison.mean_delta,
        "ci_low": comparison.ci_low,
        "ci_high": comparison.ci_high,
        "win_rate": comparison.win_rate,
        "tie_rate": comparison.tie_rate,
        "loss_rate": comparison.loss_rate,
        "num_trajectories": comparison.num_trajectories,
        "validation_loss_aggregate": aggregate_fit.validation_loss,
        "validation_loss_distribution": distribution_fit.validation_loss,
        "aggregate_converged": aggregate_fit.converged,
        "distribution_converged": distribution_fit.converged,
        "selected_nonzero_distribution_terms": distribution_fit.selected_nonzero_distribution_terms,
        "note": rule_class.note,
    }


def _optimal_rate_training_data(
    environment: HankSSJPolicyEnvironment,
    information_state: str,
    feature_names: tuple[str, ...],
    validation_seeds: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    x_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    for scenario in environment.scenarios:
        target = environment.optimal_rate_path(scenario=scenario)
        for seed in validation_seeds:
            features = environment.feature_matrix(
                scenario=scenario,
                information_state=information_state,
                seed=seed,
                feature_names=feature_names,
            )
            periods = min(features.shape[0], target.size)
            x_blocks.append(features[:periods])
            y_blocks.append(target[:periods])
    return np.vstack(x_blocks), np.concatenate(y_blocks)


def _supervised_linear_initial_rule(
    *,
    environment: HankSSJPolicyEnvironment,
    information_state: str,
    feature_names: tuple[str, ...],
    validation_seeds: list[int],
    scales: dict[str, float],
) -> LinearRule:
    x, y = _optimal_rate_training_data(environment, information_state, feature_names, validation_seeds)
    best_rule: LinearRule | None = None
    best_loss = float("inf")
    for lagged_weight in (0.0, 0.35, 0.60, 0.80):
        lagged = _lag_by_path(environment, validation_seeds, y)
        transformed = y - lagged_weight * lagged
        design = np.column_stack([np.ones(x.shape[0]), x])
        beta = np.linalg.solve(design.T @ design + 1e-8 * np.eye(design.shape[1]), design.T @ transformed)
        rule = LinearRule(
            spec=LinearRuleSpec(information_state=information_state, feature_names=feature_names),
            intercept=float(beta[0]),
            coefficients=tuple(float(value) for value in beta[1:]),
            lagged_rate_weight=float(lagged_weight),
        )
        losses = [
            environment.simulate(policy=rule, information_state=information_state, seed=item).total_loss
            for item in validation_seeds
        ]
        loss = float(np.mean(losses))
        if loss < best_loss:
            best_loss = loss
            best_rule = rule
    if best_rule is None:
        raise RuntimeError("No initial rule.")
    return _vector_to_rule(_rule_to_vector(best_rule, scales), information_state, feature_names, scales)


def _lag_by_path(environment: HankSSJPolicyEnvironment, validation_seeds: list[int], y: np.ndarray) -> np.ndarray:
    chunks: list[np.ndarray] = []
    cursor = 0
    for _scenario in environment.scenarios:
        target_size = environment.periods
        for _seed in validation_seeds:
            path = y[cursor : cursor + target_size]
            chunks.append(np.r_[0.0, path[:-1]])
            cursor += target_size
    return np.concatenate(chunks)


def _feature_scales(
    environment: HankSSJPolicyEnvironment,
    information_state: str,
    feature_names: tuple[str, ...],
    validation_seeds: list[int],
) -> dict[str, float]:
    values = []
    for scenario in environment.scenarios:
        for seed in validation_seeds:
            values.append(
                environment.feature_matrix(
                    scenario=scenario,
                    information_state=information_state,
                    seed=seed,
                    feature_names=feature_names,
                )
            )
    stacked = np.vstack(values)
    return {
        name: float(max(np.std(stacked[:, index], ddof=0), 1e-5))
        for index, name in enumerate(feature_names)
    }


def _bounds_for_features(feature_names: tuple[str, ...], *, sign_restricted: bool) -> list[tuple[float, float]]:
    coefficient_bound = 0.05
    bounds = [(-0.01, 0.01)]
    for name in feature_names:
        if sign_restricted and name in {"E_pi", "E_Y", "pi", "Y", "pi_obs", "Y_obs"}:
            bounds.append((0.0, coefficient_bound))
        else:
            bounds.append((-coefficient_bound, coefficient_bound))
    bounds.append((0.0, 0.99))
    return bounds


def _clip_to_bounds(vector: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    clipped = np.asarray(vector, dtype=float).copy()
    for index, (lower, upper) in enumerate(bounds):
        clipped[index] = min(max(clipped[index], lower), upper)
    return clipped


def _rule_to_vector(rule: LinearRule, scales: dict[str, float]) -> np.ndarray:
    standardized = [
        coefficient * max(float(scales.get(name, 1.0)), 1e-5)
        for name, coefficient in zip(rule.spec.feature_names, rule.coefficients)
    ]
    return np.asarray((rule.intercept, *standardized, rule.lagged_rate_weight), dtype=float)


def _vector_to_rule(
    vector: np.ndarray,
    information_state: str,
    feature_names: tuple[str, ...],
    scales: dict[str, float],
) -> LinearRule:
    coefficients = [
        float(value) / max(float(scales.get(name, 1.0)), 1e-5)
        for value, name in zip(vector[1:-1], feature_names)
    ]
    return LinearRule(
        spec=LinearRuleSpec(information_state=information_state, feature_names=feature_names),
        intercept=float(vector[0]),
        coefficients=tuple(coefficients),
        lagged_rate_weight=float(vector[-1]),
    )


def _count_distribution_terms(rule: LinearRule) -> int:
    return int(
        sum(
            abs(coefficient) > 1e-10 and name in DISTRIBUTION_FEATURES
            for name, coefficient in zip(rule.spec.feature_names, rule.coefficients)
        )
    )


def _mean_losses(losses: list[TrajectoryLoss]) -> TrajectoryLoss:
    fields = TrajectoryLoss.__dataclass_fields__.keys()
    return TrajectoryLoss(
        **{
            field: float(np.mean([getattr(loss, field) for loss in losses]))
            for field in fields
        }
    )


def _rule_rows(
    rule_class: AlternativeRuleClass,
    information_state: str,
    fit: FittedAlternativePolicy,
) -> list[dict[str, object]]:
    if not isinstance(fit.policy, LinearRule):
        return [
            {
                "rule_class": rule_class.name,
                "rule_class_ru": rule_class.label_ru,
                "information_state": information_state,
                "term": "nonlinear_policy",
                "coefficient": np.nan,
                "validation_loss": fit.validation_loss,
                "message": fit.message,
            }
        ]
    rows = [
        {
            "rule_class": rule_class.name,
            "rule_class_ru": rule_class.label_ru,
            "information_state": information_state,
            "term": "intercept",
            "coefficient": fit.policy.intercept,
            "validation_loss": fit.validation_loss,
            "message": fit.message,
        },
        {
            "rule_class": rule_class.name,
            "rule_class_ru": rule_class.label_ru,
            "information_state": information_state,
            "term": "lagged_rate",
            "coefficient": fit.policy.lagged_rate_weight,
            "validation_loss": fit.validation_loss,
            "message": fit.message,
        },
    ]
    for name, coefficient in zip(fit.policy.spec.feature_names, fit.policy.coefficients):
        rows.append(
            {
                "rule_class": rule_class.name,
                "rule_class_ru": rule_class.label_ru,
                "information_state": information_state,
                "term": name,
                "coefficient": coefficient,
                "validation_loss": fit.validation_loss,
                "message": fit.message,
            }
        )
    rows[0]["coefficient_vector"] = coefficient_vector(fit.policy).tolist()
    return rows


def _write_latex(summary: pd.DataFrame, path: Path) -> None:
    display = summary[
        [
            "rule_class_ru",
            "appendix_only",
            "loss_reduction",
            "ci_low",
            "ci_high",
            "win_rate",
            "selected_nonzero_distribution_terms",
        ]
    ].copy()
    display = display.rename(
        columns={
            "rule_class_ru": "Класс правила",
            "appendix_only": "Только приложение",
            "loss_reduction": "Снижение потерь",
            "ci_low": "Нижняя граница",
            "ci_high": "Верхняя граница",
            "win_rate": "Доля выигрышей",
            "selected_nonzero_distribution_terms": "Выбранные распр. признаки",
        }
    )
    display["Только приложение"] = display["Только приложение"].map({True: "да", False: "нет"})
    for column in ("Снижение потерь", "Нижняя граница", "Верхняя граница"):
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    display["Доля выигрышей"] = display["Доля выигрышей"].map(
        lambda value: "" if pd.isna(value) else f"{value:.3f}"
    )
    display["Выбранные распр. признаки"] = display["Выбранные распр. признаки"].map(
        lambda value: "" if pd.isna(value) else f"{int(value)}"
    )
    path.write_text(display.to_latex(index=False, escape=False), encoding="utf-8")


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Устойчивость к классу правила",
        "",
        "Проверяется, сохраняется ли предельная ценность распределительной информации при разумных альтернативных классах правил.",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['rule_class_ru']}: снижение потерь {row['loss_reduction']:.6g}, "
            f"интервал для разности [{row['ci_low']:.6g}, {row['ci_high']:.6g}], "
            f"доля выигрышей {row['win_rate']:.3g}."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_policy_class_robustness(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = summary.copy()
    x = np.arange(len(frame))
    means = frame["loss_reduction"].to_numpy(dtype=float)
    lower = -frame["ci_high"].to_numpy(dtype=float)
    upper = -frame["ci_low"].to_numpy(dtype=float)
    err_low = np.maximum(means - lower, 0.0)
    err_high = np.maximum(upper - means, 0.0)
    colors = ["#c06c2d" if not appendix else "#8a8f98" for appendix in frame["appendix_only"]]
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.bar(x, means, yerr=[err_low, err_high], capsize=4, color=colors, edgecolor="#222222", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(frame["rule_class_ru"], rotation=25, ha="right")
    ax.set_ylabel("Снижение потерь")
    ax.set_title("Устойчивость ценности распределительной информации к классу правила")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _parse_seed_range(value: str) -> list[int]:
    if ":" in value:
        left, right = value.split(":", maxsplit=1)
        return list(range(int(left), int(right) + 1))
    return [int(part) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
