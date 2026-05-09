from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from policy.linear_rules import LinearRule

from .kalman_filter import (
    DEFAULT_STATE_NAMES,
    FILTERED_OUTPUT_BY_STATE,
    OBSERVATION_BY_STATE,
    run_kalman_filter,
)
from .policy_evaluation import PolicyLossWeights, TrajectoryLoss
from .state_space import StateSpaceSpec


LOSS_STATE_NAMES = ("pi", "output_gap", "C")
POLICY_STATE_NAMES = (
    "pi",
    "Y",
    "C",
    "mean_mpc_centered",
    "share_low_liquidity_centered",
    "interest_exposure_centered",
)
DISTRIBUTIONAL_STATE_NAMES = (
    "mean_mpc_centered",
    "share_low_liquidity_centered",
    "interest_exposure_centered",
)


@dataclass(frozen=True)
class ClosedLoopDiagnostics:
    scenario: str
    observation_seed: int
    information_state: str
    mode: str
    iterations: int
    converged: bool
    rate_update_norm: float
    state_update_norm: float
    stability_penalty: float
    convergence_penalty: float
    max_abs_rate: float
    max_abs_rate_change: float
    missing_direct_jacobians: tuple[str, ...]
    fallback_effects: tuple[str, ...]


@dataclass(frozen=True)
class ClosedLoopResult:
    loss: TrajectoryLoss
    diagnostics: ClosedLoopDiagnostics


class ClosedLoopSSJEnvironment:
    """Evaluate frozen policy rules with locally consistent SSJ feedback.

    ``partial_local_projection`` uses baseline observations to choose the whole policy path once
    and then applies the local SSJ projection.

    ``closed_loop_local_projection`` iterates between the policy path, the counterfactual HANK/SSJ
    state, noisy observations and filtered information states.
    """

    def __init__(
        self,
        *,
        observables: pd.DataFrame,
        observations: pd.DataFrame,
        jacobians: dict[str, np.ndarray],
        state_space_specs: dict[str, StateSpaceSpec],
        loss_weights: PolicyLossWeights | None = None,
        ridge: float = 1e-10,
        discount: float = 1.0,
        max_abs_rate: float = 0.05,
        max_abs_rate_change: float = 0.03,
        stability_penalty_weight: float = 1e4,
        convergence_penalty_weight: float = 1e4,
    ) -> None:
        self.loss_weights = PolicyLossWeights() if loss_weights is None else loss_weights
        self.discount = float(discount)
        self.max_abs_rate = float(max_abs_rate)
        self.max_abs_rate_change = float(max_abs_rate_change)
        self.stability_penalty_weight = float(stability_penalty_weight)
        self.convergence_penalty_weight = float(convergence_penalty_weight)
        self.state_space_specs = state_space_specs

        self._observables = _observable_paths(observables)
        self._observations = _observation_paths(observations)
        self.scenarios = tuple(sorted({key[0] for key in self._observables}))
        self.seeds = tuple(sorted({key[1] for key in self._observations}))
        self.periods = int(max(len(frame) for frame in self._observables.values()))
        self._effects, self.effect_sources = _rate_path_effects(
            jacobians=jacobians,
            observables=observables,
            periods=self.periods,
            ridge=ridge,
        )
        self.missing_direct_jacobians = tuple(
            state for state, source in self.effect_sources.items() if source != "direct_jacobian"
        )

    @classmethod
    def from_files(
        cls,
        *,
        hank_observables_csv: Path,
        hank_observations_csv: Path,
        jacobians_npz: Path,
        state_space_spec_json: Path,
        **kwargs,
    ) -> "ClosedLoopSSJEnvironment":
        observables = pd.read_csv(hank_observables_csv)
        observations = pd.read_csv(hank_observations_csv)
        with np.load(jacobians_npz) as bundle:
            jacobians = {
                key: np.asarray(bundle[key], dtype=float)
                for key in bundle.files
                if key.startswith("J_")
            }
        state_space_specs = _load_state_space_specs(state_space_spec_json)
        return cls(
            observables=observables,
            observations=observations,
            jacobians=jacobians,
            state_space_specs=state_space_specs,
            **kwargs,
        )

    def simulate_scenario(
        self,
        *,
        policy: LinearRule,
        information_state: str,
        scenario: str,
        seed: int,
        mode: str = "closed_loop_local_projection",
        max_iterations: int = 5,
        min_iterations: int = 2,
        tolerance: float = 1e-8,
        damping: float = 0.75,
    ) -> ClosedLoopResult:
        if mode not in {"partial_local_projection", "closed_loop_local_projection"}:
            raise ValueError(f"Unknown closed-loop evaluation mode: {mode}")
        base = self._observables[(scenario,)].sort_values("period").reset_index(drop=True)
        observations = self._observations[(scenario, int(seed))].sort_values("period").reset_index(drop=True)
        periods = min(len(base), len(observations), self.periods)
        base = base.iloc[:periods].reset_index(drop=True)
        observations = observations.iloc[:periods].reset_index(drop=True)
        baseline_rate = base["i"].to_numpy(dtype=float)

        policy_rate = baseline_rate.copy()
        previous_state = self._counterfactual_state(base=base, rate_deviation=np.zeros(periods, dtype=float))
        iterations = 1 if mode == "partial_local_projection" else max_iterations
        converged = mode == "partial_local_projection"
        rate_update_norm = np.inf
        state_update_norm = np.inf

        for iteration in range(1, iterations + 1):
            rate_deviation = policy_rate - baseline_rate
            counterfactual_state = self._counterfactual_state(base=base, rate_deviation=rate_deviation)
            counterfactual_observations = self._counterfactual_observations(
                base=base,
                observations=observations,
                counterfactual_state=counterfactual_state,
            )
            new_rate = self._policy_rate_path(
                policy=policy,
                information_state=information_state,
                counterfactual_state=counterfactual_state,
                counterfactual_observations=counterfactual_observations,
            )
            rate_update_norm = float(np.linalg.norm(new_rate - policy_rate) / np.sqrt(periods))
            state_update_norm = float(
                _state_distance(counterfactual_state, previous_state, periods=periods)
            )
            if (
                mode == "closed_loop_local_projection"
                and iteration >= min_iterations
                and rate_update_norm <= tolerance
                and state_update_norm <= tolerance
            ):
                policy_rate = new_rate
                converged = True
                break
            if mode == "partial_local_projection":
                policy_rate = new_rate
                break
            policy_rate = damping * new_rate + (1.0 - damping) * policy_rate
            previous_state = counterfactual_state
        else:
            converged = False

        final_rate_deviation = policy_rate - baseline_rate
        final_state = self._counterfactual_state(base=base, rate_deviation=final_rate_deviation)
        loss, stability_penalty, convergence_penalty = self._loss(
            state=final_state,
            policy_rate=policy_rate,
            converged=converged,
            rate_update_norm=rate_update_norm,
            state_update_norm=state_update_norm,
        )
        diagnostics = ClosedLoopDiagnostics(
            scenario=scenario,
            observation_seed=int(seed),
            information_state=information_state,
            mode=mode,
            iterations=int(iteration),
            converged=bool(converged),
            rate_update_norm=float(rate_update_norm),
            state_update_norm=float(state_update_norm),
            stability_penalty=float(stability_penalty),
            convergence_penalty=float(convergence_penalty),
            max_abs_rate=float(np.max(np.abs(policy_rate))),
            max_abs_rate_change=float(np.max(np.abs(policy_rate - np.r_[0.0, policy_rate[:-1]]))),
            missing_direct_jacobians=self.missing_direct_jacobians,
            fallback_effects=tuple(
                state for state, source in self.effect_sources.items() if source != "direct_jacobian"
            ),
        )
        return ClosedLoopResult(loss=loss, diagnostics=diagnostics)

    def _counterfactual_state(self, *, base: pd.DataFrame, rate_deviation: np.ndarray) -> dict[str, np.ndarray]:
        periods = rate_deviation.size
        state: dict[str, np.ndarray] = {}
        for variable in (*LOSS_STATE_NAMES, *POLICY_STATE_NAMES):
            if variable in state:
                continue
            baseline = base[variable].to_numpy(dtype=float)
            effect = self._effects.get(variable)
            if effect is None:
                state[variable] = baseline.copy()
            else:
                state[variable] = baseline + effect[:periods, :periods] @ rate_deviation
        state["i"] = base["i"].to_numpy(dtype=float) + rate_deviation
        return state

    def _counterfactual_observations(
        self,
        *,
        base: pd.DataFrame,
        observations: pd.DataFrame,
        counterfactual_state: dict[str, np.ndarray],
    ) -> pd.DataFrame:
        result = observations[["period"]].copy()
        for state_name, observation_name in OBSERVATION_BY_STATE.items():
            baseline = base[state_name].to_numpy(dtype=float)
            observed = observations[observation_name].to_numpy(dtype=float)
            noise = observed - baseline
            result[observation_name] = counterfactual_state[state_name] + noise
        return result

    def _policy_rate_path(
        self,
        *,
        policy: LinearRule,
        information_state: str,
        counterfactual_state: dict[str, np.ndarray],
        counterfactual_observations: pd.DataFrame,
    ) -> np.ndarray:
        features = self._feature_matrix_for_state(
            information_state=information_state,
            feature_names=policy.spec.feature_names,
            counterfactual_state=counterfactual_state,
            counterfactual_observations=counterfactual_observations,
        )
        periods = features.shape[0]
        coefficients = np.asarray(policy.coefficients, dtype=float)
        deterministic = policy.intercept + features @ coefficients
        policy_rate = np.zeros(periods, dtype=float)
        lagged_rate = 0.0
        for period in range(periods):
            value = deterministic[period]
            if policy.spec.includes_lagged_rate:
                value += policy.lagged_rate_weight * lagged_rate
            policy_rate[period] = float(value)
            lagged_rate = float(value)
        return policy_rate

    def _feature_matrix_for_state(
        self,
        *,
        information_state: str,
        feature_names: tuple[str, ...],
        counterfactual_state: dict[str, np.ndarray],
        counterfactual_observations: pd.DataFrame,
    ) -> np.ndarray:
        features = _features_from_counterfactual(
            information_state=information_state,
            state=counterfactual_state,
            observations=counterfactual_observations,
            state_space_specs=self.state_space_specs,
        )
        missing = [name for name in feature_names if name not in features.columns]
        if missing:
            raise ValueError(f"Closed-loop features for {information_state} are missing {missing}.")
        return features.loc[:, list(feature_names)].to_numpy(dtype=float)

    def _loss(
        self,
        *,
        state: dict[str, np.ndarray],
        policy_rate: np.ndarray,
        converged: bool,
        rate_update_norm: float,
        state_update_norm: float,
    ) -> tuple[TrajectoryLoss, float, float]:
        periods = policy_rate.size
        rate_change = policy_rate - np.r_[0.0, policy_rate[:-1]]
        discounts = self.discount ** np.arange(periods)
        w = self.loss_weights
        inflation_loss = float(np.sum(discounts * w.inflation * state["pi"][:periods] ** 2))
        output_gap_loss = float(np.sum(discounts * w.output_gap * state["output_gap"][:periods] ** 2))
        consumption_loss = float(np.sum(discounts * w.consumption * state["C"][:periods] ** 2))
        rate_smoothing_loss = float(np.sum(discounts * w.rate_smoothing * rate_change**2))
        stability_penalty = self._stability_penalty(policy_rate, rate_change)
        convergence_penalty = 0.0
        if not converged:
            convergence_penalty = self.convergence_penalty_weight * (rate_update_norm**2 + state_update_norm**2)
        total = inflation_loss + output_gap_loss + consumption_loss + rate_smoothing_loss + stability_penalty
        return (
            TrajectoryLoss(
                total_loss=total,
                inflation_loss=inflation_loss,
                output_gap_loss=output_gap_loss,
                consumption_loss=consumption_loss,
                rate_smoothing_loss=rate_smoothing_loss,
                stability_penalty=stability_penalty,
            ),
            stability_penalty,
            convergence_penalty,
        )

    def _stability_penalty(self, policy_rate: np.ndarray, rate_change: np.ndarray) -> float:
        rate_excess = np.maximum(np.abs(policy_rate) - self.max_abs_rate, 0.0)
        change_excess = np.maximum(np.abs(rate_change) - self.max_abs_rate_change, 0.0)
        return float(self.stability_penalty_weight * (np.sum(rate_excess**2) + np.sum(change_excess**2)))


def _features_from_counterfactual(
    *,
    information_state: str,
    state: dict[str, np.ndarray],
    observations: pd.DataFrame,
    state_space_specs: dict[str, StateSpaceSpec],
) -> pd.DataFrame:
    periods = len(observations)
    if information_state == "aggregate_only":
        return pd.DataFrame({"pi_obs": observations["pi_obs"], "Y_obs": observations["Y_obs"]})
    if information_state == "aggregate_history":
        return pd.DataFrame(
            {
                "pi_obs": observations["pi_obs"],
                "Y_obs": observations["Y_obs"],
                "pi_obs_lag": observations["pi_obs"].shift(1).fillna(0.0),
                "Y_obs_lag": observations["Y_obs"].shift(1).fillna(0.0),
            }
        )
    if information_state == "observed_distribution":
        return pd.DataFrame(
            {
                "pi_obs": observations["pi_obs"],
                "Y_obs": observations["Y_obs"],
                "C_obs": observations["C_obs"],
                "mean_mpc_obs": observations["mean_mpc_centered_obs"],
                "low_liquidity_share_obs": observations["share_low_liquidity_centered_obs"],
                "interest_exposure_obs": observations["interest_exposure_centered_obs"],
            }
        )
    if information_state == "full_information":
        return pd.DataFrame(
            {
                "pi": state["pi"][:periods],
                "Y": state["Y"][:periods],
                "C": state["C"][:periods],
                "mean_mpc": state["mean_mpc_centered"][:periods],
                "low_liquidity_share": state["share_low_liquidity_centered"][:periods],
                "interest_exposure": state["interest_exposure_centered"][:periods],
            }
        )

    filter_state = "filtered_aggregates" if information_state == "filtered_aggregates" else "filtered_distribution"
    spec = state_space_specs[filter_state]
    result = run_kalman_filter(observations[list(spec.observation_names)].to_numpy(dtype=float), spec)
    frame = pd.DataFrame(
        {
            FILTERED_OUTPUT_BY_STATE[state_name]: result.means[:, index]
            for index, state_name in enumerate(spec.state_names)
        }
    )
    if information_state == "filtered_aggregates":
        return frame[["E_pi", "E_Y", "E_C"]]
    if information_state == "filtered_distribution":
        return frame[["E_pi", "E_Y", "E_C", "E_mean_mpc", "E_low_liquidity_share", "E_interest_exposure"]]
    if information_state == "filtered_distribution_mpc":
        return frame[["E_pi", "E_Y", "E_C", "E_mean_mpc"]]
    if information_state == "filtered_distribution_liquidity":
        return frame[["E_pi", "E_Y", "E_C", "E_low_liquidity_share"]]
    if information_state == "filtered_distribution_exposure":
        return frame[["E_pi", "E_Y", "E_C", "E_interest_exposure"]]
    raise ValueError(f"Unknown information state for closed-loop evaluation: {information_state}")


def _rate_path_effects(
    *,
    jacobians: dict[str, np.ndarray],
    observables: pd.DataFrame,
    periods: int,
    ridge: float,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    rate_key = "J_monetary_policy_shock_i"
    if rate_key not in jacobians:
        raise ValueError(f"Jacobian bundle does not contain {rate_key}.")
    rate_response = jacobians[rate_key][:periods, :periods]
    normal = rate_response.T @ rate_response + float(ridge) * np.eye(periods)
    shock_from_rate = np.linalg.solve(normal, rate_response.T)

    effects: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    direct_variables = ("pi", "Y", "output_gap", "C")
    for variable in direct_variables:
        key = f"J_monetary_policy_shock_{variable}"
        if key not in jacobians:
            raise ValueError(f"Jacobian bundle does not contain {key}.")
        effects[variable] = jacobians[key][:periods, :periods] @ shock_from_rate
        sources[variable] = "direct_jacobian"

    fallback = _distributional_fallback_effects(observables=observables, aggregate_effects=effects, periods=periods, ridge=ridge)
    for variable, effect in fallback.items():
        direct_key = f"J_monetary_policy_shock_{variable}"
        if direct_key in jacobians:
            effects[variable] = jacobians[direct_key][:periods, :periods] @ shock_from_rate
            sources[variable] = "direct_jacobian"
        else:
            effects[variable] = effect
            sources[variable] = "aggregate_regression_fallback"
    return effects, sources


def _distributional_fallback_effects(
    *,
    observables: pd.DataFrame,
    aggregate_effects: dict[str, np.ndarray],
    periods: int,
    ridge: float,
) -> dict[str, np.ndarray]:
    aggregate_names = ("pi", "Y", "C", "output_gap")
    x = observables[list(aggregate_names)].to_numpy(dtype=float)
    design = np.column_stack([np.ones(x.shape[0]), x])
    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    effects: dict[str, np.ndarray] = {}
    aggregate_stack = np.stack([aggregate_effects[name][:periods, :periods] for name in aggregate_names], axis=0)
    for variable in DISTRIBUTIONAL_STATE_NAMES:
        y = observables[variable].to_numpy(dtype=float)
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        effects[variable] = np.tensordot(beta[1:], aggregate_stack, axes=(0, 0))
    return effects


def _load_state_space_specs(path: Path) -> dict[str, StateSpaceSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    specs: dict[str, StateSpaceSpec] = {}
    for name, raw in payload["filters"].items():
        specs[name] = StateSpaceSpec(
            state_names=tuple(raw["state_names"]),
            observation_names=tuple(raw["observation_names"]),
            A=np.asarray(raw["A"], dtype=float),
            Q=np.asarray(raw["Q"], dtype=float),
            M=np.asarray(raw["M"], dtype=float),
            R=np.asarray(raw["R"], dtype=float),
            initial_mean=np.asarray(raw["initial_mean"], dtype=float),
            initial_cov=np.asarray(raw["initial_cov"], dtype=float),
        )
    return specs


def _observable_paths(observables: pd.DataFrame) -> dict[tuple[str], pd.DataFrame]:
    required = {"scenario", "period", "pi", "Y", "output_gap", "C", "i", *DISTRIBUTIONAL_STATE_NAMES}
    missing = required.difference(observables.columns)
    if missing:
        raise ValueError(f"HANK/SSJ observable table is missing columns: {sorted(missing)}")
    columns = ["period", "pi", "Y", "output_gap", "C", "i", *DISTRIBUTIONAL_STATE_NAMES]
    return {
        (str(scenario),): frame[columns].sort_values("period").reset_index(drop=True)
        for scenario, frame in observables.groupby("scenario", sort=False)
    }


def _observation_paths(observations: pd.DataFrame) -> dict[tuple[str, int], pd.DataFrame]:
    required = {"scenario", "period", "observation_seed", *OBSERVATION_BY_STATE.values()}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"HANK/SSJ observation table is missing columns: {sorted(missing)}")
    columns = ["period", *OBSERVATION_BY_STATE.values()]
    return {
        (str(scenario), int(seed)): frame[columns].sort_values("period").reset_index(drop=True)
        for (scenario, seed), frame in observations.groupby(["scenario", "observation_seed"], sort=False)
    }


def _state_distance(left: dict[str, np.ndarray], right: dict[str, np.ndarray], *, periods: int) -> float:
    names = tuple(DEFAULT_STATE_NAMES)
    total = 0.0
    for name in names:
        total += float(np.mean((left[name][:periods] - right[name][:periods]) ** 2))
    return float(np.sqrt(total / len(names)))


def diagnostics_to_row(diagnostics: ClosedLoopDiagnostics) -> dict[str, object]:
    row = asdict(diagnostics)
    row["missing_direct_jacobians"] = ",".join(diagnostics.missing_direct_jacobians)
    row["fallback_effects"] = ",".join(diagnostics.fallback_effects)
    return row
