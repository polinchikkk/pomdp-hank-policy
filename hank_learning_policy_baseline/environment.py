from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hank_partial_info_baseline.config import HANKPartialInfoConfig
from hank_partial_info_baseline.state_space import ReducedHANKStateSpaceModel

from .config import Stage4Config, TrainingVariant


@dataclass(frozen=True)
class RLPolicyScenarioSpec:
    variant_name: str
    scenario_name: str
    scenario_label: str
    input_mode: str
    include_distributional_state: bool
    noisy_observations: tuple[str, ...]
    noise_scale: float
    horizon: int
    gamma: float
    lambda_y: float
    lambda_i: float
    action_bound: float
    rate_bounds: tuple[float, float]
    innovation_scale: float
    distributional_state_shock_scale: float
    description: str

    def to_dict(self) -> dict:
        return {
            "variant_name": self.variant_name,
            "scenario_name": self.scenario_name,
            "scenario_label": self.scenario_label,
            "input_mode": self.input_mode,
            "include_distributional_state": self.include_distributional_state,
            "noisy_observations": list(self.noisy_observations),
            "noise_scale": self.noise_scale,
            "horizon": self.horizon,
            "gamma": self.gamma,
            "lambda_y": self.lambda_y,
            "lambda_i": self.lambda_i,
            "action_bound": self.action_bound,
            "rate_bounds": list(self.rate_bounds),
            "innovation_scale": self.innovation_scale,
            "distributional_state_shock_scale": self.distributional_state_shock_scale,
            "description": self.description,
        }


def build_scenario_spec(
    stage4_config: Stage4Config,
    variant: TrainingVariant,
) -> RLPolicyScenarioSpec:
    partial_spec = {
        spec["name"]: spec
        for spec in stage4_config.partial_config.scenario_specs()
    }[variant.scenario_name]
    rate_bound = 2.5 * stage4_config.action_bound
    return RLPolicyScenarioSpec(
        variant_name=variant.name,
        scenario_name=variant.scenario_name,
        scenario_label=variant.scenario_label,
        input_mode=variant.input_mode,
        include_distributional_state=variant.include_distributional_state,
        noisy_observations=tuple(partial_spec["noisy_observations"]),
        noise_scale=float(partial_spec["noise_scale"]),
        horizon=stage4_config.horizon,
        gamma=stage4_config.gamma,
        lambda_y=stage4_config.lambda_y,
        lambda_i=stage4_config.lambda_i,
        action_bound=stage4_config.action_bound,
        rate_bounds=(-rate_bound, rate_bound),
        innovation_scale=variant.innovation_scale,
        distributional_state_shock_scale=variant.distributional_state_shock_scale,
        description=variant.description,
    )


class HANKPolicyEnvironment:
    def __init__(
        self,
        *,
        reduced_model: ReducedHANKStateSpaceModel,
        partial_config: HANKPartialInfoConfig,
        scenario_spec: RLPolicyScenarioSpec,
        phi_pi: float,
        phi_y: float,
        rho_i: float,
    ) -> None:
        self.reduced_model = reduced_model
        self.partial_config = partial_config
        self.scenario_spec = scenario_spec
        self.phi_pi = float(phi_pi)
        self.phi_y = float(phi_y)
        self.rho_i = float(rho_i)

        self.state_dim = len(reduced_model.state_names)
        self.state_names = reduced_model.state_names
        self.state_index_map = {name: idx for idx, name in enumerate(self.state_names)}
        self.selected_state_indices = (
            tuple(range(self.state_dim))
            if scenario_spec.include_distributional_state
            else tuple(range(self.state_dim - 2))
        )
        self.observation_indices = tuple(
            reduced_model.observation_index(name)
            for name in scenario_spec.noisy_observations
        )
        self.observation_matrix = reduced_model.observation_matrix[list(self.observation_indices)]
        self.measurement_noise_std = {
            name: partial_config.base_measurement_noise()[name] * scenario_spec.noise_scale
            for name in scenario_spec.noisy_observations
        }
        self.measurement_cov = np.diag(
            [self.measurement_noise_std[name] ** 2 for name in scenario_spec.noisy_observations]
        ).astype(float)
        self.identity = np.eye(self.state_dim, dtype=float)
        self.steady_covariance = reduced_model.stationary_state_covariance()

        self._innovations = None
        self._measurement_noise = None
        self.t = 0
        self.prev_rate = 0.0
        self.prev_total_control = 0.0
        self.true_state = np.zeros((self.state_dim,), dtype=float)
        self.filtered_mean = np.zeros((self.state_dim,), dtype=float)
        self.filtered_covariance = self.steady_covariance.copy()
        self.current_observations = np.zeros((len(self.observation_indices),), dtype=float)
        self.current_context: dict[str, object] = {}

    def _rule_rate(self, state: np.ndarray, prev_rate: float) -> float:
        idx_rstar = self.state_index_map["rstar_gap"]
        idx_pi = self.state_index_map["inflation_gap"]
        idx_output = self.state_index_map["output_gap"]
        return float(
            self.rho_i * prev_rate
            + (1.0 - self.rho_i)
            * (state[idx_rstar] + self.phi_pi * state[idx_pi] + self.phi_y * state[idx_output])
        )

    def _kalman_update(
        self,
        *,
        predicted_mean: np.ndarray,
        predicted_covariance: np.ndarray,
        observation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        innovation = observation - self.observation_matrix @ predicted_mean
        innovation_covariance = self.observation_matrix @ predicted_covariance @ self.observation_matrix.T + self.measurement_cov
        innovation_covariance = 0.5 * (innovation_covariance + innovation_covariance.T)
        jitter = 1.0e-12
        sign, _ = np.linalg.slogdet(innovation_covariance)
        while sign <= 0:
            innovation_covariance = innovation_covariance + jitter * np.eye(len(self.observation_indices), dtype=float)
            jitter *= 10.0
            sign, _ = np.linalg.slogdet(innovation_covariance)
            if jitter > 1.0:
                raise RuntimeError("Innovation covariance is not positive definite.")
        innovation_precision = np.linalg.inv(innovation_covariance)
        kalman_gain = predicted_covariance @ self.observation_matrix.T @ innovation_precision
        filtered_mean = predicted_mean + kalman_gain @ innovation
        filtered_covariance = (
            (self.identity - kalman_gain @ self.observation_matrix)
            @ predicted_covariance
            @ (self.identity - kalman_gain @ self.observation_matrix).T
            + kalman_gain @ self.measurement_cov @ kalman_gain.T
        )
        filtered_covariance = 0.5 * (filtered_covariance + filtered_covariance.T)
        return filtered_mean, filtered_covariance

    def _sample_innovations(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        horizon = self.scenario_spec.horizon
        innovations = np.zeros((horizon, self.state_dim), dtype=float)
        macro_scale = float(self.scenario_spec.innovation_scale)
        innovations[:, self.state_index_map["rstar_gap"]] = rng.normal(
            scale=self.partial_config.rstar_std * macro_scale,
            size=horizon,
        )
        innovations[:, self.state_index_map["productivity_gap"]] = rng.normal(
            scale=self.partial_config.productivity_std * macro_scale,
            size=horizon,
        )
        innovations[:, self.state_index_map["fiscal_gap"]] = rng.normal(
            scale=self.partial_config.fiscal_std * macro_scale,
            size=horizon,
        )
        distributional_scale = float(self.scenario_spec.distributional_state_shock_scale)
        if distributional_scale > 0.0:
            innovations[:, self.state_index_map["low_liquidity_gap"]] = rng.normal(
                scale=distributional_scale,
                size=horizon,
            )
            innovations[:, self.state_index_map["mean_mpc_gap"]] = rng.normal(
                scale=distributional_scale,
                size=horizon,
            )
        measurement_noise = np.column_stack([
            rng.normal(scale=self.measurement_noise_std[name], size=horizon)
            for name in self.scenario_spec.noisy_observations
        ])
        return innovations, measurement_noise

    def _build_observation(self) -> np.ndarray:
        if self.scenario_spec.input_mode == "filtered_state":
            state = self.filtered_mean[list(self.selected_state_indices)]
            return np.concatenate([state, np.array([self.prev_rate], dtype=float)])
        if self.scenario_spec.input_mode == "filtered_state_uncertainty":
            state = self.filtered_mean[list(self.selected_state_indices)]
            diag_cov = np.diag(self.filtered_covariance)[list(self.selected_state_indices)]
            return np.concatenate([state, diag_cov, np.array([self.prev_rate], dtype=float)])
        if self.scenario_spec.input_mode == "raw_observations":
            return np.concatenate([self.current_observations, np.array([self.prev_rate], dtype=float)])
        raise ValueError(f"Unsupported input mode: {self.scenario_spec.input_mode}")

    def _context(self, *, policy_rate: float | None = None, residual_action: float | None = None, total_policy_shock: float | None = None) -> dict:
        filtered_rule_rate = self._rule_rate(self.filtered_mean, self.prev_rate)
        full_information_rate = self._rule_rate(self.true_state, self.prev_rate)
        if policy_rate is None:
            policy_rate = self.prev_rate
        if residual_action is None:
            residual_action = float(policy_rate - filtered_rule_rate)
        if total_policy_shock is None:
            total_policy_shock = float(policy_rate - full_information_rate)
        return {
            "t": int(self.t),
            "variant_name": self.scenario_spec.variant_name,
            "scenario_name": self.scenario_spec.scenario_name,
            "scenario_label": self.scenario_spec.scenario_label,
            "input_mode": self.scenario_spec.input_mode,
            "include_distributional_state": self.scenario_spec.include_distributional_state,
            "true_state": self.true_state.copy(),
            "filtered_state": self.filtered_mean.copy(),
            "filtered_covariance": self.filtered_covariance.copy(),
            "current_observations": self.current_observations.copy(),
            "current_rate": float(self.prev_rate),
            "policy_rate": float(policy_rate),
            "filtered_rule_rate": float(filtered_rule_rate),
            "full_information_rate": float(full_information_rate),
            "residual_action": float(residual_action),
            "policy_shock": float(total_policy_shock),
            "rate_bounds": self.scenario_spec.rate_bounds,
            "noisy_observation_names": self.scenario_spec.noisy_observations,
            "measurement_noise_std": self.measurement_noise_std,
        }

    def reset(self, seed: int) -> tuple[np.ndarray, dict]:
        self._innovations, self._measurement_noise = self._sample_innovations(seed)
        self.t = 0
        self.prev_rate = 0.0
        self.prev_total_control = 0.0
        self.true_state = self._innovations[0].copy()
        self.filtered_mean = np.zeros((self.state_dim,), dtype=float)
        self.filtered_covariance = self.steady_covariance.copy()
        self.current_observations = self.observation_matrix @ self.true_state + self._measurement_noise[0]
        self.filtered_mean, self.filtered_covariance = self._kalman_update(
            predicted_mean=np.zeros((self.state_dim,), dtype=float),
            predicted_covariance=self.steady_covariance.copy(),
            observation=self.current_observations,
        )
        self.current_context = self._context(policy_rate=self.prev_rate, residual_action=0.0, total_policy_shock=0.0)
        return self._build_observation(), self.current_context.copy()

    def step_rate(self, policy_rate: float) -> tuple[np.ndarray, float, bool, dict]:
        policy_rate = float(np.clip(policy_rate, self.scenario_spec.rate_bounds[0], self.scenario_spec.rate_bounds[1]))
        filtered_rule_rate = self._rule_rate(self.filtered_mean, self.prev_rate)
        full_information_rate = self._rule_rate(self.true_state, self.prev_rate)
        residual_action = float(policy_rate - filtered_rule_rate)
        total_policy_shock = float(policy_rate - full_information_rate)

        idx_pi = self.state_index_map["inflation_gap"]
        idx_output = self.state_index_map["output_gap"]
        inflation = float(self.true_state[idx_pi])
        output_gap = float(self.true_state[idx_output])
        rate_change = float(policy_rate - self.prev_rate)
        inflation_loss = inflation**2
        output_gap_loss = self.scenario_spec.lambda_y * output_gap**2
        rate_change_loss = self.scenario_spec.lambda_i * rate_change**2
        loss = inflation_loss + output_gap_loss + rate_change_loss
        reward = -loss

        step_info = self._context(
            policy_rate=policy_rate,
            residual_action=residual_action,
            total_policy_shock=total_policy_shock,
        )
        step_info.update({
            "inflation_loss": float(inflation_loss),
            "output_gap_loss": float(output_gap_loss),
            "rate_change_loss": float(rate_change_loss),
            "loss": float(loss),
            "reward": float(reward),
        })

        done = self.t >= self.scenario_spec.horizon - 1
        if done:
            self.current_context = step_info.copy()
            return self._build_observation(), float(reward), True, step_info

        next_t = self.t + 1
        next_true_state = (
            self.reduced_model.transition_matrix @ self.true_state
            + self.reduced_model.control_loadings * total_policy_shock
            + self._innovations[next_t]
        )
        predicted_mean = self.reduced_model.transition_matrix @ self.filtered_mean + self.reduced_model.control_loadings * total_policy_shock
        predicted_covariance = (
            self.reduced_model.transition_matrix @ self.filtered_covariance @ self.reduced_model.transition_matrix.T
            + self.reduced_model.process_noise_cov
        )
        predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)
        next_observations = self.observation_matrix @ next_true_state + self._measurement_noise[next_t]
        filtered_mean, filtered_covariance = self._kalman_update(
            predicted_mean=predicted_mean,
            predicted_covariance=predicted_covariance,
            observation=next_observations,
        )

        self.t = next_t
        self.prev_rate = policy_rate
        self.prev_total_control = total_policy_shock
        self.true_state = next_true_state
        self.filtered_mean = filtered_mean
        self.filtered_covariance = filtered_covariance
        self.current_observations = next_observations
        self.current_context = self._context(policy_rate=self.prev_rate, residual_action=0.0, total_policy_shock=self.prev_total_control)
        return self._build_observation(), float(reward), False, step_info
