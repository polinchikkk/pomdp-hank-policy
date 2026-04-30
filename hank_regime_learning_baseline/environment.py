from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regime_switching_baseline.regime_filter import _imm_update_step
from regime_switching_baseline.regime_model import RegimeSwitchingConfig, RegimeSwitchingModel
from regime_switching_baseline.regime_simulation import simulate_hidden_regimes

from .config import RegimeLearningConfig, RegimeLearningVariant


@dataclass(frozen=True)
class RegimeRLScenarioSpec:
    variant_name: str
    scenario_name: str
    scenario_label: str
    input_mode: str
    include_distributional_state: bool
    noisy_observations: tuple[str, ...]
    noise_scale: float
    gap_name: str
    gap_label: str
    gap_scale: float
    horizon: int
    gamma: float
    lambda_y: float
    lambda_i: float
    action_bound: float
    rate_bounds: tuple[float, float]
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
            "gap_name": self.gap_name,
            "gap_label": self.gap_label,
            "gap_scale": self.gap_scale,
            "horizon": self.horizon,
            "gamma": self.gamma,
            "lambda_y": self.lambda_y,
            "lambda_i": self.lambda_i,
            "action_bound": self.action_bound,
            "rate_bounds": list(self.rate_bounds),
            "description": self.description,
        }


def build_scenario_spec(
    config: RegimeLearningConfig,
    variant: RegimeLearningVariant,
) -> RegimeRLScenarioSpec:
    scenario_lookup = {}
    for spec in config.regime_config.scenario_specs():
        scenario_lookup[spec["name"]] = spec
    for spec in config.regime_config.article_scenario_specs():
        scenario_lookup[spec["name"]] = spec
    scenario = scenario_lookup[variant.scenario_name]
    rate_bound = 2.5 * config.action_bound
    return RegimeRLScenarioSpec(
        variant_name=variant.name,
        scenario_name=scenario["name"],
        scenario_label=scenario["label"],
        input_mode=variant.input_mode,
        include_distributional_state=variant.include_distributional_state,
        noisy_observations=tuple(scenario["noisy_observations"]),
        noise_scale=float(scenario["noise_scale"]),
        gap_name=scenario["gap_name"],
        gap_label=scenario["gap_label"],
        gap_scale=float(scenario["gap_scale"]),
        horizon=config.horizon,
        gamma=config.gamma,
        lambda_y=config.lambda_y,
        lambda_i=config.lambda_i,
        action_bound=config.action_bound,
        rate_bounds=(-rate_bound, rate_bound),
        description=variant.description or scenario["description"],
    )


class RegimeSwitchingPolicyEnvironment:
    def __init__(
        self,
        *,
        model: RegimeSwitchingModel,
        regime_config: RegimeSwitchingConfig,
        scenario_spec: RegimeRLScenarioSpec,
        phi_pi: float,
        phi_y: float,
        rho_i: float,
    ) -> None:
        self.model = model
        self.regime_config = regime_config
        self.scenario_spec = scenario_spec
        self.phi_pi = float(phi_pi)
        self.phi_y = float(phi_y)
        self.rho_i = float(rho_i)

        self.state_dim = len(model.state_names)
        self.state_names = model.state_names
        self.state_index_map = {name: idx for idx, name in enumerate(self.state_names)}
        self.taylor_state_indices = tuple(
            self.state_index_map[name]
            for name in ("rstar_gap", "inflation_gap", "output_gap")
        )
        self.macro_state_indices = tuple(
            self.state_index_map[name]
            for name in ("rstar_gap", "productivity_gap", "fiscal_gap", "inflation_gap", "output_gap")
        )
        self.extended_state_indices = tuple(
            self.state_index_map[name]
            for name in ("rstar_gap", "productivity_gap", "fiscal_gap", "inflation_gap", "output_gap")
        )
        self.distribution_state_indices = tuple(
            self.state_index_map[name]
            for name in ("low_liquidity_gap", "mean_mpc_gap")
        )
        self.selected_state_indices = (
            tuple(range(self.state_dim))
            if scenario_spec.include_distributional_state
            else tuple(range(self.state_dim - 2))
        )
        self.observation_indices = tuple(model.observation_index(name) for name in scenario_spec.noisy_observations)
        self.observation_matrix = model.observation_matrix[list(self.observation_indices)]
        self.measurement_noise_std = {
            name: regime_config.partial_config.base_measurement_noise()[name] * scenario_spec.noise_scale
            for name in scenario_spec.noisy_observations
        }
        self.measurement_covariance = np.diag(
            [self.measurement_noise_std[name] ** 2 for name in scenario_spec.noisy_observations]
        ).astype(float)
        self.mode_covariances_ss = model.stationary_state_covariances()

        self._hidden_regimes: np.ndarray | None = None
        self._innovations: np.ndarray | None = None
        self._measurement_noise: np.ndarray | None = None

        self.t = 0
        self.prev_rate = 0.0
        self.prev_total_control = 0.0
        self.current_hidden_regime = 0
        self.true_state = np.zeros((self.state_dim,), dtype=float)
        self.filtered_mean = np.zeros((self.state_dim,), dtype=float)
        self.filtered_covariance = np.eye(self.state_dim, dtype=float)
        self.filtered_mode_probabilities = self.model.stationary_regime_distribution()
        self.regime_conditioned_means = np.repeat(self.model.initial_state_mean()[None, :], self.model.num_regimes(), axis=0)
        self.regime_conditioned_covariances = self.mode_covariances_ss.copy()
        self.misspecified_mean = np.zeros((self.state_dim,), dtype=float)
        self.misspecified_covariance = self.mode_covariances_ss[0].copy()
        self.current_observations = np.zeros((len(self.observation_indices),), dtype=float)
        self.previous_observations = np.zeros((len(self.observation_indices),), dtype=float)
        self.current_context: dict[str, object] = {}

    @staticmethod
    def _binary_entropy(probability: float) -> float:
        clipped = float(np.clip(probability, 1.0e-10, 1.0 - 1.0e-10))
        return float(-(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)))

    def _rule_rate(self, state: np.ndarray, prev_rate: float) -> float:
        idx_rstar = self.state_index_map["rstar_gap"]
        idx_pi = self.state_index_map["inflation_gap"]
        idx_output = self.state_index_map["output_gap"]
        return float(
            self.rho_i * prev_rate
            + (1.0 - self.rho_i) * (state[idx_rstar] + self.phi_pi * state[idx_pi] + self.phi_y * state[idx_output])
        )

    def _sample_episode(self, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_regimes = simulate_hidden_regimes(
            self.model,
            horizon=self.scenario_spec.horizon,
            seed=seed,
        )
        rng = np.random.default_rng(seed + 1)
        innovations = np.zeros((self.scenario_spec.horizon, self.state_dim), dtype=float)
        for period, regime in enumerate(hidden_regimes):
            innovations[period] = rng.multivariate_normal(
                mean=np.zeros(self.state_dim, dtype=float),
                cov=self.model.process_noise_covariances[regime],
            )
        measurement_noise = np.column_stack([
            rng.normal(scale=self.measurement_noise_std[name], size=self.scenario_spec.horizon)
            for name in self.scenario_spec.noisy_observations
        ])
        return hidden_regimes, innovations, measurement_noise

    def _single_regime_kalman_update(
        self,
        *,
        predicted_mean: np.ndarray,
        predicted_covariance: np.ndarray,
        observation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        identity = np.eye(self.state_dim, dtype=float)
        innovation = observation - self.observation_matrix @ predicted_mean
        innovation_covariance = self.observation_matrix @ predicted_covariance @ self.observation_matrix.T + self.measurement_covariance
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
            (identity - kalman_gain @ self.observation_matrix)
            @ predicted_covariance
            @ (identity - kalman_gain @ self.observation_matrix).T
            + kalman_gain @ self.measurement_covariance @ kalman_gain.T
        )
        filtered_covariance = 0.5 * (filtered_covariance + filtered_covariance.T)
        return filtered_mean, filtered_covariance

    def _build_observation(self) -> np.ndarray:
        p_stress = float(self.filtered_mode_probabilities[1])
        regime_entropy = self._binary_entropy(p_stress)
        variance_diag = np.diag(self.filtered_covariance)
        variance_trace = float(np.trace(self.filtered_covariance))
        macro_state = self.filtered_mean[list(self.macro_state_indices)]
        distribution_state = self.filtered_mean[list(self.distribution_state_indices)]
        if self.scenario_spec.input_mode == "belief_state":
            state = self.filtered_mean[list(self.selected_state_indices)]
            return np.concatenate([state, np.array([p_stress, self.prev_rate], dtype=float)])
        if self.scenario_spec.input_mode == "filtered_taylor_state":
            state = self.filtered_mean[list(self.taylor_state_indices)]
            return np.concatenate([state, np.array([self.prev_rate], dtype=float)])
        if self.scenario_spec.input_mode == "filtered_extended_state":
            state = self.filtered_mean[list(self.extended_state_indices)]
            return np.concatenate([state, np.array([p_stress, self.prev_rate], dtype=float)])
        if self.scenario_spec.input_mode == "belief_state_uncertainty":
            state = self.filtered_mean[list(self.selected_state_indices)]
            diag_cov = np.diag(self.filtered_covariance)[list(self.selected_state_indices)]
            return np.concatenate([state, diag_cov, np.array([p_stress, self.prev_rate], dtype=float)])
        if self.scenario_spec.input_mode == "observed_information_state":
            return np.concatenate(
                [self.current_observations, self.previous_observations, np.array([self.prev_rate], dtype=float)]
            )
        if self.scenario_spec.input_mode == "posterior_mean_state":
            return np.concatenate([macro_state, np.array([self.prev_rate], dtype=float)])
        if self.scenario_spec.input_mode == "posterior_regime_state":
            return np.concatenate([macro_state, np.array([p_stress, self.prev_rate], dtype=float)])
        if self.scenario_spec.input_mode == "posterior_uncertainty_state":
            return np.concatenate(
                [macro_state, np.array([p_stress, regime_entropy, variance_trace, self.prev_rate], dtype=float)]
            )
        if self.scenario_spec.input_mode == "posterior_distribution_state":
            return np.concatenate(
                [
                    macro_state,
                    np.array([p_stress, regime_entropy, variance_trace], dtype=float),
                    distribution_state,
                    np.array([self.prev_rate], dtype=float),
                ]
            )
        if self.scenario_spec.input_mode == "raw_observations":
            return np.concatenate([self.current_observations, np.array([self.prev_rate], dtype=float)])
        raise ValueError(f"Unsupported input mode: {self.scenario_spec.input_mode}")

    def _context(
        self,
        *,
        policy_rate: float | None = None,
        residual_action: float | None = None,
        total_policy_shock: float | None = None,
    ) -> dict:
        filtered_rule_rate = self._rule_rate(self.filtered_mean, self.prev_rate)
        misspecified_filtered_rule_rate = self._rule_rate(self.misspecified_mean, self.prev_rate)
        full_information_rate = self._rule_rate(self.true_state, self.prev_rate)
        stress_probability = float(self.filtered_mode_probabilities[1])
        stress_entropy = self._binary_entropy(stress_probability)
        filtered_variance_diag = np.diag(self.filtered_covariance)
        filtered_variance_trace = float(np.trace(self.filtered_covariance))
        filtered_macro_variance_trace = float(np.sum(filtered_variance_diag[list(self.macro_state_indices)]))
        normal_mean = self.regime_conditioned_means[0].copy()
        stress_mean = self.regime_conditioned_means[1].copy()
        regime_mean_delta = stress_mean - normal_mean
        stress_interaction = stress_probability * regime_mean_delta
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
            "state_names": tuple(self.state_names),
            "input_mode": self.scenario_spec.input_mode,
            "include_distributional_state": self.scenario_spec.include_distributional_state,
            "hidden_regime": int(self.current_hidden_regime),
            "stress_probability": stress_probability,
            "stress_entropy": stress_entropy,
            "true_state": self.true_state.copy(),
            "filtered_state": self.filtered_mean.copy(),
            "filtered_covariance": self.filtered_covariance.copy(),
            "filtered_variance_diag": filtered_variance_diag.copy(),
            "filtered_variance_trace": filtered_variance_trace,
            "filtered_macro_variance_trace": filtered_macro_variance_trace,
            "filtered_mode_probabilities": self.filtered_mode_probabilities.copy(),
            "regime_conditioned_means": self.regime_conditioned_means.copy(),
            "regime_conditioned_covariances": self.regime_conditioned_covariances.copy(),
            "normal_regime_state_mean": normal_mean,
            "stress_regime_state_mean": stress_mean,
            "regime_mean_delta": regime_mean_delta,
            "stress_interaction_state": stress_interaction,
            "misspecified_filtered_state": self.misspecified_mean.copy(),
            "misspecified_filtered_covariance": self.misspecified_covariance.copy(),
            "current_observations": self.current_observations.copy(),
            "lagged_observations": self.previous_observations.copy(),
            "current_rate": float(self.prev_rate),
            "policy_rate": float(policy_rate),
            "filtered_rule_rate": float(filtered_rule_rate),
            "misspecified_filtered_rule_rate": float(misspecified_filtered_rule_rate),
            "full_information_rate": float(full_information_rate),
            "residual_action": float(residual_action),
            "policy_shock": float(total_policy_shock),
            "rate_bounds": self.scenario_spec.rate_bounds,
            "noisy_observation_names": self.scenario_spec.noisy_observations,
            "measurement_noise_std": self.measurement_noise_std,
        }

    def reset(self, seed: int) -> tuple[np.ndarray, dict]:
        self._hidden_regimes, self._innovations, self._measurement_noise = self._sample_episode(seed)
        self.t = 0
        self.prev_rate = 0.0
        self.prev_total_control = 0.0
        self.current_hidden_regime = int(self._hidden_regimes[0])
        self.true_state = self._innovations[0].copy()
        self.current_observations = self.observation_matrix @ self.true_state + self._measurement_noise[0]
        self.previous_observations = np.zeros_like(self.current_observations)

        previous_mode_probabilities = self.model.stationary_regime_distribution()
        previous_mode_means = np.repeat(self.model.initial_state_mean()[None, :], self.model.num_regimes(), axis=0)
        previous_mode_covariances = self.mode_covariances_ss.copy()
        (
            _predicted_mode_probabilities,
            filtered_mode_probabilities,
            _predicted_means,
            _predicted_covariances,
            regime_conditioned_means,
            regime_conditioned_covariances,
            filtered_mean,
            filtered_covariance,
            _innovations,
            _innovation_covariances,
            _increment,
        ) = _imm_update_step(
            model=self.model,
            previous_mode_probabilities=previous_mode_probabilities,
            previous_mode_means=previous_mode_means,
            previous_mode_covariances=previous_mode_covariances,
            observation=self.current_observations,
            observation_matrix=self.observation_matrix,
            measurement_covariance=self.measurement_covariance,
            previous_control=0.0,
        )
        self.filtered_mean = filtered_mean
        self.filtered_covariance = filtered_covariance
        self.filtered_mode_probabilities = filtered_mode_probabilities
        self.regime_conditioned_means = regime_conditioned_means
        self.regime_conditioned_covariances = regime_conditioned_covariances
        self.misspecified_mean, self.misspecified_covariance = self._single_regime_kalman_update(
            predicted_mean=np.zeros((self.state_dim,), dtype=float),
            predicted_covariance=self.mode_covariances_ss[0].copy(),
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

        assert self._hidden_regimes is not None
        assert self._innovations is not None
        assert self._measurement_noise is not None

        next_t = self.t + 1
        next_regime = int(self._hidden_regimes[next_t])
        transition = self.model.transition_matrices[next_regime]
        control = self.model.control_loadings[next_regime]
        next_true_state = transition @ self.true_state + control * total_policy_shock + self._innovations[next_t]
        next_observations = self.observation_matrix @ next_true_state + self._measurement_noise[next_t]
        normal_transition = self.model.transition_matrices[0]
        normal_control = self.model.control_loadings[0]
        misspecified_predicted_mean = normal_transition @ self.misspecified_mean + normal_control * total_policy_shock
        misspecified_predicted_covariance = (
            normal_transition @ self.misspecified_covariance @ normal_transition.T
            + self.model.process_noise_covariances[0]
        )
        misspecified_predicted_covariance = 0.5 * (
            misspecified_predicted_covariance + misspecified_predicted_covariance.T
        )

        (
            _predicted_mode_probabilities,
            filtered_mode_probabilities,
            _predicted_means,
            _predicted_covariances,
            regime_conditioned_means,
            regime_conditioned_covariances,
            filtered_mean,
            filtered_covariance,
            _innovations,
            _innovation_covariances,
            _increment,
        ) = _imm_update_step(
            model=self.model,
            previous_mode_probabilities=self.filtered_mode_probabilities,
            previous_mode_means=self.regime_conditioned_means,
            previous_mode_covariances=self.regime_conditioned_covariances,
            observation=next_observations,
            observation_matrix=self.observation_matrix,
            measurement_covariance=self.measurement_covariance,
            previous_control=total_policy_shock,
        )

        self.t = next_t
        self.prev_rate = policy_rate
        self.prev_total_control = total_policy_shock
        self.current_hidden_regime = next_regime
        self.true_state = next_true_state
        self.previous_observations = self.current_observations.copy()
        self.filtered_mean = filtered_mean
        self.filtered_covariance = filtered_covariance
        self.filtered_mode_probabilities = filtered_mode_probabilities
        self.regime_conditioned_means = regime_conditioned_means
        self.regime_conditioned_covariances = regime_conditioned_covariances
        self.misspecified_mean, self.misspecified_covariance = self._single_regime_kalman_update(
            predicted_mean=misspecified_predicted_mean,
            predicted_covariance=misspecified_predicted_covariance,
            observation=next_observations,
        )
        self.current_observations = next_observations
        self.current_context = self._context(
            policy_rate=self.prev_rate,
            residual_action=0.0,
            total_policy_shock=self.prev_total_control,
        )
        return self._build_observation(), float(reward), False, step_info
