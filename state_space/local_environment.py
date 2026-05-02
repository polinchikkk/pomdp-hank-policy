from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .filters import LinearFilterState, kalman_update


STATE_NAMES = (
    "inflation_gap",
    "output_gap",
    "natural_rate_gap",
    "mean_mpc",
    "low_liquidity_share",
)


@dataclass(frozen=True)
class LossWeights:
    inflation: float = 1.0
    output: float = 0.5
    rate_change: float = 0.05
    beta: float = 1.0


@dataclass(frozen=True)
class LocalEnvironmentConfig:
    horizon: int = 60
    aggregate_observation_noise: float = 1.0
    distributional_observation_noise: float = 1.0
    mpc_channel_strength: float = 1.0
    heterogeneity: str = "baseline"
    rate_bound: float = 0.06
    loss_weights: LossWeights = LossWeights()


@dataclass(frozen=True)
class SimulationResult:
    total_loss: float
    inflation_loss: float
    output_loss: float
    rate_loss: float
    features: np.ndarray | None = None
    rates: np.ndarray | None = None
    states: np.ndarray | None = None


def scenario_config(name: str, *, horizon: int = 60) -> LocalEnvironmentConfig:
    if name == "baseline":
        return LocalEnvironmentConfig(horizon=horizon)
    if name == "high_aggregate_noise":
        return LocalEnvironmentConfig(horizon=horizon, aggregate_observation_noise=2.0)
    if name == "high_heterogeneity":
        return LocalEnvironmentConfig(horizon=horizon, mpc_channel_strength=1.5, heterogeneity="high")
    if name == "noisy_distributional_data":
        return LocalEnvironmentConfig(horizon=horizon, distributional_observation_noise=2.0)
    raise ValueError(f"Unknown scenario: {name}")


class LocalHANKInformationEnvironment:
    """Локальная среда для измерения ценности распределительной информации."""

    def __init__(self, config: LocalEnvironmentConfig):
        self.config = config
        self.state_names = STATE_NAMES
        self.state_dim = len(STATE_NAMES)
        self.A = np.array(
            [
                [0.58, 0.10, 0.04, 0.00, 0.00],
                [0.08, 0.68, 0.18, 0.02, 0.03],
                [0.00, 0.00, 0.86, 0.00, 0.00],
                [0.00, 0.03, 0.00, 0.80, 0.08],
                [0.00, 0.04, 0.00, 0.12, 0.78],
            ],
            dtype=float,
        )
        heterogeneity_scale = 1.35 if config.heterogeneity == "high" else 1.0
        self.shock_std = np.array([0.0030, 0.0040, 0.0025, 0.0090, 0.0100], dtype=float)
        self.shock_std[3:] *= heterogeneity_scale
        self.process_covariance = np.diag(self.shock_std**2)
        self.initial_covariance = np.diag((4.0 * self.shock_std) ** 2)

    def simulate(
        self,
        *,
        policy,
        information_state: str,
        seed: int,
        collect_features: bool = False,
    ) -> SimulationResult:
        rng = np.random.default_rng(seed)
        shocks = rng.normal(size=(self.config.horizon, self.state_dim)) * self.shock_std
        aggregate_noise = rng.normal(size=(self.config.horizon, 2)) * self._aggregate_noise_std()
        distribution_noise = rng.normal(size=(self.config.horizon, 2)) * self._distribution_noise_std()

        state = rng.normal(scale=0.5 * self.shock_std)
        lagged_rate = 0.0
        aggregate_filter_state = LinearFilterState(mean=np.zeros(self.state_dim), covariance=self.initial_covariance.copy())
        distribution_filter_state = LinearFilterState(mean=np.zeros(self.state_dim), covariance=self.initial_covariance.copy())

        total_loss = 0.0
        inflation_loss = 0.0
        output_loss = 0.0
        rate_loss = 0.0
        discount = 1.0
        collected_features: list[list[float]] = []
        rates: list[float] = []
        states: list[np.ndarray] = []

        for period in range(self.config.horizon):
            observation = self._observation(state, aggregate_noise[period], distribution_noise[period])
            if information_state in {
                "filtered_aggregates",
                "distributional",
                "distributional_mpc",
                "distributional_liquidity",
            }:
                aggregate_filter_state = self._update_filter(
                    aggregate_filter_state,
                    "filtered_aggregates",
                    observation,
                )
            if information_state in {"distributional", "distributional_mpc", "distributional_liquidity"}:
                distribution_filter_state = self._update_filter(
                    distribution_filter_state,
                    "distributional",
                    observation,
                )

            features = self._features(
                information_state,
                state,
                observation,
                aggregate_filter_state,
                distribution_filter_state,
            )
            rate = float(policy.rate(features, lagged_rate))
            rate = float(np.clip(rate, -self.config.rate_bound, self.config.rate_bound))

            period_losses = self._period_losses(state, rate, lagged_rate)
            inflation_loss += discount * period_losses[0]
            output_loss += discount * period_losses[1]
            rate_loss += discount * period_losses[2]
            total_loss += discount * sum(period_losses)

            if collect_features:
                collected_features.append([features[name] for name in policy.spec.feature_names])
                rates.append(rate)
                states.append(state.copy())

            next_state = self.A @ state + self._policy_effect(rate, state) + shocks[period]
            if information_state in {
                "filtered_aggregates",
                "distributional",
                "distributional_mpc",
                "distributional_liquidity",
            }:
                aggregate_filter_state = self._predict_filter(aggregate_filter_state, rate)
            if information_state in {"distributional", "distributional_mpc", "distributional_liquidity"}:
                distribution_filter_state = self._predict_filter(distribution_filter_state, rate)

            state = next_state
            lagged_rate = rate
            discount *= self.config.loss_weights.beta

        return SimulationResult(
            total_loss=float(total_loss),
            inflation_loss=float(inflation_loss),
            output_loss=float(output_loss),
            rate_loss=float(rate_loss),
            features=np.asarray(collected_features, dtype=float) if collect_features else None,
            rates=np.asarray(rates, dtype=float) if collect_features else None,
            states=np.asarray(states, dtype=float) if collect_features else None,
        )

    def feature_scales(self, *, policy, information_state: str, seeds: list[int]) -> dict[str, float]:
        values = []
        for seed in seeds:
            result = self.simulate(
                policy=policy,
                information_state=information_state,
                seed=seed,
                collect_features=True,
            )
            if result.features is not None and result.features.size:
                values.append(result.features)
        if not values:
            return {name: 1.0 for name in policy.spec.feature_names}
        stacked = np.vstack(values)
        scales = np.std(stacked, axis=0)
        return {
            name: float(scale if scale > 1e-5 else 1.0)
            for name, scale in zip(policy.spec.feature_names, scales)
        }

    def _period_losses(self, state: np.ndarray, rate: float, lagged_rate: float) -> tuple[float, float, float]:
        weights = self.config.loss_weights
        inflation = weights.inflation * float(state[0] ** 2)
        output = weights.output * float(state[1] ** 2)
        rate_change = weights.rate_change * float((rate - lagged_rate) ** 2)
        return inflation, output, rate_change

    def _policy_effect(self, rate: float, state: np.ndarray) -> np.ndarray:
        transmission = self._transmission_strength(state)
        return np.array(
            [
                -0.18 * transmission * rate,
                -0.70 * transmission * rate,
                0.00,
                0.04 * rate,
                0.05 * rate,
            ],
            dtype=float,
        )

    def _predict_filter(self, filter_state: LinearFilterState, rate: float) -> LinearFilterState:
        mean = self.A @ filter_state.mean + self._policy_effect(rate, filter_state.mean)
        covariance = self.A @ filter_state.covariance @ self.A.T + self.process_covariance
        covariance = 0.5 * (covariance + covariance.T)
        return LinearFilterState(mean=mean, covariance=covariance)

    def _update_filter(
        self,
        filter_state: LinearFilterState,
        information_state: str,
        observation: dict[str, np.ndarray],
    ) -> LinearFilterState:
        if information_state == "filtered_aggregates":
            matrix = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
            obs = observation["aggregate"]
            cov = np.diag(self._aggregate_noise_std() ** 2)
        elif information_state == "distributional":
            matrix = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
            obs = np.concatenate([observation["aggregate"], observation["distributional"]])
            cov = np.diag(np.concatenate([self._aggregate_noise_std(), self._distribution_noise_std()]) ** 2)
        else:
            raise ValueError(f"Filtering is not defined for {information_state}.")
        return kalman_update(
            predicted_mean=filter_state.mean,
            predicted_covariance=filter_state.covariance,
            observation=obs,
            observation_matrix=matrix,
            observation_covariance=cov,
        )

    def _features(
        self,
        information_state: str,
        state: np.ndarray,
        observation: dict[str, np.ndarray],
        aggregate_filter_state: LinearFilterState,
        distribution_filter_state: LinearFilterState,
    ) -> dict[str, float]:
        if information_state == "aggregate_only":
            return {
                "observed_inflation_gap": float(observation["aggregate"][0]),
                "observed_output_gap": float(observation["aggregate"][1]),
            }
        if information_state == "filtered_aggregates":
            return {
                "filtered_inflation_gap": float(aggregate_filter_state.mean[0]),
                "filtered_output_gap": float(aggregate_filter_state.mean[1]),
                "filtered_natural_rate_gap": float(aggregate_filter_state.mean[2]),
            }
        if information_state == "distributional":
            return {
                "filtered_inflation_gap": float(aggregate_filter_state.mean[0]),
                "filtered_output_gap": float(aggregate_filter_state.mean[1]),
                "filtered_natural_rate_gap": float(aggregate_filter_state.mean[2]),
                "filtered_mean_mpc": float(distribution_filter_state.mean[3]),
                "filtered_low_liquidity_share": float(distribution_filter_state.mean[4]),
            }
        if information_state == "distributional_mpc":
            return {
                "filtered_inflation_gap": float(aggregate_filter_state.mean[0]),
                "filtered_output_gap": float(aggregate_filter_state.mean[1]),
                "filtered_natural_rate_gap": float(aggregate_filter_state.mean[2]),
                "filtered_mean_mpc": float(distribution_filter_state.mean[3]),
            }
        if information_state == "distributional_liquidity":
            return {
                "filtered_inflation_gap": float(aggregate_filter_state.mean[0]),
                "filtered_output_gap": float(aggregate_filter_state.mean[1]),
                "filtered_natural_rate_gap": float(aggregate_filter_state.mean[2]),
                "filtered_low_liquidity_share": float(distribution_filter_state.mean[4]),
            }
        if information_state == "full_information":
            return {
                "true_inflation_gap": float(state[0]),
                "true_output_gap": float(state[1]),
                "true_natural_rate_gap": float(state[2]),
                "true_mean_mpc": float(state[3]),
                "true_low_liquidity_share": float(state[4]),
            }
        raise ValueError(f"Unknown information state: {information_state}")

    def _observation(
        self,
        state: np.ndarray,
        aggregate_noise: np.ndarray,
        distribution_noise: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "aggregate": state[[0, 1]] + aggregate_noise,
            "distributional": state[[3, 4]] + distribution_noise,
        }

    def _aggregate_noise_std(self) -> np.ndarray:
        return self.config.aggregate_observation_noise * np.array([0.0045, 0.0055], dtype=float)

    def _distribution_noise_std(self) -> np.ndarray:
        return self.config.distributional_observation_noise * np.array([0.0120, 0.0140], dtype=float)

    def _transmission_strength(self, state: np.ndarray) -> float:
        distributional_shift = 1.8 * state[3] + 1.4 * state[4]
        strength = 0.34 * (1.0 + self.config.mpc_channel_strength * distributional_shift)
        return float(np.clip(strength, 0.08, 0.90))
