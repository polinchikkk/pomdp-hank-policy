from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BasePolicy:
    def reset(self) -> None:
        return None

    def rate(self, observation: np.ndarray, info: dict) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class ClassicalFilteredRulePolicy(BasePolicy):
    action_bound: float

    def rate(self, observation: np.ndarray, info: dict) -> float:
        lower, upper = info["rate_bounds"]
        return float(np.clip(info["filtered_rule_rate"], lower, upper))


@dataclass(frozen=True)
class FullInformationRulePolicy(BasePolicy):
    action_bound: float

    def rate(self, observation: np.ndarray, info: dict) -> float:
        lower, upper = info["rate_bounds"]
        return float(np.clip(info["full_information_rate"], lower, upper))


class PPOPolicy(BasePolicy):
    def __init__(
        self,
        *,
        actor_params: dict[str, np.ndarray],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        normalization_clip: float,
        action_bound: float,
    ) -> None:
        self.actor_params = {name: np.asarray(value, dtype=float).copy() for name, value in actor_params.items()}
        self.feature_mean = np.asarray(feature_mean, dtype=float).copy()
        self.feature_std = np.asarray(feature_std, dtype=float).copy()
        self.normalization_clip = float(normalization_clip)
        self.action_bound = float(action_bound)

    def rate(self, observation: np.ndarray, info: dict) -> float:
        features = np.asarray(observation, dtype=float)[None, :]
        normalized = (features - self.feature_mean[None, :]) / self.feature_std[None, :]
        normalized = np.clip(normalized, -self.normalization_clip, self.normalization_clip)

        hidden_1 = np.tanh(normalized @ self.actor_params["w1"].T + self.actor_params["b1"][None, :])
        hidden_2 = np.tanh(hidden_1 @ self.actor_params["w2"].T + self.actor_params["b2"][None, :])
        raw_output = hidden_2 @ self.actor_params["w3"][:, None] + self.actor_params["b3"]
        residual = self.action_bound * np.tanh(raw_output[:, 0])[0]
        target_rate = float(info["filtered_rule_rate"] + residual)
        lower, upper = info["rate_bounds"]
        return float(np.clip(target_rate, lower, upper))
