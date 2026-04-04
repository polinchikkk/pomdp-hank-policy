from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PPOConfig
from .policies import PPOPolicy


@dataclass(frozen=True)
class PPOCheckpoint:
    iteration: int
    policy: PPOPolicy
    validation_return: float
    mean_episode_return: float


def _normalize_features(
    features: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    normalization_clip: float,
) -> np.ndarray:
    normalized = (features - feature_mean[None, :]) / feature_std[None, :]
    return np.clip(normalized, -normalization_clip, normalization_clip)


def _init_actor_params(
    *,
    feature_dim: int,
    hidden_dim_1: int,
    hidden_dim_2: int,
    rng: np.random.Generator,
    initial_log_std: float,
) -> dict[str, np.ndarray]:
    return {
        "w1": rng.normal(0.0, 0.08, size=(hidden_dim_1, feature_dim)),
        "b1": np.zeros((hidden_dim_1,), dtype=float),
        "w2": rng.normal(0.0, 0.08, size=(hidden_dim_2, hidden_dim_1)),
        "b2": np.zeros((hidden_dim_2,), dtype=float),
        "w3": rng.normal(0.0, 0.05, size=(hidden_dim_2,)),
        "b3": np.array(0.0, dtype=float),
        "log_std": np.array(initial_log_std, dtype=float),
    }


def _init_critic_params(
    *,
    feature_dim: int,
    hidden_dim_1: int,
    hidden_dim_2: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    return {
        "w1": rng.normal(0.0, 0.08, size=(hidden_dim_1, feature_dim)),
        "b1": np.zeros((hidden_dim_1,), dtype=float),
        "w2": rng.normal(0.0, 0.08, size=(hidden_dim_2, hidden_dim_1)),
        "b2": np.zeros((hidden_dim_2,), dtype=float),
        "w3": rng.normal(0.0, 0.05, size=(hidden_dim_2,)),
        "b3": np.array(0.0, dtype=float),
    }


def _actor_forward(
    actor_params: dict[str, np.ndarray],
    features: np.ndarray,
    action_bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hidden_pre_1 = features @ actor_params["w1"].T + actor_params["b1"][None, :]
    hidden_1 = np.tanh(hidden_pre_1)
    hidden_pre_2 = hidden_1 @ actor_params["w2"].T + actor_params["b2"][None, :]
    hidden_2 = np.tanh(hidden_pre_2)
    raw_output = hidden_2 @ actor_params["w3"][:, None] + actor_params["b3"]
    bounded_mean = action_bound * np.tanh(raw_output[:, 0])
    log_std = np.full_like(bounded_mean, float(actor_params["log_std"]))
    return hidden_pre_1, hidden_1, hidden_pre_2, hidden_2, raw_output[:, 0], bounded_mean, log_std


def _critic_forward(
    critic_params: dict[str, np.ndarray],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hidden_pre_1 = features @ critic_params["w1"].T + critic_params["b1"][None, :]
    hidden_1 = np.tanh(hidden_pre_1)
    hidden_pre_2 = hidden_1 @ critic_params["w2"].T + critic_params["b2"][None, :]
    hidden_2 = np.tanh(hidden_pre_2)
    values = hidden_2 @ critic_params["w3"][:, None] + critic_params["b3"]
    return hidden_pre_1, hidden_1, hidden_pre_2, hidden_2, values[:, 0]


def _gaussian_log_prob(actions: np.ndarray, means: np.ndarray, log_stds: np.ndarray) -> np.ndarray:
    variances = np.exp(2.0 * log_stds)
    return -0.5 * (np.log(2.0 * np.pi) + 2.0 * log_stds + np.square(actions - means) / variances)


def _gaussian_entropy(log_stds: np.ndarray) -> np.ndarray:
    return log_stds + 0.5 * (1.0 + np.log(2.0 * np.pi))


def _parameter_like(reference: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.zeros_like(value) for name, value in reference.items()}


def _optimizer_state(params: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray] | int]:
    return {"step": 0, "m": _parameter_like(params), "v": _parameter_like(params)}


def _adam_update(
    params: dict[str, np.ndarray],
    gradients: dict[str, np.ndarray],
    optimizer_state: dict[str, dict[str, np.ndarray] | int],
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
) -> None:
    optimizer_state["step"] = int(optimizer_state["step"]) + 1
    step = int(optimizer_state["step"])
    moment1 = optimizer_state["m"]
    moment2 = optimizer_state["v"]
    assert isinstance(moment1, dict)
    assert isinstance(moment2, dict)

    for name in params:
        moment1[name] = beta1 * moment1[name] + (1.0 - beta1) * gradients[name]
        moment2[name] = beta2 * moment2[name] + (1.0 - beta2) * np.square(gradients[name])
        moment1_hat = moment1[name] / (1.0 - beta1**step)
        moment2_hat = moment2[name] / (1.0 - beta2**step)
        params[name] -= learning_rate * moment1_hat / (np.sqrt(moment2_hat) + epsilon)


def _clip_gradients(gradients: dict[str, np.ndarray], max_grad_norm: float) -> dict[str, np.ndarray]:
    total_norm = float(np.sqrt(sum(float(np.sum(np.square(value))) for value in gradients.values())))
    if total_norm <= max_grad_norm or total_norm == 0.0:
        return gradients
    scale = max_grad_norm / (total_norm + 1.0e-8)
    return {name: value * scale for name, value in gradients.items()}


def _estimate_feature_normalizer(
    *,
    env_factory,
    num_episodes: int,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    for episode_index in range(num_episodes):
        env = env_factory()
        observation, info = env.reset(seed=seed_base + episode_index)
        done = False
        while not done:
            rows.append(np.asarray(observation, dtype=float))
            observation, _, done, _ = env.step_rate(float(info["filtered_rule_rate"]))
            info = env.current_context.copy()
    matrix = np.asarray(rows, dtype=float)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    std = np.where(std < 1.0e-6, 1.0, std)
    return mean, std


def _collect_rollout_batch(
    *,
    env_factory,
    actor_params: dict[str, np.ndarray],
    critic_params: dict[str, np.ndarray],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    ppo_config: PPOConfig,
    action_bound: float,
    gamma: float,
    rollout_seed_base: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(rollout_seed_base)
    feature_rows: list[np.ndarray] = []
    action_rows: list[float] = []
    reward_rows: list[float] = []
    value_rows: list[float] = []
    log_prob_rows: list[float] = []
    old_mean_rows: list[float] = []
    old_log_std_rows: list[float] = []
    done_rows: list[float] = []
    episode_returns: list[float] = []

    for episode_index in range(ppo_config.rollout_episodes):
        env = env_factory()
        observation, info = env.reset(seed=rollout_seed_base + episode_index)
        done = False
        episode_return = 0.0
        while not done:
            raw_features = np.asarray(observation, dtype=float)
            features = _normalize_features(
                raw_features[None, :],
                feature_mean,
                feature_std,
                ppo_config.normalization_clip,
            )[0]
            _, _, _, _, values = _critic_forward(critic_params, features[None, :])
            _, _, _, _, _, means, log_stds = _actor_forward(actor_params, features[None, :], action_bound)
            sampled_action = float(rng.normal(loc=means[0], scale=np.exp(log_stds[0])))
            sampled_action = float(np.clip(sampled_action, -action_bound, action_bound))
            applied_rate = float(np.clip(info["filtered_rule_rate"] + sampled_action, info["rate_bounds"][0], info["rate_bounds"][1]))
            applied_action = float(applied_rate - info["filtered_rule_rate"])
            log_probability = float(
                _gaussian_log_prob(
                    np.array([applied_action], dtype=float),
                    means,
                    log_stds,
                )[0]
            )
            next_observation, reward, done, _ = env.step_rate(applied_rate)

            feature_rows.append(features)
            action_rows.append(applied_action)
            reward_rows.append(float(reward))
            value_rows.append(float(values[0]))
            log_prob_rows.append(log_probability)
            old_mean_rows.append(float(means[0]))
            old_log_std_rows.append(float(log_stds[0]))
            done_rows.append(float(done))

            episode_return += float(reward)
            observation = next_observation
            info = env.current_context.copy()
        episode_returns.append(episode_return)

    features_array = np.asarray(feature_rows, dtype=float)
    actions_array = np.asarray(action_rows, dtype=float)
    rewards_array = np.asarray(reward_rows, dtype=float)
    values_array = np.asarray(value_rows, dtype=float)
    old_log_probs_array = np.asarray(log_prob_rows, dtype=float)
    old_means_array = np.asarray(old_mean_rows, dtype=float)
    old_log_stds_array = np.asarray(old_log_std_rows, dtype=float)
    dones_array = np.asarray(done_rows, dtype=float)

    advantages = np.zeros_like(rewards_array)
    returns = np.zeros_like(rewards_array)
    gae = 0.0
    next_value = 0.0
    for index in range(len(rewards_array) - 1, -1, -1):
        non_terminal = 1.0 - dones_array[index]
        delta = rewards_array[index] + gamma * next_value * non_terminal - values_array[index]
        gae = delta + gamma * ppo_config.gae_lambda * non_terminal * gae
        advantages[index] = gae
        returns[index] = gae + values_array[index]
        next_value = values_array[index]
    advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1.0e-8)

    batch = {
        "features": features_array,
        "actions": actions_array,
        "returns": returns,
        "advantages": advantages,
        "old_log_probs": old_log_probs_array,
        "old_means": old_means_array,
        "old_log_stds": old_log_stds_array,
    }
    diagnostics = {
        "mean_episode_return": float(np.mean(episode_returns)),
        "std_episode_return": float(np.std(episode_returns)),
        "num_steps": float(len(features_array)),
    }
    return batch, diagnostics


def _actor_gradients(
    actor_params: dict[str, np.ndarray],
    features: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    old_log_probs: np.ndarray,
    action_bound: float,
    clip_epsilon: float,
    entropy_coefficient: float,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    hidden_pre_1, hidden_1, hidden_pre_2, hidden_2, raw_output, means, log_stds = _actor_forward(
        actor_params,
        features,
        action_bound,
    )
    stds = np.exp(log_stds)
    new_log_probs = _gaussian_log_prob(actions, means, log_stds)
    ratios = np.exp(new_log_probs - old_log_probs)
    positive_advantages = advantages >= 0.0
    clipped_mask = (positive_advantages & (ratios > 1.0 + clip_epsilon)) | (
        (~positive_advantages) & (ratios < 1.0 - clip_epsilon)
    )
    active = (~clipped_mask).astype(float)
    weights = -advantages * ratios * active / len(actions)

    d_log_prob_d_mean = (actions - means) / np.square(stds)
    d_log_prob_d_log_std = np.square(actions - means) / np.square(stds) - 1.0
    d_mean_d_raw = action_bound * (1.0 - np.square(np.tanh(raw_output)))
    d_loss_d_raw = weights * d_log_prob_d_mean * d_mean_d_raw

    grad_w3 = hidden_2.T @ d_loss_d_raw
    grad_b3 = float(np.sum(d_loss_d_raw))
    grad_hidden_2 = d_loss_d_raw[:, None] * actor_params["w3"][None, :]
    grad_hidden_pre_2 = grad_hidden_2 * (1.0 - np.square(hidden_2))
    grad_w2 = grad_hidden_pre_2.T @ hidden_1
    grad_b2 = np.sum(grad_hidden_pre_2, axis=0)
    grad_hidden_1 = grad_hidden_pre_2 @ actor_params["w2"]
    grad_hidden_pre_1 = grad_hidden_1 * (1.0 - np.square(hidden_1))
    grad_w1 = grad_hidden_pre_1.T @ features
    grad_b1 = np.sum(grad_hidden_pre_1, axis=0)
    grad_log_std = np.sum(weights * d_log_prob_d_log_std) - entropy_coefficient

    gradients = {
        "w1": grad_w1,
        "b1": grad_b1,
        "w2": grad_w2,
        "b2": grad_b2,
        "w3": grad_w3,
        "b3": np.array(grad_b3, dtype=float),
        "log_std": np.array(grad_log_std, dtype=float),
    }
    diagnostics = {
        "mean_ratio": float(np.mean(ratios)),
        "clip_fraction": float(np.mean(clipped_mask)),
        "mean_log_prob": float(np.mean(new_log_probs)),
        "mean_action_std": float(np.mean(stds)),
        "mean_entropy": float(np.mean(_gaussian_entropy(log_stds))),
    }
    return gradients, diagnostics


def _critic_gradients(
    critic_params: dict[str, np.ndarray],
    features: np.ndarray,
    returns: np.ndarray,
) -> tuple[dict[str, np.ndarray], float]:
    hidden_pre_1, hidden_1, hidden_pre_2, hidden_2, values = _critic_forward(critic_params, features)
    value_errors = (values - returns) / len(returns)

    grad_w3 = hidden_2.T @ value_errors
    grad_b3 = float(np.sum(value_errors))
    grad_hidden_2 = value_errors[:, None] * critic_params["w3"][None, :]
    grad_hidden_pre_2 = grad_hidden_2 * (1.0 - np.square(hidden_2))
    grad_w2 = grad_hidden_pre_2.T @ hidden_1
    grad_b2 = np.sum(grad_hidden_pre_2, axis=0)
    grad_hidden_1 = grad_hidden_pre_2 @ critic_params["w2"]
    grad_hidden_pre_1 = grad_hidden_1 * (1.0 - np.square(hidden_1))
    grad_w1 = grad_hidden_pre_1.T @ features
    grad_b1 = np.sum(grad_hidden_pre_1, axis=0)

    gradients = {
        "w1": grad_w1,
        "b1": grad_b1,
        "w2": grad_w2,
        "b2": grad_b2,
        "w3": grad_w3,
        "b3": np.array(grad_b3, dtype=float),
    }
    critic_loss = 0.5 * float(np.mean(np.square(values - returns)))
    return gradients, critic_loss


def _validation_return(
    *,
    env_factory,
    actor_params: dict[str, np.ndarray],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    normalization_clip: float,
    action_bound: float,
    validation_seed_base: int,
    validation_episodes: int,
) -> float:
    returns: list[float] = []
    for episode_index in range(validation_episodes):
        env = env_factory()
        observation, info = env.reset(seed=validation_seed_base + episode_index)
        done = False
        episode_return = 0.0
        while not done:
            features = _normalize_features(
                np.asarray(observation, dtype=float)[None, :],
                feature_mean,
                feature_std,
                normalization_clip,
            )
            _, _, _, _, _, means, _ = _actor_forward(actor_params, features, action_bound)
            policy_rate = float(np.clip(info["filtered_rule_rate"] + means[0], info["rate_bounds"][0], info["rate_bounds"][1]))
            observation, reward, done, _ = env.step_rate(policy_rate)
            episode_return += float(reward)
            info = env.current_context.copy()
        returns.append(episode_return)
    return float(np.mean(returns))


def train_ppo_policy(
    *,
    env_factory,
    ppo_config: PPOConfig,
    action_bound: float,
    gamma: float,
    training_seed: int,
    label: str,
) -> tuple[PPOPolicy, pd.DataFrame, list[PPOCheckpoint]]:
    env = env_factory()
    initial_observation, _ = env.reset(seed=training_seed)
    feature_dim = initial_observation.shape[0]

    rng = np.random.default_rng(training_seed)
    actor_params = _init_actor_params(
        feature_dim=feature_dim,
        hidden_dim_1=ppo_config.hidden_dim_1,
        hidden_dim_2=ppo_config.hidden_dim_2,
        rng=rng,
        initial_log_std=ppo_config.initial_log_std,
    )
    critic_params = _init_critic_params(
        feature_dim=feature_dim,
        hidden_dim_1=ppo_config.hidden_dim_1,
        hidden_dim_2=ppo_config.hidden_dim_2,
        rng=rng,
    )
    actor_optimizer = _optimizer_state(actor_params)
    critic_optimizer = _optimizer_state(critic_params)
    feature_mean, feature_std = _estimate_feature_normalizer(
        env_factory=env_factory,
        num_episodes=ppo_config.normalization_episodes,
        seed_base=training_seed * 10_000,
    )

    best_actor_params = {name: value.copy() for name, value in actor_params.items()}
    best_validation = -np.inf
    best_return_so_far = -np.inf
    history_rows: list[dict[str, float | int | str]] = []
    checkpoints: list[PPOCheckpoint] = []

    for iteration in range(ppo_config.num_iterations):
        batch, rollout_diagnostics = _collect_rollout_batch(
            env_factory=env_factory,
            actor_params=actor_params,
            critic_params=critic_params,
            feature_mean=feature_mean,
            feature_std=feature_std,
            ppo_config=ppo_config,
            action_bound=action_bound,
            gamma=gamma,
            rollout_seed_base=training_seed * 100_000 + iteration * 1_000,
        )
        features = batch["features"]
        actions = batch["actions"]
        returns = batch["returns"]
        advantages = batch["advantages"]
        old_log_probs = batch["old_log_probs"]

        indices = np.arange(len(actions))
        actor_diag = {
            "mean_ratio": np.nan,
            "clip_fraction": np.nan,
            "mean_log_prob": np.nan,
            "mean_action_std": np.nan,
            "mean_entropy": np.nan,
        }
        critic_loss = np.nan

        for _epoch in range(ppo_config.num_epochs):
            rng.shuffle(indices)
            for start in range(0, len(indices), ppo_config.minibatch_size):
                minibatch = indices[start : start + ppo_config.minibatch_size]
                actor_gradients, actor_diag = _actor_gradients(
                    actor_params,
                    features[minibatch],
                    actions[minibatch],
                    advantages[minibatch],
                    old_log_probs[minibatch],
                    action_bound,
                    ppo_config.clip_epsilon,
                    ppo_config.entropy_coefficient,
                )
                actor_gradients = _clip_gradients(actor_gradients, ppo_config.max_grad_norm)
                _adam_update(
                    actor_params,
                    actor_gradients,
                    actor_optimizer,
                    learning_rate=ppo_config.actor_learning_rate,
                )
                actor_params["log_std"] = np.clip(
                    actor_params["log_std"],
                    ppo_config.min_log_std,
                    ppo_config.max_log_std,
                )

                critic_gradients, critic_loss = _critic_gradients(
                    critic_params,
                    features[minibatch],
                    returns[minibatch],
                )
                critic_gradients = _clip_gradients(critic_gradients, ppo_config.max_grad_norm)
                _adam_update(
                    critic_params,
                    critic_gradients,
                    critic_optimizer,
                    learning_rate=ppo_config.critic_learning_rate,
                )

        validation_return = _validation_return(
            env_factory=env_factory,
            actor_params=actor_params,
            feature_mean=feature_mean,
            feature_std=feature_std,
            normalization_clip=ppo_config.normalization_clip,
            action_bound=action_bound,
            validation_seed_base=training_seed * 1_000_000 + iteration * 100,
            validation_episodes=ppo_config.validation_episodes,
        )
        if validation_return > best_validation:
            best_validation = validation_return
            best_actor_params = {name: value.copy() for name, value in actor_params.items()}
        best_return_so_far = max(best_return_so_far, rollout_diagnostics["mean_episode_return"])

        if (
            iteration % ppo_config.checkpoint_interval == 0
            or iteration == ppo_config.num_iterations - 1
            or validation_return >= best_validation
        ):
            checkpoints.append(
                PPOCheckpoint(
                    iteration=int(iteration),
                    policy=PPOPolicy(
                        actor_params={name: value.copy() for name, value in actor_params.items()},
                        feature_mean=feature_mean,
                        feature_std=feature_std,
                        normalization_clip=ppo_config.normalization_clip,
                        action_bound=action_bound,
                    ),
                    validation_return=float(validation_return),
                    mean_episode_return=float(rollout_diagnostics["mean_episode_return"]),
                )
            )

        history_rows.append({
            "label": label,
            "training_seed": int(training_seed),
            "iteration": int(iteration),
            "mean_episode_return": float(rollout_diagnostics["mean_episode_return"]),
            "std_episode_return": float(rollout_diagnostics["std_episode_return"]),
            "best_return_so_far": float(best_return_so_far),
            "validation_return": float(validation_return),
            "best_validation_return": float(best_validation),
            "critic_loss": float(critic_loss),
            "mean_ratio": float(actor_diag["mean_ratio"]),
            "clip_fraction": float(actor_diag["clip_fraction"]),
            "mean_log_prob": float(actor_diag["mean_log_prob"]),
            "mean_action_std": float(actor_diag["mean_action_std"]),
            "mean_entropy": float(actor_diag["mean_entropy"]),
            "num_steps": float(rollout_diagnostics["num_steps"]),
        })

    policy = PPOPolicy(
        actor_params=best_actor_params,
        feature_mean=feature_mean,
        feature_std=feature_std,
        normalization_clip=ppo_config.normalization_clip,
        action_bound=action_bound,
    )
    return policy, pd.DataFrame(history_rows), checkpoints
