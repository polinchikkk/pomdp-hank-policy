from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_policy_episode(
    *,
    env_factory,
    policy,
    scenario_spec,
    evaluation_seed: int,
    policy_name: str,
    policy_label: str,
    training_seed: int | None,
) -> pd.DataFrame:
    env = env_factory()
    observation, info = env.reset(seed=evaluation_seed)
    policy.reset()
    rows: list[dict[str, float | int | str]] = []
    done = False

    while not done:
        chosen_rate = float(policy.rate(observation, info))
        next_observation, reward, done, step_info = env.step_rate(chosen_rate)
        row: dict[str, float | int | str] = {
            "variant_name": scenario_spec.variant_name,
            "scenario_name": scenario_spec.scenario_name,
            "scenario_label": scenario_spec.scenario_label,
            "policy_name": policy_name,
            "policy_label": policy_label,
            "training_seed": -1 if training_seed is None else int(training_seed),
            "evaluation_seed": int(evaluation_seed),
            "period": int(step_info["t"]),
            "input_mode": scenario_spec.input_mode,
            "include_distributional_state": int(scenario_spec.include_distributional_state),
            "hidden_regime": int(step_info["hidden_regime"]),
            "stress_probability": float(step_info["stress_probability"]),
            "stress_entropy": float(step_info["stress_entropy"]),
            "reward": float(reward),
            "loss": float(step_info["loss"]),
            "inflation_loss": float(step_info["inflation_loss"]),
            "output_gap_loss": float(step_info["output_gap_loss"]),
            "rate_change_loss": float(step_info["rate_change_loss"]),
            "policy_rate": float(step_info["policy_rate"]),
            "filtered_rule_rate": float(step_info["filtered_rule_rate"]),
            "full_information_rate": float(step_info["full_information_rate"]),
            "residual_action": float(step_info["residual_action"]),
            "policy_shock": float(step_info["policy_shock"]),
            "current_rate": float(step_info["current_rate"]),
            "lagged_policy_rate": float(step_info["current_rate"]),
            "filtered_variance_trace": float(step_info["filtered_variance_trace"]),
            "filtered_macro_variance_trace": float(step_info["filtered_macro_variance_trace"]),
            "regime_mean_delta_norm": float(np.linalg.norm(np.asarray(step_info["regime_mean_delta"], dtype=float))),
            "stress_interaction_norm": float(np.linalg.norm(np.asarray(step_info["stress_interaction_state"], dtype=float))),
        }
        true_state = np.asarray(step_info["true_state"], dtype=float)
        filtered_state = np.asarray(step_info["filtered_state"], dtype=float)
        normal_state = np.asarray(step_info["normal_regime_state_mean"], dtype=float)
        stress_state = np.asarray(step_info["stress_regime_state_mean"], dtype=float)
        regime_delta = np.asarray(step_info["regime_mean_delta"], dtype=float)
        stress_interaction = np.asarray(step_info["stress_interaction_state"], dtype=float)
        filtered_variance = np.diag(np.asarray(step_info["filtered_covariance"], dtype=float))
        for index, state_name in enumerate(env.state_names):
            row[f"true_{state_name}"] = float(true_state[index])
            row[f"filtered_{state_name}"] = float(filtered_state[index])
            row[f"normal_mean_{state_name}"] = float(normal_state[index])
            row[f"stress_mean_{state_name}"] = float(stress_state[index])
            row[f"regime_delta_{state_name}"] = float(regime_delta[index])
            row[f"stress_interaction_{state_name}"] = float(stress_interaction[index])
            row[f"filtered_var_{state_name}"] = float(filtered_variance[index])
        observations = np.asarray(step_info["current_observations"], dtype=float)
        lagged_observations = np.asarray(step_info["lagged_observations"], dtype=float)
        for index, observation_name in enumerate(step_info["noisy_observation_names"]):
            row[f"observed_{observation_name}"] = float(observations[index])
            row[f"lagged_observed_{observation_name}"] = float(lagged_observations[index])
        rows.append(row)
        observation = next_observation
        info = env.current_context.copy()

    return pd.DataFrame(rows)
