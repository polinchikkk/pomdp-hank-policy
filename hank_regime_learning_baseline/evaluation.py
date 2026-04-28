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
        }
        true_state = np.asarray(step_info["true_state"], dtype=float)
        filtered_state = np.asarray(step_info["filtered_state"], dtype=float)
        filtered_variance = np.diag(np.asarray(step_info["filtered_covariance"], dtype=float))
        for index, state_name in enumerate(env.state_names):
            row[f"true_{state_name}"] = float(true_state[index])
            row[f"filtered_{state_name}"] = float(filtered_state[index])
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


def _state_rmse(policy_trace: pd.DataFrame, state_names: tuple[str, ...]) -> tuple[float, float]:
    stacked = []
    distribution_stacked = []
    for state_name in state_names:
        error = (
            policy_trace[f"filtered_{state_name}"].to_numpy(dtype=float)
            - policy_trace[f"true_{state_name}"].to_numpy(dtype=float)
        )
        stacked.append(error)
        if state_name in {"low_liquidity_gap", "mean_mpc_gap"}:
            distribution_stacked.append(error)
    all_errors = np.column_stack(stacked)
    dist_errors = np.column_stack(distribution_stacked)
    return (
        float(np.sqrt(np.mean(np.square(all_errors)))),
        float(np.sqrt(np.mean(np.square(dist_errors)))),
    )


def _is_unstable(policy_trace: pd.DataFrame) -> bool:
    columns = ["true_inflation_gap", "true_output_gap", "policy_rate", "true_low_liquidity_gap", "true_mean_mpc_gap"]
    values = policy_trace[columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        return True
    limits = np.array([0.06, 0.12, 0.06, 0.08, 0.08], dtype=float)
    return bool(np.any(np.abs(values) > limits[None, :]))


def evaluate_policy_trace(
    *,
    policy_trace: pd.DataFrame,
    reference_trace: pd.DataFrame,
    scenario_spec,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    losses = policy_trace["loss"].to_numpy(dtype=float)
    reference_losses = reference_trace["loss"].to_numpy(dtype=float)
    rate = policy_trace["policy_rate"].to_numpy(dtype=float)
    reference_rate = reference_trace["policy_rate"].to_numpy(dtype=float)
    rate_gap = rate - reference_rate
    state_names = (
        "rstar_gap",
        "productivity_gap",
        "fiscal_gap",
        "inflation_gap",
        "output_gap",
        "low_liquidity_gap",
        "mean_mpc_gap",
    )
    mean_state_rmse, distribution_state_rmse = _state_rmse(policy_trace, state_names)
    stress_probability = policy_trace["stress_probability"].to_numpy(dtype=float)
    hidden_regime = policy_trace["hidden_regime"].to_numpy(dtype=int)
    path_frame = pd.DataFrame({
        "variant_name": scenario_spec.variant_name,
        "scenario_name": scenario_spec.scenario_name,
        "scenario_label": scenario_spec.scenario_label,
        "policy_name": policy_trace["policy_name"].iloc[0],
        "policy_label": policy_trace["policy_label"].iloc[0],
        "training_seed": int(policy_trace["training_seed"].iloc[0]),
        "evaluation_seed": int(policy_trace["evaluation_seed"].iloc[0]),
        "period": policy_trace["period"].to_numpy(dtype=int),
        "hidden_regime": hidden_regime,
        "stress_probability": stress_probability,
        "policy_rate": rate,
        "reference_rate": reference_rate,
        "rate_gap": rate_gap,
        "abs_rate_gap": np.abs(rate_gap),
        "policy_loss": losses,
        "reference_loss": reference_losses,
        "cumulative_policy_loss": np.cumsum(losses),
        "cumulative_reference_loss": np.cumsum(reference_losses),
        "inflation_gap": policy_trace["true_inflation_gap"].to_numpy(dtype=float),
        "output_gap": policy_trace["true_output_gap"].to_numpy(dtype=float),
        "low_liquidity_gap": policy_trace["true_low_liquidity_gap"].to_numpy(dtype=float),
        "mean_mpc_gap": policy_trace["true_mean_mpc_gap"].to_numpy(dtype=float),
    })
    metrics = {
        "variant_name": scenario_spec.variant_name,
        "scenario_name": scenario_spec.scenario_name,
        "scenario_label": scenario_spec.scenario_label,
        "policy_name": policy_trace["policy_name"].iloc[0],
        "policy_label": policy_trace["policy_label"].iloc[0],
        "training_seed": int(policy_trace["training_seed"].iloc[0]),
        "evaluation_seed": int(policy_trace["evaluation_seed"].iloc[0]),
        "input_mode": scenario_spec.input_mode,
        "include_distributional_state": int(scenario_spec.include_distributional_state),
        "mean_policy_loss": float(np.mean(losses)),
        "cumulative_policy_loss": float(np.sum(losses)),
        "mean_reference_loss": float(np.mean(reference_losses)),
        "cumulative_reference_loss": float(np.sum(reference_losses)),
        "mean_excess_loss": float(np.mean(losses - reference_losses)),
        "cumulative_excess_loss": float(np.sum(losses - reference_losses)),
        "mean_abs_rate_gap": float(np.mean(np.abs(rate_gap))),
        "policy_rate_rmse": float(np.sqrt(np.mean(np.square(rate_gap)))),
        "policy_instrument_volatility": float(np.std(rate)),
        "mean_state_rmse": mean_state_rmse,
        "distribution_state_rmse": distribution_state_rmse,
        "regime_accuracy": float(np.mean((stress_probability >= 0.5).astype(int) == hidden_regime)),
        "stress_brier_score": float(np.mean(np.square(stress_probability - hidden_regime))),
        "unstable": int(_is_unstable(policy_trace)),
        "impact_nominal_rate_pp": float(100.0 * rate[0]),
        "impact_inflation_pp": float(100.0 * policy_trace["true_inflation_gap"].iloc[0]),
        "impact_output_gap_pct": float(100.0 * policy_trace["true_output_gap"].iloc[0]),
        "impact_low_liquidity_gap_pp": float(100.0 * policy_trace["true_low_liquidity_gap"].iloc[0]),
        "impact_mean_mpc_gap_pp": float(100.0 * policy_trace["true_mean_mpc_gap"].iloc[0]),
        "peak_low_liquidity_gap": float(np.max(np.abs(policy_trace["true_low_liquidity_gap"].to_numpy(dtype=float)))),
        "peak_mean_mpc_gap": float(np.max(np.abs(policy_trace["true_mean_mpc_gap"].to_numpy(dtype=float)))),
        "peak_low_liquidity_gap_difference_vs_reference": float(
            np.max(
                np.abs(
                    policy_trace["true_low_liquidity_gap"].to_numpy(dtype=float)
                    - reference_trace["true_low_liquidity_gap"].to_numpy(dtype=float)
                )
            )
        ),
        "peak_mean_mpc_gap_difference_vs_reference": float(
            np.max(
                np.abs(
                    policy_trace["true_mean_mpc_gap"].to_numpy(dtype=float)
                    - reference_trace["true_mean_mpc_gap"].to_numpy(dtype=float)
                )
            )
        ),
    }
    return metrics, path_frame


def summarize_training_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(
            columns=["variant_name", "training_seed", "best_validation_return", "final_validation_return", "final_mean_episode_return"]
        )
    grouped = history.sort_values(["label", "training_seed", "iteration"]).groupby(["label", "training_seed"], as_index=False)
    rows = []
    for (label, training_seed), frame in grouped:
        rows.append({
            "variant_name": label,
            "training_seed": int(training_seed),
            "best_validation_return": float(frame["best_validation_return"].max()),
            "final_validation_return": float(frame["validation_return"].iloc[-1]),
            "final_mean_episode_return": float(frame["mean_episode_return"].iloc[-1]),
        })
    return pd.DataFrame(rows)


def build_policy_comparison(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    focus = policy_metrics[policy_metrics["policy_name"].isin(["classical_filtered_rule", "learning_policy", "full_information_rule"])].copy()
    rows = []
    for variant_name, frame in focus.groupby("variant_name"):
        by_policy = {}
        for name, sub in frame.groupby("policy_name"):
            numeric_columns = sub.select_dtypes(include=[np.number]).columns
            aggregated = {}
            for column in numeric_columns:
                if column in {"training_seed", "evaluation_seed"}:
                    continue
                if column == "unstable":
                    aggregated[column] = int(np.max(sub[column].to_numpy(dtype=int)))
                else:
                    aggregated[column] = float(np.mean(sub[column].to_numpy(dtype=float)))
            aggregated["policy_name"] = name
            aggregated["policy_label"] = sub["policy_label"].iloc[0]
            aggregated["scenario_name"] = sub["scenario_name"].iloc[0]
            aggregated["scenario_label"] = sub["scenario_label"].iloc[0]
            by_policy[name] = aggregated
        if "classical_filtered_rule" not in by_policy or "learning_policy" not in by_policy:
            continue
        classical = by_policy["classical_filtered_rule"]
        rl = by_policy["learning_policy"]
        full_info = by_policy.get("full_information_rule", classical)
        rows.append({
            "variant_name": variant_name,
            "scenario_name": frame["scenario_name"].iloc[0],
            "scenario_label": frame["scenario_label"].iloc[0],
            "classical_mean_policy_loss": float(classical["mean_policy_loss"]),
            "rl_mean_policy_loss": float(rl["mean_policy_loss"]),
            "full_information_mean_policy_loss": float(full_info["mean_policy_loss"]),
            "delta_mean_policy_loss_rl_minus_classical": float(rl["mean_policy_loss"] - classical["mean_policy_loss"]),
            "delta_cumulative_policy_loss_rl_minus_classical": float(rl["cumulative_policy_loss"] - classical["cumulative_policy_loss"]),
            "classical_policy_rate_rmse": float(classical["policy_rate_rmse"]),
            "rl_policy_rate_rmse": float(rl["policy_rate_rmse"]),
            "classical_regime_accuracy": float(classical["regime_accuracy"]),
            "rl_regime_accuracy": float(rl["regime_accuracy"]),
            "classical_unstable": int(classical["unstable"]),
            "rl_unstable": int(rl["unstable"]),
        })
    return pd.DataFrame(rows)
