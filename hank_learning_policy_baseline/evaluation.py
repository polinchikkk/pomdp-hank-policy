from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hank_full_baseline.distribution import household_path_levels, path_distribution_statistics
from hank_full_baseline.household_solver import compute_mpc_path
from hank_full_baseline.irfs import aggregate_paths_frame, group_paths_frame
from hank_full_baseline.transition import solve_transition
from hank_partial_info_baseline.evaluation import (
    build_distributional_summary,
    compare_distributional_groups,
)


@dataclass(frozen=True)
class PolicyEpisodeResult:
    policy_trace: pd.DataFrame
    aggregate_paths: pd.DataFrame
    distribution_stats: pd.DataFrame
    group_paths: pd.DataFrame


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
            "reward": float(reward),
            "loss": float(step_info["loss"]),
            "inflation_loss": float(step_info["inflation_loss"]),
            "output_gap_loss": float(step_info["output_gap_loss"]),
            "rate_change_loss": float(step_info["rate_change_loss"]),
            "policy_rate_reduced": float(step_info["policy_rate"]),
            "filtered_rule_rate_reduced": float(step_info["filtered_rule_rate"]),
            "full_information_rate_reduced": float(step_info["full_information_rate"]),
            "residual_action": float(step_info["residual_action"]),
            "policy_shock": float(step_info["policy_shock"]),
            "current_rate": float(step_info["current_rate"]),
        }

        true_state = np.asarray(step_info["true_state"], dtype=float)
        filtered_state = np.asarray(step_info["filtered_state"], dtype=float)
        filtered_variance = np.diag(np.asarray(step_info["filtered_covariance"], dtype=float))
        for index, state_name in enumerate(env.state_names):
            row[f"true_{state_name}"] = float(true_state[index])
            row[f"filtered_{state_name}"] = float(filtered_state[index])
            row[f"filtered_var_{state_name}"] = float(filtered_variance[index])

        observations = np.asarray(step_info["current_observations"], dtype=float)
        for index, observation_name in enumerate(step_info["noisy_observation_names"]):
            row[f"observed_{observation_name}"] = float(observations[index])

        rows.append(row)
        observation = next_observation
        info = env.current_context.copy()

    return pd.DataFrame(rows)


def evaluate_episode_on_full_hank(
    *,
    bundle,
    ss,
    hank_config,
    mpc_ss,
    policy_trace: pd.DataFrame,
    scenario_spec,
    policy_name: str,
    policy_label: str,
    training_seed: int | None,
    evaluation_seed: int,
) -> PolicyEpisodeResult:
    shock_inputs = {
        "rstar": policy_trace["true_rstar_gap"].to_numpy(dtype=float),
        "Z": policy_trace["true_productivity_gap"].to_numpy(dtype=float),
        "G": policy_trace["true_fiscal_gap"].to_numpy(dtype=float),
        "monetary_policy_shock": policy_trace["policy_shock"].to_numpy(dtype=float),
    }
    transition = solve_transition(bundle, shock_inputs)

    aggregate_paths = aggregate_paths_frame(
        ss,
        transition,
        scenario_spec.variant_name,
        scenario_spec.scenario_label,
    )
    aggregate_paths.insert(0, "evaluation_seed", int(evaluation_seed))
    aggregate_paths.insert(0, "training_seed", -1 if training_seed is None else int(training_seed))
    aggregate_paths.insert(0, "policy_label", policy_label)
    aggregate_paths.insert(0, "policy_name", policy_name)

    path_levels = household_path_levels(ss, transition)
    mpc_path = compute_mpc_path(path_levels)
    distribution_stats = path_distribution_statistics(ss, path_levels, hank_config, mpc_path)
    distribution_stats.insert(0, "evaluation_seed", int(evaluation_seed))
    distribution_stats.insert(0, "training_seed", -1 if training_seed is None else int(training_seed))
    distribution_stats.insert(0, "policy_label", policy_label)
    distribution_stats.insert(0, "policy_name", policy_name)
    distribution_stats.insert(0, "scenario_label", scenario_spec.scenario_label)
    distribution_stats.insert(0, "scenario", scenario_spec.variant_name)

    group_paths = group_paths_frame(
        ss,
        transition,
        hank_config,
        mpc_ss,
        scenario_spec.variant_name,
        scenario_spec.scenario_label,
    )
    group_paths.insert(0, "evaluation_seed", int(evaluation_seed))
    group_paths.insert(0, "training_seed", -1 if training_seed is None else int(training_seed))
    group_paths.insert(0, "policy_label", policy_label)
    group_paths.insert(0, "policy_name", policy_name)

    return PolicyEpisodeResult(
        policy_trace=policy_trace,
        aggregate_paths=aggregate_paths,
        distribution_stats=distribution_stats,
        group_paths=group_paths,
    )


def _loss_from_aggregate_paths(
    aggregate_paths: pd.DataFrame,
    *,
    lambda_y: float,
    lambda_i: float,
) -> np.ndarray:
    inflation = aggregate_paths["pi_deviation"].to_numpy(dtype=float)
    output_gap = aggregate_paths["output_gap_deviation"].to_numpy(dtype=float)
    rate = aggregate_paths["i_deviation"].to_numpy(dtype=float)
    return np.square(inflation) + lambda_y * np.square(output_gap) + lambda_i * np.square(np.diff(rate, prepend=0.0))


def _distributional_peaks(
    *,
    group_paths: pd.DataFrame,
    distribution_stats: pd.DataFrame,
    steady_state_statistics: dict[str, float],
) -> dict[str, float]:
    liquid_groups = group_paths[group_paths["grouping"] == "liquid_wealth_quantile"]
    q1 = liquid_groups[liquid_groups["group"] == "liquid_q1"]["consumption_pct_deviation"].to_numpy(dtype=float)
    q5 = liquid_groups[liquid_groups["group"] == "liquid_q5"]["consumption_pct_deviation"].to_numpy(dtype=float)
    low_liquidity_change = distribution_stats["share_low_liquidity"].to_numpy(dtype=float) - float(
        steady_state_statistics["share_low_liquidity"]
    )
    mean_mpc_change = distribution_stats["mean_mpc"].to_numpy(dtype=float) - float(
        steady_state_statistics["mean_mpc"]
    )
    return {
        "peak_consumption_q1": float(np.min(q1)) if q1.size else 0.0,
        "peak_consumption_q5": float(np.min(q5)) if q5.size else 0.0,
        "peak_low_liquidity_share_change": float(np.max(np.abs(low_liquidity_change))),
        "peak_mean_mpc_change": float(np.max(np.abs(mean_mpc_change))),
    }


def _trace_filter_metrics(policy_trace: pd.DataFrame) -> dict[str, float]:
    true_columns = sorted(column for column in policy_trace.columns if column.startswith("true_"))
    errors = []
    distribution_errors = []
    for true_column in true_columns:
        state_name = true_column.removeprefix("true_")
        filtered_column = f"filtered_{state_name}"
        if filtered_column not in policy_trace:
            continue
        error = policy_trace[filtered_column].to_numpy(dtype=float) - policy_trace[true_column].to_numpy(dtype=float)
        errors.append(error)
        if state_name in {"low_liquidity_gap", "mean_mpc_gap"}:
            distribution_errors.append(error)
    if not errors:
        return {"mean_state_rmse": 0.0, "distribution_state_rmse": 0.0, "mean_filter_variance": 0.0}
    stacked = np.column_stack(errors)
    distribution_stacked = np.column_stack(distribution_errors) if distribution_errors else np.zeros((len(policy_trace), 1))
    variance_columns = [column for column in policy_trace.columns if column.startswith("filtered_var_")]
    variance_matrix = policy_trace[variance_columns].to_numpy(dtype=float) if variance_columns else np.zeros((len(policy_trace), 1))
    return {
        "mean_state_rmse": float(np.sqrt(np.mean(np.square(stacked)))),
        "distribution_state_rmse": float(np.sqrt(np.mean(np.square(distribution_stacked)))),
        "mean_filter_variance": float(np.mean(variance_matrix)),
    }


def _is_unstable(aggregate_paths: pd.DataFrame) -> bool:
    series = aggregate_paths[["pi_deviation", "output_gap_deviation", "i_deviation", "C_deviation"]].to_numpy(dtype=float)
    if not np.all(np.isfinite(series)):
        return True
    limits = np.array([0.05, 0.10, 0.05, 0.15], dtype=float)
    return bool(np.any(np.abs(series) > limits[None, :]))


def evaluate_policy_run(
    *,
    run: PolicyEpisodeResult,
    reference_run: PolicyEpisodeResult,
    scenario_spec,
    stage4_config,
    steady_state_statistics: dict[str, float],
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    losses = _loss_from_aggregate_paths(
        run.aggregate_paths,
        lambda_y=stage4_config.lambda_y,
        lambda_i=stage4_config.lambda_i,
    )
    reference_losses = _loss_from_aggregate_paths(
        reference_run.aggregate_paths,
        lambda_y=stage4_config.lambda_y,
        lambda_i=stage4_config.lambda_i,
    )
    rate = run.aggregate_paths["i_deviation"].to_numpy(dtype=float)
    reference_rate = reference_run.aggregate_paths["i_deviation"].to_numpy(dtype=float)
    rate_gap = rate - reference_rate

    filter_metrics = _trace_filter_metrics(run.policy_trace)
    unstable = _is_unstable(run.aggregate_paths)
    distributional_peaks = _distributional_peaks(
        group_paths=run.group_paths,
        distribution_stats=run.distribution_stats,
        steady_state_statistics=steady_state_statistics,
    )

    path_frame = pd.DataFrame({
        "variant_name": scenario_spec.variant_name,
        "scenario_name": scenario_spec.scenario_name,
        "scenario_label": scenario_spec.scenario_label,
        "policy_name": run.aggregate_paths["policy_name"].iloc[0],
        "policy_label": run.aggregate_paths["policy_label"].iloc[0],
        "training_seed": run.aggregate_paths["training_seed"].iloc[0],
        "evaluation_seed": run.aggregate_paths["evaluation_seed"].iloc[0],
        "period": run.aggregate_paths["period"].to_numpy(dtype=int),
        "policy_rate": rate,
        "reference_rate": reference_rate,
        "rate_gap": rate_gap,
        "abs_rate_gap": np.abs(rate_gap),
        "policy_loss": losses,
        "reference_loss": reference_losses,
        "cumulative_policy_loss": np.cumsum(losses),
        "cumulative_reference_loss": np.cumsum(reference_losses),
        "inflation_deviation": run.aggregate_paths["pi_deviation"].to_numpy(dtype=float),
        "output_gap_deviation": run.aggregate_paths["output_gap_deviation"].to_numpy(dtype=float),
        "consumption_deviation": run.aggregate_paths["C_deviation"].to_numpy(dtype=float),
        "employment_deviation": run.aggregate_paths["N_deviation"].to_numpy(dtype=float),
        "real_rate_deviation": run.aggregate_paths["r_deviation"].to_numpy(dtype=float),
    })

    group_comparison = compare_distributional_groups(
        scenario_name=scenario_spec.variant_name,
        scenario_label=scenario_spec.scenario_label,
        full_group_paths=reference_run.group_paths,
        filtered_group_paths=run.group_paths,
    )
    group_comparison.insert(0, "evaluation_seed", int(run.aggregate_paths["evaluation_seed"].iloc[0]))
    group_comparison.insert(0, "training_seed", int(run.aggregate_paths["training_seed"].iloc[0]))
    group_comparison.insert(0, "policy_label", run.aggregate_paths["policy_label"].iloc[0])
    group_comparison.insert(0, "policy_name", run.aggregate_paths["policy_name"].iloc[0])

    distributional_summary = build_distributional_summary(
        scenario_name=scenario_spec.variant_name,
        scenario_label=scenario_spec.scenario_label,
        group_comparison=group_comparison,
        distribution_stats_full=reference_run.distribution_stats,
        distribution_stats_filtered=run.distribution_stats,
    )

    metrics = {
        "variant_name": scenario_spec.variant_name,
        "scenario_name": scenario_spec.scenario_name,
        "scenario_label": scenario_spec.scenario_label,
        "policy_name": run.aggregate_paths["policy_name"].iloc[0],
        "policy_label": run.aggregate_paths["policy_label"].iloc[0],
        "training_seed": int(run.aggregate_paths["training_seed"].iloc[0]),
        "evaluation_seed": int(run.aggregate_paths["evaluation_seed"].iloc[0]),
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
        "mean_state_rmse": filter_metrics["mean_state_rmse"],
        "distribution_state_rmse": filter_metrics["distribution_state_rmse"],
        "mean_filter_variance": filter_metrics["mean_filter_variance"],
        "unstable": int(unstable),
        "impact_inflation_pp": float(100.0 * run.aggregate_paths["pi_deviation"].iloc[0]),
        "impact_output_gap_pct": float(100.0 * run.aggregate_paths["output_gap_deviation"].iloc[0]),
        "impact_consumption_pct": float(run.aggregate_paths["C_pct"].iloc[0]),
        "impact_nominal_rate_pp": float(100.0 * run.aggregate_paths["i_deviation"].iloc[0]),
        "impact_real_rate_pp": float(100.0 * run.aggregate_paths["r_deviation"].iloc[0]),
        "impact_employment_pct": float(run.aggregate_paths["N_pct"].iloc[0]),
        **distributional_peaks,
        "peak_consumption_q1_gap_vs_full_information": float(distributional_summary["peak_consumption_q1_gap_vs_full_information"]),
        "peak_consumption_q5_gap_vs_full_information": float(distributional_summary["peak_consumption_q5_gap_vs_full_information"]),
        "peak_low_liquidity_share_difference_vs_full_information": float(distributional_summary["peak_low_liquidity_share_difference"]),
        "peak_mean_mpc_difference_vs_full_information": float(distributional_summary["peak_mean_mpc_difference"]),
    }
    return metrics, path_frame, group_comparison


def summarize_training_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=[
            "variant_name",
            "training_seed",
            "best_validation_return",
            "final_validation_return",
            "final_mean_episode_return",
        ])
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
        scenario_name = frame["scenario_name"].iloc[0]
        scenario_label = frame["scenario_label"].iloc[0]
        by_policy = {name: sub.iloc[0] for name, sub in frame.groupby("policy_name")}
        if "classical_filtered_rule" not in by_policy or "learning_policy" not in by_policy:
            continue
        classical = by_policy["classical_filtered_rule"]
        rl = by_policy["learning_policy"]
        full_info = by_policy.get("full_information_rule", classical)
        rows.append({
            "variant_name": variant_name,
            "scenario_name": scenario_name,
            "scenario_label": scenario_label,
            "classical_mean_policy_loss": float(classical["mean_policy_loss"]),
            "rl_mean_policy_loss": float(rl["mean_policy_loss"]),
            "full_information_mean_policy_loss": float(full_info["mean_policy_loss"]),
            "delta_mean_policy_loss_rl_minus_classical": float(rl["mean_policy_loss"] - classical["mean_policy_loss"]),
            "delta_cumulative_policy_loss_rl_minus_classical": float(rl["cumulative_policy_loss"] - classical["cumulative_policy_loss"]),
            "classical_policy_rate_rmse": float(classical["policy_rate_rmse"]),
            "rl_policy_rate_rmse": float(rl["policy_rate_rmse"]),
            "classical_mean_abs_rate_gap": float(classical["mean_abs_rate_gap"]),
            "rl_mean_abs_rate_gap": float(rl["mean_abs_rate_gap"]),
            "classical_peak_consumption_q1": float(classical["peak_consumption_q1"]),
            "rl_peak_consumption_q1": float(rl["peak_consumption_q1"]),
            "classical_peak_mean_mpc_change": float(classical["peak_mean_mpc_change"]),
            "rl_peak_mean_mpc_change": float(rl["peak_mean_mpc_change"]),
            "classical_unstable": int(classical["unstable"]),
            "rl_unstable": int(rl["unstable"]),
        })
    return pd.DataFrame(rows)
