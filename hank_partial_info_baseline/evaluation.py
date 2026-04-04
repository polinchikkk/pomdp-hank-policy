from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .state_space import ScenarioSimulation


def _safe_correlation(actual: np.ndarray, estimated: np.ndarray) -> float:
    if actual.std() <= 0.0 or estimated.std() <= 0.0:
        return 1.0
    return float(np.corrcoef(actual, estimated)[0, 1])


def evaluate_filter_metrics(
    simulation: ScenarioSimulation,
    state_names: tuple[str, ...],
    distribution_state_names: tuple[str, ...],
    confidence_scale: float,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    true_states = simulation.true_states
    filtered_states = simulation.filtered_states
    variances = np.maximum(
        np.diagonal(simulation.filtered_covariances, axis1=1, axis2=2),
        0.0,
    )
    stds = np.sqrt(variances)
    lower = filtered_states - confidence_scale * stds
    upper = filtered_states + confidence_scale * stds
    errors = filtered_states - true_states

    rows = []
    distribution_indices = [state_names.index(name) for name in distribution_state_names]
    for index, name in enumerate(state_names):
        rows.append({
            "scenario": simulation.scenario_name,
            "scenario_label": simulation.scenario_label,
            "state": name,
            "rmse": float(np.sqrt(np.mean(np.square(errors[:, index])))),
            "mae": float(np.mean(np.abs(errors[:, index]))),
            "correlation": _safe_correlation(true_states[:, index], filtered_states[:, index]),
            "mean_std": float(np.mean(stds[:, index])),
            "coverage": float(np.mean((true_states[:, index] >= lower[:, index]) & (true_states[:, index] <= upper[:, index]))),
        })

    state_frame = pd.DataFrame(rows)
    distribution_rmse = float(np.sqrt(np.mean(np.square(errors[:, distribution_indices]))))
    diagnostics = {
        "scenario": simulation.scenario_name,
        "scenario_label": simulation.scenario_label,
        "mean_state_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_state_mae": float(np.mean(np.abs(errors))),
        "distribution_factor_rmse": distribution_rmse,
        "mean_filter_std": float(np.mean(stds)),
        "mean_coverage": float(np.mean((true_states >= lower) & (true_states <= upper))),
        "log_likelihood": float(simulation.filter_results.log_likelihood),
    }
    return diagnostics, state_frame


def evaluate_policy_metrics(
    *,
    scenario_name: str,
    scenario_label: str,
    full_aggregate_paths: pd.DataFrame,
    filtered_aggregate_paths: pd.DataFrame,
    full_distribution_stats: pd.DataFrame,
    filtered_distribution_stats: pd.DataFrame,
    lambda_y: float,
    lambda_i: float,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    full = full_aggregate_paths.sort_values("period").reset_index(drop=True)
    filt = filtered_aggregate_paths.sort_values("period").reset_index(drop=True)
    full_dist = full_distribution_stats.sort_values("period").reset_index(drop=True)
    filt_dist = filtered_distribution_stats.sort_values("period").reset_index(drop=True)

    full_i = full["i_deviation"].to_numpy(dtype=float)
    filt_i = filt["i_deviation"].to_numpy(dtype=float)
    full_pi = full["pi_deviation"].to_numpy(dtype=float)
    filt_pi = filt["pi_deviation"].to_numpy(dtype=float)
    full_y = full["output_gap_deviation"].to_numpy(dtype=float)
    filt_y = filt["output_gap_deviation"].to_numpy(dtype=float)

    full_loss = np.square(full_pi) + lambda_y * np.square(full_y) + lambda_i * np.square(np.diff(full_i, prepend=0.0))
    filt_loss = np.square(filt_pi) + lambda_y * np.square(filt_y) + lambda_i * np.square(np.diff(filt_i, prepend=0.0))
    excess_loss = filt_loss - full_loss
    rate_gap = filt_i - full_i

    path_frame = pd.DataFrame({
        "scenario": scenario_name,
        "scenario_label": scenario_label,
        "period": full["period"].to_numpy(dtype=int),
        "full_information_rate": full_i,
        "filtered_rate": filt_i,
        "rate_gap": rate_gap,
        "abs_rate_gap": np.abs(rate_gap),
        "full_loss": full_loss,
        "filtered_loss": filt_loss,
        "excess_loss": excess_loss,
        "full_pi": full_pi,
        "filtered_pi": filt_pi,
        "full_output_gap": full_y,
        "filtered_output_gap": filt_y,
        "full_consumption": full["C_deviation"].to_numpy(dtype=float),
        "filtered_consumption": filt["C_deviation"].to_numpy(dtype=float),
        "full_low_liquidity_share": full_dist["share_low_liquidity"].to_numpy(dtype=float),
        "filtered_low_liquidity_share": filt_dist["share_low_liquidity"].to_numpy(dtype=float),
        "full_mean_mpc": full_dist["mean_mpc"].to_numpy(dtype=float),
        "filtered_mean_mpc": filt_dist["mean_mpc"].to_numpy(dtype=float),
    })

    metrics = {
        "scenario": scenario_name,
        "scenario_label": scenario_label,
        "mean_policy_loss": float(np.mean(filt_loss)),
        "cumulative_policy_loss": float(np.sum(filt_loss)),
        "mean_full_information_loss": float(np.mean(full_loss)),
        "cumulative_full_information_loss": float(np.sum(full_loss)),
        "mean_excess_loss": float(np.mean(excess_loss)),
        "cumulative_excess_loss": float(np.sum(excess_loss)),
        "mean_abs_rate_gap": float(np.mean(np.abs(rate_gap))),
        "policy_rate_rmse": float(np.sqrt(np.mean(np.square(rate_gap)))),
        "mean_abs_inflation_gap": float(np.mean(np.abs(filt_pi - full_pi))),
        "mean_abs_output_gap_difference": float(np.mean(np.abs(filt_y - full_y))),
        "mean_abs_consumption_difference": float(np.mean(np.abs(
            filt["C_deviation"].to_numpy(dtype=float) - full["C_deviation"].to_numpy(dtype=float)
        ))),
        "peak_low_liquidity_share_difference": float(np.max(np.abs(
            filt_dist["share_low_liquidity"].to_numpy(dtype=float) - full_dist["share_low_liquidity"].to_numpy(dtype=float)
        ))),
        "peak_mean_mpc_difference": float(np.max(np.abs(
            filt_dist["mean_mpc"].to_numpy(dtype=float) - full_dist["mean_mpc"].to_numpy(dtype=float)
        ))),
    }
    return metrics, path_frame


def compare_distributional_groups(
    *,
    scenario_name: str,
    scenario_label: str,
    full_group_paths: pd.DataFrame,
    filtered_group_paths: pd.DataFrame,
) -> pd.DataFrame:
    full = full_group_paths.copy()
    full["policy_mode"] = "full_information"
    filt = filtered_group_paths.copy()
    filt["policy_mode"] = "filtered_information"
    merged = full.merge(
        filt,
        on=["period", "grouping", "group"],
        how="inner",
        suffixes=("_full", "_filtered"),
    )
    merged["scenario"] = scenario_name
    merged["scenario_label"] = scenario_label
    merged["consumption_pct_gap"] = (
        merged["consumption_pct_deviation_filtered"] - merged["consumption_pct_deviation_full"]
    )
    merged["mean_liquid_assets_gap"] = merged["mean_liquid_assets_filtered"] - merged["mean_liquid_assets_full"]
    merged["mean_illiquid_assets_gap"] = merged["mean_illiquid_assets_filtered"] - merged["mean_illiquid_assets_full"]
    return merged


def build_distributional_summary(
    scenario_name: str,
    scenario_label: str,
    group_comparison: pd.DataFrame,
    distribution_stats_full: pd.DataFrame,
    distribution_stats_filtered: pd.DataFrame,
) -> dict[str, float | str]:
    liquid_groups = group_comparison[group_comparison["grouping"] == "liquid_wealth_quantile"]
    liquid_q1 = liquid_groups[liquid_groups["group"] == "liquid_q1"]
    liquid_q5 = liquid_groups[liquid_groups["group"] == "liquid_q5"]

    return {
        "scenario": scenario_name,
        "scenario_label": scenario_label,
        "peak_consumption_q1_filtered": float(liquid_q1["consumption_pct_deviation_filtered"].min()) if not liquid_q1.empty else 0.0,
        "peak_consumption_q5_filtered": float(liquid_q5["consumption_pct_deviation_filtered"].min()) if not liquid_q5.empty else 0.0,
        "peak_consumption_q1_gap_vs_full_information": float(np.max(np.abs(liquid_q1["consumption_pct_gap"]))) if not liquid_q1.empty else 0.0,
        "peak_consumption_q5_gap_vs_full_information": float(np.max(np.abs(liquid_q5["consumption_pct_gap"]))) if not liquid_q5.empty else 0.0,
        "peak_low_liquidity_share_difference": float(np.max(np.abs(
            distribution_stats_filtered["share_low_liquidity"].to_numpy(dtype=float)
            - distribution_stats_full["share_low_liquidity"].to_numpy(dtype=float)
        ))),
        "peak_mean_mpc_difference": float(np.max(np.abs(
            distribution_stats_filtered["mean_mpc"].to_numpy(dtype=float)
            - distribution_stats_full["mean_mpc"].to_numpy(dtype=float)
        ))),
    }


def scenario_metric_frame(records: list[Mapping[str, float | str]]) -> pd.DataFrame:
    return pd.DataFrame(list(records))
