from __future__ import annotations

import numpy as np
import pandas as pd

from .regime_simulation import RegimePolicyRun


def _safe_correlation(actual: np.ndarray, estimated: np.ndarray) -> float:
    if actual.std() <= 0.0 or estimated.std() <= 0.0:
        return 1.0
    return float(np.corrcoef(actual, estimated)[0, 1])


def _mean_detection_delay(hidden_regimes: np.ndarray, stress_probabilities: np.ndarray, threshold: float = 0.5) -> float:
    switch_points = np.where((hidden_regimes[1:] == 1) & (hidden_regimes[:-1] == 0))[0] + 1
    if len(switch_points) == 0:
        return 0.0
    delays = []
    for switch_point in switch_points:
        detection = np.where(stress_probabilities[switch_point:] >= threshold)[0]
        delays.append(float(detection[0]) if len(detection) else float(len(stress_probabilities) - switch_point))
    return float(np.mean(delays))


def evaluate_regime_filter(filtered_run: RegimePolicyRun) -> tuple[dict[str, float | str], pd.DataFrame]:
    if filtered_run.filtered_states is None or filtered_run.filtered_mode_probabilities is None or filtered_run.filter_results is None:
        raise ValueError("Filtered run is required for regime filter evaluation.")

    true_states = filtered_run.true_states
    filtered_states = filtered_run.filtered_states
    filtered_covariances = filtered_run.filtered_covariances
    state_names = (
        "rstar_gap",
        "productivity_gap",
        "fiscal_gap",
        "inflation_gap",
        "output_gap",
        "low_liquidity_gap",
        "mean_mpc_gap",
    )
    variances = np.maximum(np.diagonal(filtered_covariances, axis1=1, axis2=2), 0.0)
    errors = filtered_states - true_states
    stress_probabilities = filtered_run.filtered_mode_probabilities[:, 1]
    hidden_regimes = filtered_run.hidden_regimes
    regime_predictions = np.argmax(filtered_run.filtered_mode_probabilities, axis=1)

    rows = []
    for state_index, state_name in enumerate(state_names):
        rows.append({
            "scenario": filtered_run.scenario_name,
            "scenario_label": filtered_run.scenario_label,
            "state": state_name,
            "rmse": float(np.sqrt(np.mean(np.square(errors[:, state_index])))),
            "mae": float(np.mean(np.abs(errors[:, state_index]))),
            "correlation": _safe_correlation(true_states[:, state_index], filtered_states[:, state_index]),
            "mean_std": float(np.mean(np.sqrt(variances[:, state_index]))),
        })
    rows.append({
        "scenario": filtered_run.scenario_name,
        "scenario_label": filtered_run.scenario_label,
        "state": "stress_probability",
        "rmse": float(np.sqrt(np.mean(np.square(stress_probabilities - (hidden_regimes == 1).astype(float))))),
        "mae": float(np.mean(np.abs(stress_probabilities - (hidden_regimes == 1).astype(float)))),
        "correlation": _safe_correlation((hidden_regimes == 1).astype(float), stress_probabilities),
        "mean_std": float(np.mean(stress_probabilities * (1.0 - stress_probabilities))),
    })
    state_frame = pd.DataFrame(rows)
    metrics = {
        "scenario": filtered_run.scenario_name,
        "scenario_label": filtered_run.scenario_label,
        "mean_state_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "distribution_state_rmse": float(np.sqrt(np.mean(np.square(errors[:, -2:])))),
        "regime_accuracy": float(np.mean(regime_predictions == hidden_regimes)),
        "stress_brier_score": float(np.mean(np.square(stress_probabilities - (hidden_regimes == 1).astype(float)))),
        "mean_stress_probability": float(np.mean(stress_probabilities)),
        "mean_stress_probability_in_stress": float(np.mean(stress_probabilities[hidden_regimes == 1])) if np.any(hidden_regimes == 1) else 0.0,
        "mean_stress_probability_in_normal": float(np.mean(stress_probabilities[hidden_regimes == 0])) if np.any(hidden_regimes == 0) else 0.0,
        "mean_detection_delay": _mean_detection_delay(hidden_regimes, stress_probabilities),
        "log_likelihood": float(filtered_run.filter_results.log_likelihood),
    }
    return metrics, state_frame


def evaluate_policy_under_regime_uncertainty(
    *,
    filtered_run: RegimePolicyRun,
    full_information_run: RegimePolicyRun,
    lambda_y: float,
    lambda_i: float,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    idx_pi = 3
    idx_output = 4
    idx_low_liq = 5
    idx_mean_mpc = 6

    full_pi = full_information_run.true_states[:, idx_pi]
    full_output = full_information_run.true_states[:, idx_output]
    filt_pi = filtered_run.true_states[:, idx_pi]
    filt_output = filtered_run.true_states[:, idx_output]

    full_loss = np.square(full_pi) + lambda_y * np.square(full_output) + lambda_i * np.square(np.diff(full_information_run.policy_rate, prepend=0.0))
    filt_loss = np.square(filt_pi) + lambda_y * np.square(filt_output) + lambda_i * np.square(np.diff(filtered_run.policy_rate, prepend=0.0))
    rate_gap = filtered_run.policy_rate - full_information_run.policy_rate

    all_obs_names = list(filtered_run.noisy_observation_names)
    path_frame = pd.DataFrame({
        "scenario": filtered_run.scenario_name,
        "scenario_label": filtered_run.scenario_label,
        "period": np.arange(len(filtered_run.policy_rate), dtype=int),
        "hidden_regime": filtered_run.hidden_regimes,
        "full_information_rate": full_information_run.policy_rate,
        "filtered_rate": filtered_run.policy_rate,
        "rate_gap": rate_gap,
        "abs_rate_gap": np.abs(rate_gap),
        "full_policy_loss": full_loss,
        "filtered_policy_loss": filt_loss,
        "excess_loss": filt_loss - full_loss,
        "full_inflation_gap": full_pi,
        "filtered_inflation_gap": filt_pi,
        "full_output_gap": full_output,
        "filtered_output_gap": filt_output,
        "full_low_liquidity_gap": full_information_run.true_states[:, idx_low_liq],
        "filtered_low_liquidity_gap": filtered_run.true_states[:, idx_low_liq],
        "full_mean_mpc_gap": full_information_run.true_states[:, idx_mean_mpc],
        "filtered_mean_mpc_gap": filtered_run.true_states[:, idx_mean_mpc],
    })
    if filtered_run.filtered_mode_probabilities is not None:
        path_frame["stress_probability"] = filtered_run.filtered_mode_probabilities[:, 1]

    metrics = {
        "scenario": filtered_run.scenario_name,
        "scenario_label": filtered_run.scenario_label,
        "mean_policy_loss": float(np.mean(filt_loss)),
        "cumulative_policy_loss": float(np.sum(filt_loss)),
        "mean_full_information_loss": float(np.mean(full_loss)),
        "cumulative_full_information_loss": float(np.sum(full_loss)),
        "delta_mean_policy_loss_filtered_minus_full_information": float(np.mean(filt_loss) - np.mean(full_loss)),
        "delta_cumulative_policy_loss_filtered_minus_full_information": float(np.sum(filt_loss) - np.sum(full_loss)),
        "mean_abs_rate_gap": float(np.mean(np.abs(rate_gap))),
        "policy_rate_rmse": float(np.sqrt(np.mean(np.square(rate_gap)))),
        "policy_instrument_volatility": float(np.std(filtered_run.policy_rate)),
        "impact_nominal_rate_pp": float(100.0 * filtered_run.policy_rate[0]),
        "impact_inflation_pp": float(100.0 * filt_pi[0]),
        "impact_output_gap_pct": float(100.0 * filt_output[0]),
        "impact_low_liquidity_gap_pp": float(100.0 * filtered_run.true_states[0, idx_low_liq]),
        "impact_mean_mpc_gap_pp": float(100.0 * filtered_run.true_states[0, idx_mean_mpc]),
        "peak_low_liquidity_gap_difference": float(np.max(np.abs(
            filtered_run.true_states[:, idx_low_liq] - full_information_run.true_states[:, idx_low_liq]
        ))),
        "peak_mean_mpc_gap_difference": float(np.max(np.abs(
            filtered_run.true_states[:, idx_mean_mpc] - full_information_run.true_states[:, idx_mean_mpc]
        ))),
    }
    return metrics, path_frame
