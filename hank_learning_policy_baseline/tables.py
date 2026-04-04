from __future__ import annotations

import pandas as pd


def policy_performance_table(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "scenario_label": "сценарий",
        "policy_label": "политика",
        "mean_policy_loss": "средняя потеря",
        "cumulative_policy_loss": "накопленная потеря",
        "policy_rate_rmse": "RMSE ставки",
        "mean_abs_rate_gap": "среднее абсолютное отклонение ставки",
        "policy_instrument_volatility": "волатильность инструмента",
        "unstable": "нестабильность",
    }
    return policy_metrics[list(columns)].rename(columns=columns)


def macro_summary_table(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "scenario_label": "сценарий",
        "policy_label": "политика",
        "impact_inflation_pp": "impact инфляции, п.п.",
        "impact_output_gap_pct": "impact разрыва выпуска, %",
        "impact_consumption_pct": "impact потребления, %",
        "impact_nominal_rate_pp": "impact ставки, п.п.",
        "impact_employment_pct": "impact занятости, %",
    }
    return policy_metrics[list(columns)].rename(columns=columns)


def distributional_summary_table(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "scenario_label": "сценарий",
        "policy_label": "политика",
        "peak_consumption_q1": "пик отклика потребления нижнего квантиля",
        "peak_consumption_q5": "пик отклика потребления верхнего квантиля",
        "peak_low_liquidity_share_change": "пик изменения доли low-liquid",
        "peak_mean_mpc_change": "пик изменения средней MPC",
    }
    return policy_metrics[list(columns)].rename(columns=columns)


def ablation_table(policy_comparison: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "scenario_label": "сценарий",
        "classical_mean_policy_loss": "classical средняя потеря",
        "rl_mean_policy_loss": "RL средняя потеря",
        "delta_mean_policy_loss_rl_minus_classical": "RL минус classical",
        "classical_policy_rate_rmse": "classical RMSE ставки",
        "rl_policy_rate_rmse": "RL RMSE ставки",
    }
    return policy_comparison[list(columns)].rename(columns=columns)
