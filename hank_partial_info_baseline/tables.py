from __future__ import annotations

import pandas as pd


def information_scenarios_table(scenario_spec: list[dict]) -> pd.DataFrame:
    rows = []
    for entry in scenario_spec:
        rows.append({
            "сценарий": entry["scenario"],
            "информационный режим": entry.get("information_regime", ""),
            "наблюдаемые переменные": entry["observed_variables"],
            "масштаб шума": entry["noise_scale"],
            "используется распределительная статистика": "да" if entry["uses_distribution_stats"] else "нет",
            "лаг распределительных сигналов": (
                "нет"
                if entry.get("distribution_signal_lag") is None
                else str(entry["distribution_signal_lag"])
            ),
        })
    return pd.DataFrame(rows)


def filter_quality_table(filter_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "scenario_label": "сценарий",
        "mean_state_rmse": "средний RMSE состояния",
        "distribution_factor_rmse": "RMSE распределительного фактора",
        "log_likelihood": "логарифм правдоподобия",
    }
    return filter_metrics[list(columns)].rename(columns=columns)


def policy_quality_table(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "scenario_label": "сценарий",
        "mean_policy_loss": "средняя потеря",
        "cumulative_policy_loss": "накопленная потеря",
        "policy_rate_rmse": "RMSE ставки",
        "cumulative_excess_loss": "дополнительная потеря относительно полной информации",
    }
    return policy_metrics[list(columns)].rename(columns=columns)


def distributional_consequences_table(distribution_summary: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "scenario_label": "сценарий",
        "peak_consumption_q1_filtered": "пик отклика потребления нижнего квантиля",
        "peak_consumption_q5_filtered": "пик отклика потребления верхнего квантиля",
        "peak_low_liquidity_share_difference": "изменение доли low-liquid households",
        "peak_mean_mpc_difference": "изменение средней MPC",
    }
    return distribution_summary[list(columns)].rename(columns=columns)
