from __future__ import annotations

import numpy as np
import pandas as pd

from .calibration import calibration_table_metadata
from .distribution import build_group_masks, stationary_distribution
from .grids import state_mesh, weighted_quantile
from .labels import pretty_channel_label, pretty_group_label


def calibration_table(config, solved_values=None):
    values = config.__dict__
    solved_values = {} if solved_values is None else solved_values
    rows = []
    aliases = {
        "beta": solved_values.get("beta", config.beta_guess),
        "chi1": solved_values.get("chi1", config.chi1_guess),
    }
    for meta in calibration_table_metadata():
        parameter = meta["parameter"]
        value = aliases[parameter] if parameter in aliases else values[parameter]
        rows.append({
            "параметр": meta["label"],
            "значение": value,
            "интерпретация": meta["description"],
            "источник или комментарий": meta["source"],
        })
    return pd.DataFrame(rows)


def policy_rule_table(scenarios):
    rows = []
    for scenario in scenarios:
        cfg = scenario["config"]
        rows.append({
            "сценарий": scenario["name"],
            "обозначение": scenario["label"],
            "phi_pi": cfg.phi_pi,
            "phi_y": cfg.phi_y,
            "rho_i": cfg.rho_i,
            "размер шока": cfg.mp_shock_size,
            "персистентность шока": cfg.mp_shock_persistence,
            "экономический смысл": scenario["description"],
        })
    return pd.DataFrame(rows)


def steady_state_moments_table(ss, mpc, config):
    D = stationary_distribution(ss)
    mesh = state_mesh(ss)
    groups = build_group_masks(ss, config)["groups"]
    low_liq_share = float(np.sum(D * groups["low_liquid"]))
    wealthy_htm_share = float(np.sum(D * groups["wealthy_htm"]))
    median_b = weighted_quantile(mesh["b"], D, 0.5)
    median_a = weighted_quantile(mesh["a"], D, 0.5)
    avg_mpc = float(np.sum(D * mpc))
    median_mpc = weighted_quantile(mpc, D, 0.5)
    share_high_mpc = float(np.sum(D * (mpc > 0.2)))

    rows = [
        {"показатель": "Выпуск", "значение": float(ss["Y"])},
        {"показатель": "Инфляция", "значение": float(ss["pi"])},
        {"показатель": "Номинальная ставка", "значение": float(ss["i"])},
        {"показатель": "Среднее потребление", "значение": float(ss["C"])},
        {"показатель": "Доля домохозяйств с низкой ликвидностью", "значение": low_liq_share},
        {"показатель": "Медианное ликвидное богатство", "значение": median_b},
        {"показатель": "Медианное неликвидное богатство", "значение": median_a},
        {"показатель": "Средняя MPC", "значение": avg_mpc},
        {"показатель": "Медианная MPC", "значение": median_mpc},
        {"показатель": "Доля домохозяйств с MPC выше 0.2", "значение": share_high_mpc},
        {"показатель": "Доля состоятельных домохозяйств с низкой ликвидностью (WHtM)", "значение": wealthy_htm_share},
    ]
    return pd.DataFrame(rows)


def _half_life(series):
    series = np.asarray(series)
    peak = np.max(np.abs(series))
    if peak == 0:
        return 0
    threshold = 0.5 * peak
    peak_idx = int(np.argmax(np.abs(series)))
    tail = np.abs(series[peak_idx:])
    below = np.where(tail <= threshold)[0]
    if len(below) == 0:
        return len(series) - 1 - peak_idx
    return int(below[0])


def shock_effects_table(aggregate_irf, scenario_name="baseline"):
    rows = []
    labels = {
        "pi": "Инфляция",
        "Y": "Выпуск",
        "C": "Потребление",
        "N": "Занятость",
        "i": "Номинальная ставка",
    }
    aggregate_irf = aggregate_irf[aggregate_irf["scenario"] == scenario_name]
    for variable, label in labels.items():
        subset = aggregate_irf.loc[aggregate_irf["variable"] == variable].sort_values("period")
        series = subset["value"].to_numpy()
        min_idx = int(np.argmin(series))
        max_idx = int(np.argmax(series))
        rows.append({
            "переменная": label,
            "отклик при ударе": float(series[0]),
            "минимум отклика": float(series[min_idx]),
            "период минимума": min_idx,
            "максимум отклика": float(series[max_idx]),
            "период максимума": max_idx,
            "полупериод затухания": _half_life(series),
            "накопленный отклик": float(series.sum()),
        })
    return pd.DataFrame(rows)


def group_differences_table(group_stats, group_paths, scenario_name="baseline"):
    rows = []
    key_groups = {"low_liquid", "wealthy_htm", "high_liquid", "mpc_low", "mpc_mid", "mpc_high"}
    group_paths = group_paths[group_paths["scenario"] == scenario_name]
    for _, stats in group_stats.iterrows():
        group = stats["group"]
        if group not in key_groups:
            continue
        subset = group_paths[group_paths["group"] == group].sort_values("period")
        if subset.empty:
            continue
        cons = subset["consumption_pct_deviation"].to_numpy()
        income = 100.0 * (subset["mean_disposable_income"].to_numpy() / subset["mean_disposable_income"].iloc[0] - 1.0)
        liquid = subset["mean_liquid_assets"].to_numpy()
        min_idx = int(np.argmin(cons))
        max_idx = int(np.argmax(cons))
        integral_response = float(cons.sum())
        peak_income = float(income[np.argmax(np.abs(income))])
        liquid_change = float(liquid[min_idx] - liquid[0])
        rows.append({
            "код группы": group,
            "группа": pretty_group_label(group),
            "отклик при ударе": float(cons[0]),
            "минимум отклика потребления": float(cons[min_idx]),
            "период минимума": min_idx,
            "максимум отклика потребления": float(cons[max_idx]),
            "период максимума": max_idx,
            "интегральный отклик потребления": integral_response,
            "пик отклика дохода": peak_income,
            "изменение ликвидного богатства к периоду минимума": liquid_change,
        })
    return pd.DataFrame(rows)


def channel_summary_table(channel_decomposition, scenario_name="baseline"):
    rows = []
    subset = channel_decomposition[channel_decomposition["scenario"] == scenario_name]
    for component in ["intertemporal_financial_channel", "labor_income_channel", "redistribution_liquidity_residual"]:
        comp = subset[subset["component"] == component].sort_values("period")
        series = comp["value"].to_numpy()
        rows.append({
            "код канала": component,
            "канал": pretty_channel_label(component),
            "вклад в пик отклика потребления": float(series[np.argmax(np.abs(series))]),
            "вклад в интегральный отклик": float(series.sum()),
        })
    return pd.DataFrame(rows)
