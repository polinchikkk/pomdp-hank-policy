from __future__ import annotations


GROUP_LABELS = {
    "low_liquid": "Низкая ликвидность",
    "low_liquid_non_whtm": "Низкая ликвидность без WHtM",
    "wealthy_htm": "Состоятельные с низкой ликвидностью (WHtM)",
    "high_liquid": "Высоколиквидные",
    "other_households": "Остальные домохозяйства",
    "high_total_wealth": "Высокое совокупное богатство",
    "mpc_low": "Низкая MPC",
    "mpc_mid": "Средняя MPC",
    "mpc_high": "Высокая MPC",
    "income_low": "Низкий доход",
    "income_high": "Высокий доход",
}


CHANNEL_LABELS = {
    "intertemporal_financial_channel": "Межвременной и финансовый канал",
    "labor_income_channel": "Канал трудового дохода",
    "redistribution_liquidity_residual": "Остаточный канал перераспределения и ликвидности",
    "household_total": "Отклик household-блока",
    "general_equilibrium_total": "Полный отклик",
}


ROBUSTNESS_SCENARIO_LABELS = {
    "baseline": "Базовая калибровка",
    "lower_chi0": "Меньший сдвиг в ребалансировочных издержках",
    "high_omega": "Ещё больший клин liquid-rate",
    "high_sigma_z": "Более высокий доходный риск",
    "stronger_household_combo": "Комбинация более жёсткого household block",
    "persistent_mp_shock": "Более персистентный монетарный шок",
}


def pretty_group_label(group_name: str) -> str:
    if group_name in GROUP_LABELS:
        return GROUP_LABELS[group_name]
    if group_name.startswith("liquid_q"):
        idx = group_name.removeprefix("liquid_q")
        return f"{idx}-й квантиль ликвидного богатства"
    if group_name.startswith("illiquid_q"):
        idx = group_name.removeprefix("illiquid_q")
        return f"{idx}-й квантиль неликвидного богатства"
    if group_name.startswith("wealth_q"):
        idx = group_name.removeprefix("wealth_q")
        return f"{idx}-й квантиль совокупного богатства"
    if group_name.startswith("income_mid_"):
        idx = group_name.removeprefix("income_mid_")
        return f"Средний доход {idx}"
    return group_name


def pretty_channel_label(channel_name: str) -> str:
    return CHANNEL_LABELS.get(channel_name, channel_name)


def pretty_robustness_label(scenario_name: str) -> str:
    return ROBUSTNESS_SCENARIO_LABELS.get(scenario_name, scenario_name)
