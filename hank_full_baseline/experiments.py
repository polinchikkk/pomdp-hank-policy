from __future__ import annotations

import numpy as np
from dataclasses import replace


def _ar1_shock_path(size, period, persistence, horizon):
    shock = np.zeros(horizon)
    shock[period] = size
    for t in range(period + 1, horizon):
        shock[t] = persistence * shock[t - 1]
    return shock


def monetary_policy_experiment(config):
    shock = _ar1_shock_path(
        size=config.mp_shock_size,
        period=config.shock_period,
        persistence=config.mp_shock_persistence,
        horizon=config.shock_T,
    )
    return {
        "name": "monetary_policy_shock",
        "description": "Неожиданный положительный шок к правилу процентной ставки через monetary_policy_shock.",
        "inputs": {"monetary_policy_shock": shock},
        "horizons": tuple(h for h in (0, 1, 4, 8, 16, 24, 40) if h < config.shock_T),
        "shock_size": config.mp_shock_size,
        "shock_persistence": config.mp_shock_persistence,
    }


def policy_scenarios(config):
    return [
        {
            "name": "baseline",
            "label": "Базовое правило",
            "description": "Стандартное правило Тейлора с базовой реакцией на инфляцию и выпуск.",
            "config": config,
        },
        {
            "name": "high_phi_pi",
            "label": "Сильная реакция на инфляцию",
            "description": "Более агрессивное правило Тейлора по инфляции.",
            "config": replace(config, phi_pi=2.0),
        },
        {
            "name": "high_rho_i",
            "label": "Высокая инерция ставки",
            "description": "Правило с более высокой инерционностью процентной ставки.",
            "config": replace(config, rho_i=0.9),
        },
        {
            "name": "low_phi_y",
            "label": "Слабая реакция на выпуск",
            "description": "Правило с отключенной или почти нулевой реакцией на выпуск.",
            "config": replace(config, phi_y=0.0),
        },
        {
            "name": "persistent_shock",
            "label": "Более персистентный монетарный шок",
            "description": "То же правило ставки, но с более персистентным policy shock.",
            "config": replace(config, mp_shock_persistence=0.6),
        },
    ]


def household_robustness_scenarios(config):
    return [
        {
            "name": "baseline",
            "label": "Базовая калибровка",
            "description": "Исходная калибровка HANK baseline.",
            "config": config,
        },
        {
            "name": "lower_chi0",
            "label": "Меньший сдвиг в ребалансировочных издержках",
            "description": "Более низкий chi0 усиливает локальные adjustment costs near zero и может сделать constrained-region household block острее.",
            "config": replace(config, chi0=0.25),
        },
        {
            "name": "high_omega",
            "label": "Ещё больший клин liquid-rate",
            "description": "Ещё более высокий спрэд между доходностью ликвидного актива и ставкой политики поверх нового baseline.",
            "config": replace(config, omega=0.017),
        },
        {
            "name": "high_sigma_z",
            "label": "Более высокий доходный риск",
            "description": "Более высокая безусловная дисперсия лог-идосинкратического дохода усиливает precautionary motive и cross-sectional heterogeneity.",
            "config": replace(config, sigma_z=0.92),
        },
        {
            "name": "stronger_household_combo",
            "label": "Комбинация более жёсткого household block",
            "description": "Одновременное снижение chi0, повышение liquid-rate wedge и усиление доходного риска как кандидат на более сильную HANK-калибровку.",
            "config": replace(config, chi0=0.25, omega=0.016, sigma_z=0.92),
        },
    ]
