from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .calibration import sequence_jacobian_tutorial_calibration
from .distribution import build_group_masks, household_path_levels, stationary_distribution
from .experiments import household_robustness_scenarios, monetary_policy_experiment
from .grids import state_mesh, weighted_quantile
from .household_solver import compute_mpc
from .irfs import group_consumption_irfs
from .labels import pretty_group_label, pretty_robustness_label
from .steady_state import solve_steady_state
from .transition import channel_decomposition, solve_transition


def mpc_moments_frame(ss, mpc, config):
    D = stationary_distribution(ss)
    groups = build_group_masks(ss, config, mpc=mpc)["groups"]
    rows = [
        {"категория": "Распределение MPC", "момент": "Средняя MPC", "значение": float(np.sum(D * mpc))},
        {"категория": "Распределение MPC", "момент": "Медианная MPC", "значение": weighted_quantile(mpc, D, 0.5)},
        {"категория": "Распределение MPC", "момент": "Доля MPC > 0.2", "значение": float(np.sum(D * (mpc > 0.2)))},
        {"категория": "Распределение MPC", "момент": "Доля MPC > 0.3", "значение": float(np.sum(D * (mpc > 0.3)))},
        {"категория": "Распределение MPC", "момент": "Доля MPC > 0.5", "значение": float(np.sum(D * (mpc > 0.5)))},
    ]

    for group_name in ["low_liquid", "wealthy_htm", "high_liquid", "mpc_low", "mpc_mid", "mpc_high"]:
        mask = groups[group_name]
        mass = np.sum(D * mask)
        if mass <= 1e-12:
            continue
        rows.append({
            "категория": "MPC по группам",
            "момент": f"{pretty_group_label(group_name)}: средняя MPC",
            "значение": float(np.sum(D * mpc * mask) / mass),
        })

    for idx in range(1, 6):
        mask = groups[f"liquid_q{idx}"]
        mass = np.sum(D * mask)
        if mass <= 1e-12:
            continue
        rows.append({
            "категория": "MPC по квантилям ликвидного богатства",
            "момент": f"{idx}-й квантиль: средняя MPC",
            "значение": float(np.sum(D * mpc * mask) / mass),
        })
    return pd.DataFrame(rows)


def transfer_mpc_moments_frame(ss, transfer_mpc, config):
    D = stationary_distribution(ss)
    groups = build_group_masks(ss, config)["groups"]
    rows = [
        {"категория": "Распределение MPC из трансфертного шока", "момент": "Средняя MPC", "значение": float(np.sum(D * transfer_mpc))},
        {"категория": "Распределение MPC из трансфертного шока", "момент": "Медианная MPC", "значение": weighted_quantile(transfer_mpc, D, 0.5)},
        {"категория": "Распределение MPC из трансфертного шока", "момент": "Доля MPC > 0.2", "значение": float(np.sum(D * (transfer_mpc > 0.2)))},
        {"категория": "Распределение MPC из трансфертного шока", "момент": "Доля MPC > 0.3", "значение": float(np.sum(D * (transfer_mpc > 0.3)))},
        {"категория": "Распределение MPC из трансфертного шока", "момент": "Доля MPC > 0.5", "значение": float(np.sum(D * (transfer_mpc > 0.5)))},
    ]

    for group_name in ["low_liquid", "wealthy_htm", "high_liquid"]:
        mask = groups[group_name]
        mass = np.sum(D * mask)
        if mass <= 1e-12:
            continue
        rows.append({
            "категория": "MPC из трансфертного шока по группам",
            "момент": f"{pretty_group_label(group_name)}: средняя MPC",
            "значение": float(np.sum(D * transfer_mpc * mask) / mass),
        })

    for idx in range(1, 6):
        mask = groups[f"liquid_q{idx}"]
        mass = np.sum(D * mask)
        if mass <= 1e-12:
            continue
        rows.append({
            "категория": "MPC из трансфертного шока по квантилям ликвидного богатства",
            "момент": f"{idx}-й квантиль: средняя MPC",
            "значение": float(np.sum(D * transfer_mpc * mask) / mass),
        })

    return pd.DataFrame(rows)


def group_profile_frame(ss, mpc, config):
    D = stationary_distribution(ss)
    hh = ss.internals["hh"]
    mesh = state_mesh(ss)
    groups = build_group_masks(ss, config, mpc=mpc)["groups"]
    labor_income = mesh["z"]
    financial_income = ss["rb"] * mesh["b"] + ss["ra"] * mesh["a"]
    disposable_income = labor_income + financial_income
    total_consumption = float(np.sum(D * hh["c"]))
    total_disposable_income = float(np.sum(D * disposable_income))

    rows = []
    for group_name in ["low_liquid", "wealthy_htm", "high_liquid", "high_total_wealth", "mpc_high"]:
        mask = groups[group_name]
        mass = np.sum(D * mask)
        if mass <= 1e-12:
            continue
        rows.append({
            "код группы": group_name,
            "группа": pretty_group_label(group_name),
            "доля группы": float(mass),
            "средняя MPC": float(np.sum(D * mpc * mask) / mass),
            "среднее ликвидное богатство": float(np.sum(D * mesh["b"] * mask) / mass),
            "среднее неликвидное богатство": float(np.sum(D * mesh["a"] * mask) / mass),
            "среднее совокупное богатство": float(np.sum(D * (mesh["a"] + mesh["b"]) * mask) / mass),
            "доля в совокупном потреблении": float(np.sum(D * hh["c"] * mask) / total_consumption),
            "доля в располагаемом доходе": float(np.sum(D * disposable_income * mask) / total_disposable_income),
        })
    return pd.DataFrame(rows)


def wealthy_htm_sensitivity_frame(
    ss,
    transition,
    mpc,
    config,
    low_liquidity_thresholds=(0.0, 0.1, 0.25, 0.5),
    illiquid_quantiles=(0.5, 0.6, 0.7),
):
    D = stationary_distribution(ss)
    hh = ss.internals["hh"]
    mesh = state_mesh(ss)
    path_levels = household_path_levels(ss, transition)
    rows = []
    for low_b in low_liquidity_thresholds:
        for a_q in illiquid_quantiles:
            cfg = replace(
                config,
                low_liquidity_threshold=float(low_b),
                wealthy_htm_a_quantile=float(a_q),
            )
            masks = build_group_masks(ss, cfg, mpc=mpc)
            mask = masks["groups"]["wealthy_htm"]
            mass = np.sum(D * mask)
            if mass <= 1e-12:
                rows.append({
                    "low_liquidity_threshold": float(low_b),
                    "wealthy_htm_a_quantile": float(a_q),
                    "wealthy_htm_a_cutoff": masks["thresholds"]["wealthy_htm_a_cutoff"],
                    "share_wealthy_htm": 0.0,
                    "mean_mpc_wealthy_htm": 0.0,
                    "mean_liquid_wealth": 0.0,
                    "mean_illiquid_wealth": 0.0,
                    "peak_consumption_response": 0.0,
                    "integral_consumption_response": 0.0,
                })
                continue
            baseline_consumption = np.sum(hh["D"] * hh["c"] * mask)
            path_consumption = np.sum(path_levels["D"] * path_levels["c"] * mask[None, ...], axis=(1, 2, 3))
            consumption_pct = (
                np.zeros(path_consumption.shape[0])
                if abs(baseline_consumption) <= 1e-12
                else 100.0 * (path_consumption - baseline_consumption) / baseline_consumption
            )
            rows.append({
                "low_liquidity_threshold": float(low_b),
                "wealthy_htm_a_quantile": float(a_q),
                "wealthy_htm_a_cutoff": masks["thresholds"]["wealthy_htm_a_cutoff"],
                "share_wealthy_htm": float(mass),
                "mean_mpc_wealthy_htm": float(np.sum(D * mpc * mask) / mass),
                "mean_liquid_wealth": float(np.sum(D * mesh["b"] * mask) / mass),
                "mean_illiquid_wealth": float(np.sum(D * mesh["a"] * mask) / mass),
                "peak_consumption_response": float(consumption_pct[np.argmax(np.abs(consumption_pct))]),
                "integral_consumption_response": float(np.sum(consumption_pct)),
            })
    return pd.DataFrame(rows)


def household_robustness_frames(base_config, precomputed_baseline=None):
    summary_rows = []
    group_rows = []
    baseline_guess = None
    if precomputed_baseline is not None:
        baseline_guess = {
            "beta_guess": float(precomputed_baseline["ss"]["beta"]),
            "chi1_guess": float(precomputed_baseline["ss"]["chi1"]),
        }
    for scenario in household_robustness_scenarios(base_config):
        scenario_name = scenario["name"]
        scenario_label = scenario["label"]
        cfg = scenario["config"]
        if scenario_name == "baseline" and precomputed_baseline is not None:
            bundle = precomputed_baseline["bundle"]
            ss = precomputed_baseline["ss"]
            mpc = precomputed_baseline["mpc"]
            transition = precomputed_baseline["transition"]
        else:
            if baseline_guess is not None:
                cfg = replace(cfg, **baseline_guess)
            try:
                bundle = solve_steady_state(cfg)
                ss = bundle["ss"]
                mpc = compute_mpc(ss)
                experiment = monetary_policy_experiment(cfg)
                transition = solve_transition(bundle, experiment["inputs"])
            except Exception:
                continue
        channels = channel_decomposition(bundle, transition)
        groups = build_group_masks(ss, cfg, mpc=mpc)["groups"]
        D = stationary_distribution(ss)

        group_irf = group_consumption_irfs(ss, transition, cfg, mpc, scenario_name, scenario_label)
        def peak_for_group(name: str) -> float:
            subset = group_irf[group_irf["group"] == name].sort_values("period")
            if subset.empty:
                return 0.0
            values = subset["value"].to_numpy(dtype=float)
            return float(values[np.argmax(np.abs(values))])

        summary_rows.append({
            "scenario": scenario_name,
            "scenario_label": pretty_robustness_label(scenario_name),
            "solved_beta": float(ss["beta"]),
            "solved_chi1": float(ss["chi1"]),
            "mean_mpc": float(np.sum(D * mpc)),
            "median_mpc": weighted_quantile(mpc, D, 0.5),
            "share_mpc_above_0_2": float(np.sum(D * (mpc > 0.2))),
            "share_low_liquidity": float(np.sum(D * groups["low_liquid"])),
            "share_wealthy_htm": float(np.sum(D * groups["wealthy_htm"])),
            "peak_output_response": float(np.min(100.0 * transition["Y"] / ss["Y"])),
            "peak_consumption_response": float(np.min(100.0 * transition["C"] / ss["C"])),
            "peak_inflation_response": float(np.min(100.0 * transition["pi"])),
            "peak_low_liquid_consumption": peak_for_group("low_liquid"),
            "peak_wealthy_htm_consumption": peak_for_group("wealthy_htm"),
            "peak_high_liquid_consumption": peak_for_group("high_liquid"),
            "peak_redistribution_channel": float(
                (100.0 * channels["redistribution_liquidity_residual"] / ss["C"])[
                    np.argmax(np.abs(100.0 * channels["redistribution_liquidity_residual"] / ss["C"]))
                ]
            ),
        })

        for group_name in ["low_liquid", "wealthy_htm", "high_liquid", "mpc_high"]:
            subset = group_irf[group_irf["group"] == group_name].sort_values("period")
            if subset.empty:
                continue
            values = subset["value"].to_numpy(dtype=float)
            group_rows.append({
                "scenario": scenario_name,
                "scenario_label": pretty_robustness_label(scenario_name),
                "group": group_name,
                "group_label": pretty_group_label(group_name),
                "peak_consumption_response": float(values[np.argmax(np.abs(values))]),
                "integral_consumption_response": float(np.sum(values)),
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(group_rows)


def reference_alignment_frames(base_config, precomputed_baseline=None):
    def summarize_row(name, label, source, cfg, ss, mpc, transition):
        D = stationary_distribution(ss)
        groups = build_group_masks(ss, cfg, mpc=mpc)["groups"]
        group_mpc = {}
        for group_name in ["low_liquid", "wealthy_htm", "high_liquid"]:
            mask = groups[group_name]
            mass = np.sum(D * mask)
            group_mpc[group_name] = 0.0 if mass <= 1e-12 else float(np.sum(D * mpc * mask) / mass)
        return {
            "specification": name,
            "label": label,
            "source": source,
            "chi0": float(cfg.chi0),
            "omega": float(cfg.omega),
            "sigma_z": float(cfg.sigma_z),
            "rho_z": float(cfg.rho_z),
            "beta_guess": float(cfg.beta_guess),
            "chi1_guess": float(cfg.chi1_guess),
            "solved_beta": float(ss["beta"]),
            "solved_chi1": float(ss["chi1"]),
            "mean_mpc": float(np.sum(D * mpc)),
            "median_mpc": float(weighted_quantile(mpc, D, 0.5)),
            "share_low_liquidity": float(np.sum(D * groups["low_liquid"])),
            "share_wealthy_htm": float(np.sum(D * groups["wealthy_htm"])),
            "low_liquid_mpc": group_mpc["low_liquid"],
            "wealthy_htm_mpc": group_mpc["wealthy_htm"],
            "high_liquid_mpc": group_mpc["high_liquid"],
            "peak_output_response": float(np.min(100.0 * transition["Y"] / ss["Y"])),
            "peak_consumption_response": float(np.min(100.0 * transition["C"] / ss["C"])),
            "peak_inflation_response": float(np.min(100.0 * transition["pi"])),
        }

    if precomputed_baseline is not None:
        baseline_bundle = precomputed_baseline["bundle"]
        baseline_ss = precomputed_baseline["ss"]
        baseline_mpc = precomputed_baseline["mpc"]
        baseline_transition = precomputed_baseline["transition"]
    else:
        baseline_bundle = solve_steady_state(base_config)
        baseline_ss = baseline_bundle["ss"]
        baseline_mpc = compute_mpc(baseline_ss)
        baseline_transition = solve_transition(baseline_bundle, monetary_policy_experiment(base_config)["inputs"])

    reference_cfg = sequence_jacobian_tutorial_calibration(base_config, preserve_grid=True)
    reference_bundle = solve_steady_state(reference_cfg)
    reference_ss = reference_bundle["ss"]
    reference_mpc = compute_mpc(reference_ss)
    reference_transition = solve_transition(reference_bundle, monetary_policy_experiment(reference_cfg)["inputs"])

    summary = pd.DataFrame([
        summarize_row(
            "project_baseline",
            "Текущая усиленная baseline-калибровка",
            "Проектный full HANK baseline",
            base_config,
            baseline_ss,
            baseline_mpc,
            baseline_transition,
        ),
        summarize_row(
            "sequence_jacobian_tutorial",
            "Калибровка tutorial two_asset (на reduced grid)",
            "sequence-jacobian Tutorial 4 / two_asset.ipynb",
            reference_cfg,
            reference_ss,
            reference_mpc,
            reference_transition,
        ),
    ])

    parameter_rows = []
    for parameter, label, comment in [
        ("chi0", "chi0", "Сдвиг в знаменателе функции издержек ребалансировки"),
        ("omega", "omega", "Клин между liquid return и policy rate"),
        ("sigma_z", "sigma_z", "Безусловная дисперсия лог-идосинкратического дохода"),
        ("rho_z", "rho_z", "Персистентность идосинкратического дохода"),
        ("beta_guess", "beta guess", "Начальное значение beta в steady state"),
        ("chi1_guess", "chi1 guess", "Начальное значение масштаба издержек"),
    ]:
        baseline_value = float(getattr(base_config, parameter))
        reference_value = float(getattr(reference_cfg, parameter))
        parameter_rows.append({
            "параметр": label,
            "проектный baseline": baseline_value,
            "tutorial two_asset": reference_value,
            "разница": baseline_value - reference_value,
            "комментарий": comment,
        })

    return pd.DataFrame(parameter_rows), summary
