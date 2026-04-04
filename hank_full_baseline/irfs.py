from __future__ import annotations

import numpy as np
import pandas as pd

from .distribution import build_group_masks, household_path_levels, stationary_distribution
from .grids import state_mesh
from .labels import pretty_channel_label, pretty_group_label


REAL_VARS = {"Y", "C", "N", "w", "A", "B", "I"}
RATE_VARS = {"pi", "i", "r", "ra", "rb", "tax"}


def aggregate_paths_frame(ss, transition, scenario_name, scenario_label):
    periods = np.arange(transition.T)
    rows = []
    for period in periods:
        row = {
            "scenario": scenario_name,
            "scenario_label": scenario_label,
            "period": int(period),
        }
        for variable in ["Y", "output_gap", "pi", "i", "C", "I", "N", "w", "r", "ra", "rb", "A", "B", "tax"]:
            deviation = float(transition[variable][period])
            row[f"{variable}_deviation"] = deviation
            row[f"{variable}_level"] = float(ss[variable] + deviation) if variable in ss else deviation
            if variable in REAL_VARS:
                row[f"{variable}_pct"] = 100.0 * deviation / float(ss[variable])
            elif variable == "output_gap":
                row[f"{variable}_pct"] = 100.0 * deviation
            else:
                row[f"{variable}_pct"] = 100.0 * deviation
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_irf_frame(ss, transition, scenario_name, scenario_label):
    wide = aggregate_paths_frame(ss, transition, scenario_name, scenario_label)
    rows = []
    variables = ["Y", "C", "I", "N", "w", "output_gap", "pi", "i", "r", "ra", "rb", "A", "B", "tax"]
    for variable in variables:
        units = "% отклонения" if variable in REAL_VARS or variable == "output_gap" else "п.п."
        for _, entry in wide.iterrows():
            rows.append({
                "scenario": scenario_name,
                "scenario_label": scenario_label,
                "period": int(entry["period"]),
                "variable": variable,
                "value": float(entry[f"{variable}_pct"]),
                "raw_deviation": float(entry[f"{variable}_deviation"]),
                "units": units,
            })
    return pd.DataFrame(rows)


def _safe_group_mean(values, weights, mask):
    weighted_mass = np.sum(weights * mask)
    if abs(weighted_mass) < 1e-12:
        return 0.0
    return float(np.sum(values * weights * mask) / weighted_mass)


def _group_series(path_levels, mask, variable):
    D_t = path_levels["D"]
    x_t = path_levels[variable]
    return np.sum(D_t * x_t * mask[None, ...], axis=(1, 2, 3))


def _group_pct_deviation(path_levels, ss_levels, mask, variable):
    baseline = np.sum(ss_levels["D"] * ss_levels[variable] * mask)
    path = _group_series(path_levels, mask, variable)
    if abs(baseline) < 1e-12:
        return np.zeros(path.shape[0])
    return 100.0 * (path - baseline) / baseline


def group_consumption_irfs(ss, transition, config, mpc, scenario_name, scenario_label):
    masks = build_group_masks(ss, config, mpc=mpc)["groups"]
    path_levels = household_path_levels(ss, transition)
    ss_levels = ss.internals["hh"]

    groups = (
        [f"liquid_q{i}" for i in range(1, 6)]
        + ["low_liquid", "wealthy_htm", "high_liquid"]
        + ["mpc_low", "mpc_mid", "mpc_high"]
    )
    rows = []
    for group_name in groups:
        if group_name not in masks:
            continue
        series = _group_pct_deviation(path_levels, ss_levels, masks[group_name], "c")
        for period, value in enumerate(series):
            rows.append({
                "scenario": scenario_name,
                "scenario_label": scenario_label,
                "period": period,
                "group": group_name,
                "variable": "consumption",
                "value": float(value),
            })
    return pd.DataFrame(rows)


def group_income_irfs(ss, transition, config, mpc, scenario_name, scenario_label):
    masks = build_group_masks(ss, config, mpc=mpc)["groups"]
    path_levels = household_path_levels(ss, transition)
    ss_hh = ss.internals["hh"]

    z_path = path_levels["z_grid"][:, :, None, None]
    b_path = path_levels["b_grid"][:, None, :, None]
    a_path = path_levels["a_grid"][:, None, None, :]
    rb_path = (transition["rb"] + ss["rb"])[:, None, None, None]
    ra_path = (transition["ra"] + ss["ra"])[:, None, None, None]
    labor_income_path = z_path
    financial_income_path = rb_path * b_path + ra_path * a_path
    disposable_income_path = labor_income_path + financial_income_path

    z_ss = ss_hh["z_grid"][:, None, None]
    b_ss = ss_hh["b_grid"][None, :, None]
    a_ss = ss_hh["a_grid"][None, None, :]
    labor_income_ss = z_ss
    financial_income_ss = ss["rb"] * b_ss + ss["ra"] * a_ss
    disposable_income_ss = labor_income_ss + financial_income_ss

    groups = ["low_liquid", "wealthy_htm", "high_liquid", "mpc_low", "mpc_mid", "mpc_high"]
    rows = []
    for group_name in groups:
        if group_name not in masks:
            continue
        mask = masks[group_name]
        for var_name, path, base in [
            ("labor_income", labor_income_path, labor_income_ss),
            ("financial_income", financial_income_path, financial_income_ss),
            ("disposable_income", disposable_income_path, disposable_income_ss),
        ]:
            baseline = np.sum(ss_hh["D"] * base * mask)
            series = np.sum(path_levels["D"] * path * mask[None, ...], axis=(1, 2, 3))
            pct = np.zeros(series.shape[0]) if abs(baseline) < 1e-12 else 100.0 * (series - baseline) / baseline
            for period, value in enumerate(pct):
                rows.append({
                    "scenario": scenario_name,
                    "scenario_label": scenario_label,
                    "period": period,
                    "group": group_name,
                    "variable": var_name,
                    "value": float(value),
                })
    return pd.DataFrame(rows)


def aggregate_income_channels_frame(ss, transition, scenario_name, scenario_label):
    path_levels = household_path_levels(ss, transition)
    hh = ss.internals["hh"]
    mesh = state_mesh(ss)
    b_ss = mesh["b"]
    a_ss = mesh["a"]
    z_ss = mesh["z"]

    rows = []
    for period in range(transition.T):
        D_t = path_levels["D"][period]
        z_t = path_levels["z_grid"][period][:, None, None]
        b_t = path_levels["b_grid"][period][None, :, None]
        a_t = path_levels["a_grid"][period][None, None, :]
        labor = float(np.sum(D_t * z_t))
        financial = float(np.sum(D_t * ((ss["rb"] + transition["rb"][period]) * b_t + (ss["ra"] + transition["ra"][period]) * a_t)))
        disposable = labor + financial
        labor_ss = float(np.sum(hh["D"] * z_ss))
        financial_ss = float(np.sum(hh["D"] * (ss["rb"] * b_ss + ss["ra"] * a_ss)))
        disposable_ss = labor_ss + financial_ss
        rows.extend([
            {
                "scenario": scenario_name,
                "scenario_label": scenario_label,
                "period": period,
                "channel": "labor_income",
                "value": 100.0 * (labor - labor_ss) / labor_ss,
            },
            {
                "scenario": scenario_name,
                "scenario_label": scenario_label,
                "period": period,
                "channel": "financial_income",
                "value": 100.0 * (financial - financial_ss) / financial_ss if abs(financial_ss) > 1e-12 else 0.0,
            },
            {
                "scenario": scenario_name,
                "scenario_label": scenario_label,
                "period": period,
                "channel": "disposable_income",
                "value": 100.0 * (disposable - disposable_ss) / disposable_ss,
            },
        ])
    return pd.DataFrame(rows)


def channel_decomposition_frame(ss, transition, channels, scenario_name, scenario_label):
    rows = []
    ss_c = ss["C"]
    total = 100.0 * transition["C"] / ss_c
    for name, series in channels.items():
        transformed = 100.0 * series / ss_c
        for period, value in enumerate(transformed):
            rows.append({
                "scenario": scenario_name,
                "scenario_label": scenario_label,
                "period": period,
                "component": name,
                "component_label": pretty_channel_label(name),
                "value": float(value),
            })
    for period, value in enumerate(total):
        rows.append({
            "scenario": scenario_name,
            "scenario_label": scenario_label,
            "period": period,
            "component": "general_equilibrium_total",
            "component_label": pretty_channel_label("general_equilibrium_total"),
            "value": float(value),
        })
    return pd.DataFrame(rows)


def group_contribution_frame(ss, transition, config, mpc, scenario_name, scenario_label):
    masks = build_group_masks(ss, config, mpc=mpc)["groups"]
    path_levels = household_path_levels(ss, transition)
    ss_levels = ss.internals["hh"]
    rows = []
    for group_name in ["low_liquid_non_whtm", "wealthy_htm", "high_liquid", "other_households"]:
        if group_name not in masks:
            continue
        path = _group_series(path_levels, masks[group_name], "c")
        baseline = np.sum(ss_levels["D"] * ss_levels["c"] * masks[group_name])
        contribution = 100.0 * (path - baseline) / ss["C"]
        for period, value in enumerate(contribution):
            rows.append({
                "scenario": scenario_name,
                "scenario_label": scenario_label,
                "period": period,
                "group": group_name,
                "group_label": pretty_group_label(group_name),
                "value": float(value),
            })
    aggregate_total = 100.0 * transition["C"] / ss["C"]
    for period, value in enumerate(aggregate_total):
        rows.append({
            "scenario": scenario_name,
            "scenario_label": scenario_label,
            "period": period,
            "group": "aggregate_total",
            "group_label": "Совокупный отклик потребления",
            "value": float(value),
        })
    return pd.DataFrame(rows)


def group_paths_frame(ss, transition, config, mpc, scenario_name, scenario_label):
    masks = build_group_masks(ss, config, mpc=mpc)["groups"]
    hh_ss = ss.internals["hh"]
    path_levels = household_path_levels(ss, transition)
    groups = []
    groups.extend(("liquid_wealth_quantile", f"liquid_q{i}") for i in range(1, 6))
    groups.extend(("illiquid_wealth_quantile", f"illiquid_q{i}") for i in range(1, 6))
    groups.extend(("total_wealth_quantile", f"wealth_q{i}") for i in range(1, 6))
    groups.extend([
        ("income_group", "income_low"),
        ("income_group", "income_mid_1"),
        ("income_group", "income_high"),
        ("mpc_group", "mpc_low"),
        ("mpc_group", "mpc_mid"),
        ("mpc_group", "mpc_high"),
        ("balance_sheet_group", "low_liquid"),
        ("balance_sheet_group", "wealthy_htm"),
        ("balance_sheet_group", "high_liquid"),
        ("exclusive_balance_sheet_group", "low_liquid_non_whtm"),
        ("exclusive_balance_sheet_group", "other_households"),
    ])

    rows = []
    for grouping, group_name in groups:
        if group_name not in masks:
            continue
        mask = masks[group_name]
        baseline_c = np.sum(hh_ss["D"] * hh_ss["c"] * mask)
        for period in range(path_levels["D"].shape[0]):
            D_t = path_levels["D"][period]
            z_t = path_levels["z_grid"][period][:, None, None]
            b_t = path_levels["b_grid"][period][None, :, None]
            a_t = path_levels["a_grid"][period][None, None, :]
            c_t = path_levels["c"][period]
            labor_income_t = z_t
            financial_income_t = (ss["rb"] + transition["rb"][period]) * b_t + (ss["ra"] + transition["ra"][period]) * a_t
            disposable_t = labor_income_t + financial_income_t
            group_mass = np.sum(D_t * mask)
            aggregate_group_consumption = float(np.sum(D_t * c_t * mask))
            baseline_group_consumption = float(baseline_c)
            rows.append({
                "scenario": scenario_name,
                "scenario_label": scenario_label,
                "period": period,
                "grouping": grouping,
                "group": group_name,
                "group_label": pretty_group_label(group_name),
                "group_mass": float(group_mass),
                "mean_consumption": _safe_group_mean(c_t, D_t, mask),
                "mean_labor_income": _safe_group_mean(labor_income_t, D_t, mask),
                "mean_financial_income": _safe_group_mean(financial_income_t, D_t, mask),
                "mean_disposable_income": _safe_group_mean(disposable_t, D_t, mask),
                "mean_liquid_assets": _safe_group_mean(b_t, D_t, mask),
                "mean_illiquid_assets": _safe_group_mean(a_t, D_t, mask),
                "mean_total_wealth": _safe_group_mean(a_t + b_t, D_t, mask),
                "mean_mpc": _safe_group_mean(mpc, hh_ss["D"], mask),
                "aggregate_group_consumption": aggregate_group_consumption,
                "baseline_group_consumption": baseline_group_consumption,
                "aggregate_group_consumption_change": aggregate_group_consumption - baseline_group_consumption,
                "aggregate_group_consumption_contribution_pct": 100.0 * (aggregate_group_consumption - baseline_group_consumption) / ss["C"],
                "consumption_pct_deviation": 0.0 if abs(baseline_c) < 1e-12 else 100.0 * (np.sum(D_t * c_t * mask) - baseline_c) / baseline_c,
            })
    return pd.DataFrame(rows)


def steady_state_group_statistics(ss, mpc, config):
    masks = build_group_masks(ss, config, mpc=mpc)
    groups = masks["groups"]
    D = stationary_distribution(ss)
    mesh = state_mesh(ss)
    rows = []
    for group_name in [
        "low_liquid",
        "wealthy_htm",
        "high_liquid",
        "high_total_wealth",
        "mpc_low",
        "mpc_mid",
        "mpc_high",
    ]:
        if group_name not in groups:
            continue
        mask = groups[group_name]
        mass = np.sum(D * mask)
        if abs(mass) < 1e-12:
            continue
        rows.append({
            "group": group_name,
            "group_label": pretty_group_label(group_name),
            "share": float(mass),
            "mean_mpc": float(np.sum(D * mpc * mask) / mass),
            "mean_liquid_wealth": float(np.sum(D * mesh["b"] * mask) / mass),
            "mean_illiquid_wealth": float(np.sum(D * mesh["a"] * mask) / mass),
        })
    return pd.DataFrame(rows)
