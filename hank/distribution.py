from __future__ import annotations

import numpy as np
import pandas as pd

from .grids import central_interval, extract_household_ss, quantile_masks, state_mesh, weighted_quantile


def stationary_distribution(ss):
    return extract_household_ss(ss)["D"]


def household_levels(ss):
    return extract_household_ss(ss)


def aggregate_from_distribution(values, distribution):
    return float(np.sum(np.asarray(values) * np.asarray(distribution)))


def marginal_distributions(ss):
    hh = household_levels(ss)
    D = hh["D"]
    return {
        "b": D.sum(axis=(0, 2)),
        "a": D.sum(axis=(0, 1)),
        "joint": D.sum(axis=0),
        "b_grid": hh["b_grid"],
        "a_grid": hh["a_grid"],
    }


def household_path_levels(ss, impulse):
    hh_ss = household_levels(ss)
    hh_irf = impulse.internals["hh"]
    levels = {}
    for key, value in hh_irf.items():
        levels[key] = value + hh_ss[key][None, ...]
    return levels


def weighted_gini(values, weights):
    values = np.asarray(values).reshape(-1)
    weights = np.asarray(weights).reshape(-1)
    mask = weights > 0
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumw = np.cumsum(weights)
    cumxw = np.cumsum(values * weights)
    if cumxw[-1] <= 0:
        return 0.0
    relw = cumw / cumw[-1]
    relx = cumxw / cumxw[-1]
    area = np.trapz(relx, relw)
    return float(1.0 - 2.0 * area)


def build_group_masks(ss, config, mpc=None):
    D = stationary_distribution(ss)
    mesh = state_mesh(ss)
    b = mesh["b"]
    a = mesh["a"]
    wealth = a + b

    liquid_masks = quantile_masks(b, D, 5)
    illiquid_masks = quantile_masks(a, D, 5)
    wealth_masks = quantile_masks(wealth, D, 5)

    low_liquid_cutoff = config.low_liquidity_threshold
    high_liquid_cutoff = weighted_quantile(b, D, config.high_liquidity_quantile)
    wealthy_a_cutoff = weighted_quantile(a, D, config.wealthy_htm_a_quantile)
    high_total_cutoff = weighted_quantile(wealth, D, 0.8)

    groups = {
        f"liquid_q{i + 1}": mask for i, mask in enumerate(liquid_masks)
    }
    groups.update({
        f"illiquid_q{i + 1}": mask for i, mask in enumerate(illiquid_masks)
    })
    groups.update({
        f"wealth_q{i + 1}": mask for i, mask in enumerate(wealth_masks)
    })
    groups["low_liquid"] = b <= low_liquid_cutoff
    groups["wealthy_htm"] = (b <= low_liquid_cutoff) & (a >= wealthy_a_cutoff)
    groups["high_liquid"] = b >= high_liquid_cutoff
    groups["low_liquid_non_whtm"] = groups["low_liquid"] & (~groups["wealthy_htm"])
    groups["other_households"] = ~(groups["low_liquid_non_whtm"] | groups["wealthy_htm"] | groups["high_liquid"])
    groups["high_total_wealth"] = wealth >= high_total_cutoff
    for idx, z_state in enumerate(household_levels(ss)["z_grid"]):
        if idx == 0:
            label = "income_low"
        elif idx == len(household_levels(ss)["z_grid"]) - 1:
            label = "income_high"
        else:
            label = f"income_mid_{idx}"
        groups[label] = np.isclose(mesh["z"], z_state)

    if mpc is not None:
        mpc_masks = quantile_masks(mpc, D, 3)
        groups["mpc_low"] = mpc_masks[0]
        groups["mpc_mid"] = mpc_masks[1]
        groups["mpc_high"] = mpc_masks[2]

    return {
        "groups": groups,
        "thresholds": {
            "low_liquid_cutoff": float(low_liquid_cutoff),
            "high_liquid_cutoff": float(high_liquid_cutoff),
            "wealthy_htm_a_cutoff": float(wealthy_a_cutoff),
            "high_total_wealth_cutoff": float(high_total_cutoff),
        },
    }


def distribution_snapshots(ss, path_levels, horizons):
    snapshots_b = []
    snapshots_a = []
    for horizon in horizons:
        D = path_levels["D"][horizon]
        snapshots_b.append(pd.DataFrame({
            "period": horizon,
            "b": path_levels["b_grid"][horizon],
            "mass": D.sum(axis=(0, 2)),
        }))
        snapshots_a.append(pd.DataFrame({
            "period": horizon,
            "a": path_levels["a_grid"][horizon],
            "mass": D.sum(axis=(0, 1)),
        }))
    return pd.concat(snapshots_b, ignore_index=True), pd.concat(snapshots_a, ignore_index=True)


def group_share(mask, distribution):
    return float(np.sum(distribution[mask]))


def central_ranges_for_assets(ss, config):
    hh = household_levels(ss)
    D = stationary_distribution(ss)
    mesh = state_mesh(ss)
    return {
        "b": central_interval(mesh["b"], D, config.central_mass),
        "a": central_interval(mesh["a"], D, config.central_mass),
        "wealth": central_interval(mesh["a"] + mesh["b"], D, config.central_mass),
        "b_zero": 0.0,
        "a_zero": 0.0,
        "b_grid": hh["b_grid"],
        "a_grid": hh["a_grid"],
    }


def path_distribution_statistics(ss, path_levels, config, mpc_path=None):
    rows = []
    ss_groups = build_group_masks(ss, config)
    wealthy_a_cutoff = ss_groups["thresholds"]["wealthy_htm_a_cutoff"]
    low_liquid_cutoff = ss_groups["thresholds"]["low_liquid_cutoff"]

    for period in range(path_levels["D"].shape[0]):
        D_t = path_levels["D"][period]
        b_grid_t = np.broadcast_to(path_levels["b_grid"][period][None, :, None], D_t.shape)
        a_grid_t = np.broadcast_to(path_levels["a_grid"][period][None, None, :], D_t.shape)
        wealth_t = a_grid_t + b_grid_t
        low_liq_mask = (b_grid_t <= low_liquid_cutoff).astype(float)
        whtm_mask = ((b_grid_t <= low_liquid_cutoff) & (a_grid_t >= wealthy_a_cutoff)).astype(float)
        top20_mask = (wealth_t >= weighted_quantile(wealth_t, D_t, 0.8)).astype(float)
        stats = {
            "period": period,
            "mean_liquid_wealth": float(np.sum(D_t * b_grid_t)),
            "median_liquid_wealth": weighted_quantile(b_grid_t, D_t, 0.5),
            "mean_illiquid_wealth": float(np.sum(D_t * a_grid_t)),
            "median_illiquid_wealth": weighted_quantile(a_grid_t, D_t, 0.5),
            "mean_total_wealth": float(np.sum(D_t * wealth_t)),
            "median_total_wealth": weighted_quantile(wealth_t, D_t, 0.5),
            "p10_liquid_wealth": weighted_quantile(b_grid_t, D_t, 0.1),
            "p25_liquid_wealth": weighted_quantile(b_grid_t, D_t, 0.25),
            "p75_liquid_wealth": weighted_quantile(b_grid_t, D_t, 0.75),
            "p90_liquid_wealth": weighted_quantile(b_grid_t, D_t, 0.9),
            "p10_illiquid_wealth": weighted_quantile(a_grid_t, D_t, 0.1),
            "p25_illiquid_wealth": weighted_quantile(a_grid_t, D_t, 0.25),
            "p75_illiquid_wealth": weighted_quantile(a_grid_t, D_t, 0.75),
            "p90_illiquid_wealth": weighted_quantile(a_grid_t, D_t, 0.9),
            "share_low_liquidity": float(np.sum(D_t * low_liq_mask)),
            "share_wealthy_htm": float(np.sum(D_t * whtm_mask)),
            "share_top20_total_wealth": float(np.sum(D_t * top20_mask)),
            "gini_liquid_wealth": weighted_gini(b_grid_t, D_t),
            "gini_total_wealth": weighted_gini(wealth_t, D_t),
        }
        if mpc_path is not None:
            mpc_t = mpc_path[period]
            stats["mean_mpc"] = float(np.sum(D_t * mpc_t))
            stats["median_mpc"] = weighted_quantile(mpc_t, D_t, 0.5)
            stats["interest_exposure"] = float(np.sum(D_t * b_grid_t * mpc_t))
        rows.append(stats)
    return pd.DataFrame(rows)
