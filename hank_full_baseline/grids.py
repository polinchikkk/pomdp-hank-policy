from __future__ import annotations

import numpy as np


def extract_household_ss(ss):
    return ss.internals["hh"]


def state_mesh(ss):
    hh = extract_household_ss(ss)
    D = hh["D"]
    z = np.broadcast_to(hh["z_grid"][:, None, None], D.shape)
    b = np.broadcast_to(hh["b_grid"][None, :, None], D.shape)
    a = np.broadcast_to(hh["a_grid"][None, None, :], D.shape)
    return {"z": z, "b": b, "a": a}


def weighted_quantile(values, weights, quantile):
    values = np.asarray(values).reshape(-1)
    weights = np.asarray(weights).reshape(-1)
    mask = weights > 0
    values = values[mask]
    weights = weights[mask]
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cumulative = cumulative / cumulative[-1]
    return float(np.interp(quantile, cumulative, values))


def quantile_edges(values, weights, n_quantiles):
    return [weighted_quantile(values, weights, q / n_quantiles) for q in range(1, n_quantiles)]


def quantile_masks(values, weights, n_quantiles):
    values = np.asarray(values)
    weights = np.asarray(weights)

    flat_values = values.reshape(-1)
    flat_weights = weights.reshape(-1)
    positive_mass = flat_weights > 0

    if not np.any(positive_mass):
        return [np.zeros_like(values, dtype=bool) for _ in range(n_quantiles)]

    order = np.argsort(flat_values, kind="stable")
    sorted_weights = flat_weights[order]
    cumulative = np.cumsum(sorted_weights)
    total_mass = cumulative[-1]
    midpoint_rank = (cumulative - 0.5 * sorted_weights) / total_mass
    quantile_index = np.minimum((midpoint_rank * n_quantiles).astype(int), n_quantiles - 1)

    masks = []
    for quantile in range(n_quantiles):
        flat_mask = np.zeros_like(flat_values, dtype=bool)
        flat_mask[order[quantile_index == quantile]] = True
        masks.append(flat_mask.reshape(values.shape))
    return masks


def central_interval(values, weights, central_mass):
    lower_q = (1.0 - central_mass) / 2.0
    upper_q = 1.0 - lower_q
    return (
        weighted_quantile(values, weights, lower_q),
        weighted_quantile(values, weights, upper_q),
    )
