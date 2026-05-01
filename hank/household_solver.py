from __future__ import annotations

import numpy as np

from .distribution import household_levels, stationary_distribution
from .grids import state_mesh


def compute_mpc(ss):
    hh = household_levels(ss)
    return compute_mpc_from_levels(hh["c"], hh["b_grid"])


def compute_mpc_from_levels(c, b_grid):
    db = np.diff(b_grid)
    dc = np.diff(c, axis=1)
    slope = dc / db[None, :, None]
    mpc = np.empty_like(c)
    mpc[:, 0, :] = slope[:, 0, :]
    mpc[:, -1, :] = slope[:, -1, :]
    if c.shape[1] > 2:
        mpc[:, 1:-1, :] = 0.5 * (slope[:, :-1, :] + slope[:, 1:, :])
    return np.clip(mpc, 0.0, 1.5)


def compute_mpc_path(path_levels):
    periods = path_levels["c"].shape[0]
    values = []
    for period in range(periods):
        values.append(compute_mpc_from_levels(path_levels["c"][period], path_levels["b_grid"][period]))
    return np.stack(values, axis=0)


def compute_transfer_mpc(bundle, shock_size, horizon=4):
    if shock_size == 0:
        raise ValueError("transfer shock size must be non-zero")
    horizon = max(int(horizon), 1)
    transfer_path = np.zeros(horizon)
    transfer_path[0] = shock_size
    impulse = run_household_partial_response(
        bundle,
        {"transfer": transfer_path},
        outputs=("C",),
        include_internals=True,
    )
    transfer_mpc = impulse.internals["hh"]["c"][0] / shock_size
    aggregate_mpc_path = impulse["C"] / shock_size
    return {
        "mpc": transfer_mpc,
        "aggregate_mpc": float(aggregate_mpc_path[0]),
        "aggregate_mpc_path": aggregate_mpc_path,
        "shock_size": float(shock_size),
        "horizon": horizon,
    }


def household_budget_residual(ss):
    hh = household_levels(ss)
    mesh = state_mesh(ss)
    z = mesh["z"]
    b = mesh["b"]
    a = mesh["a"]
    residual = z + (1 + ss["rb"]) * b + (1 + ss["ra"]) * a - hh["chi"] - hh["a"] - hh["b"] - hh["c"]
    return residual


def aggregate_consistency(ss):
    hh = household_levels(ss)
    D = stationary_distribution(ss)
    return {
        "A_from_distribution": float(np.sum(D * hh["a"])),
        "B_from_distribution": float(np.sum(D * hh["b"])),
        "C_from_distribution": float(np.sum(D * hh["c"])),
    }


def run_household_partial_response(bundle, input_paths, outputs=("C",), include_internals=False):
    internals = ["hh"] if include_internals else {}
    return bundle["household_block"].impulse_nonlinear(
        bundle["ss"],
        input_paths,
        outputs=list(outputs),
        internals=internals,
    )
