from __future__ import annotations

"""Эксперимент 4: отдельная ценность MPC и доли низколиквидных домохозяйств."""


STATISTIC_SETS = (
    ("aggregate_only", ()),
    ("aggregate_plus_mpc", ("mean_mpc",)),
    ("aggregate_plus_liquidity", ("low_liquidity_share",)),
    ("aggregate_plus_mpc_and_liquidity", ("mean_mpc", "low_liquidity_share")),
)
