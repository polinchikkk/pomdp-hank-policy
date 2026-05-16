from __future__ import annotations

from .household_solver import run_household_partial_response


DEFAULT_TRANSITION_OUTPUTS = [
    "Y",
    "output_gap",
    "pi",
    "i",
    "C",
    "I",
    "N",
    "w",
    "r",
    "ra",
    "rb",
    "A",
    "B",
    "tax",
    "asset_mkt",
    "goods_mkt",
]


def solve_transition(bundle, shock_inputs, outputs=None, **kwargs):
    outputs = DEFAULT_TRANSITION_OUTPUTS if outputs is None else list(outputs)
    return bundle["model"].solve_impulse_nonlinear(
        bundle["ss"],
        bundle["unknowns"],
        bundle["targets"],
        shock_inputs,
        outputs=outputs,
        internals=["hh"],
        **kwargs,
    )


def channel_decomposition(bundle, full_transition):
    rate_inputs = {
        "rb": full_transition["rb"],
        "ra": full_transition["ra"],
    }
    income_inputs = {
        "tax": full_transition["tax"],
        "w": full_transition["w"],
        "N": full_transition["N"],
    }
    all_inputs = {
        "rb": full_transition["rb"],
        "ra": full_transition["ra"],
        "tax": full_transition["tax"],
        "w": full_transition["w"],
        "N": full_transition["N"],
    }

    rate_channel = run_household_partial_response(bundle, rate_inputs, outputs=("C",))
    income_channel = run_household_partial_response(bundle, income_inputs, outputs=("C",))
    household_total = run_household_partial_response(bundle, all_inputs, outputs=("C",))

    return {
        "household_total": household_total["C"],
        "intertemporal_financial_channel": rate_channel["C"],
        "labor_income_channel": income_channel["C"],
        "redistribution_liquidity_residual": household_total["C"] - rate_channel["C"] - income_channel["C"],
    }
