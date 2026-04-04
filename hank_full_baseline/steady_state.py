from __future__ import annotations

from .calibration import HANKCalibration
from .model import build_models


def solve_steady_state(config: HANKCalibration):
    spec = build_models(config)
    calibration_solution = spec["model_ss"].solve_steady_state(
        spec["calibration"],
        spec["ss_unknowns"],
        spec["ss_targets"],
        solver="broyden_custom",
    )
    ss = spec["model"].steady_state(calibration_solution)
    spec["cali"] = calibration_solution
    spec["ss"] = ss
    spec["household_block"] = spec["model"].blocks[9]
    return spec


def steady_state_aggregates(ss):
    keys = [
        "Y",
        "C",
        "I",
        "N",
        "w",
        "pi",
        "i",
        "r",
        "ra",
        "rb",
        "output_gap",
        "A",
        "B",
        "Bg",
        "G",
        "wealth",
        "goods_mkt",
        "asset_mkt",
        "tax",
    ]
    return {key: float(ss[key]) for key in keys}
