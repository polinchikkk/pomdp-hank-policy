from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hank.calibration import default_calibration
from hank.distribution import build_group_masks, household_path_levels, path_distribution_statistics, stationary_distribution
from hank.grids import state_mesh
from hank.household_solver import compute_mpc, compute_mpc_path
from hank.steady_state import solve_steady_state
from hank.transition import solve_transition


@dataclass(frozen=True)
class ShockLibrarySpec:
    shock_size: float
    horizon: int
    shocks: tuple[str, ...]
    variables: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class StochasticPathSpec:
    source_shock_library: str
    output_path: str
    trajectory_seeds: tuple[int, ...]
    shock_standard_deviations: dict[str, float]
    shock_persistence: dict[str, float]
    note: str


BASELINE_SHOCKS = ("rstar", "Z", "G")
EXTENDED_SHOCKS = (*BASELINE_SHOCKS, "sigma_z")
SHOCKS = BASELINE_SHOCKS
VARIABLES = (
    "pi",
    "Y",
    "output_gap",
    "C",
    "i",
    "mean_mpc_centered",
    "share_low_liquidity_centered",
    "interest_exposure_centered",
)


def build_shock_response_library(
    *,
    output_dir: Path,
    shock_size: float = 0.001,
    shocks: tuple[str, ...] = SHOCKS,
    config=None,
) -> pd.DataFrame:
    """Build HANK transition responses to non-policy shocks."""

    config = default_calibration() if config is None else config
    bundle = solve_steady_state(config)
    ss = bundle["ss"]
    mpc = compute_mpc(ss)
    steady = _steady_distributional_values(ss, mpc, config)

    rows: list[dict[str, object]] = []
    for shock in shocks:
        shock_path = np.zeros(config.shock_T, dtype=float)
        shock_path[0] = float(shock_size)
        transition = solve_transition(bundle, {shock: shock_path})
        mpc_path = compute_mpc_path(household_path_levels(ss, transition))
        dist = path_distribution_statistics(
            ss,
            household_path_levels(ss, transition),
            config,
            mpc_path=mpc_path,
        )
        by_period = dist.set_index("period")
        for period in range(config.shock_T):
            aggregate_values = {
                "pi": float(transition["pi"][period]),
                "Y": float(transition["Y"][period]),
                "output_gap": float(transition["output_gap"][period]),
                "C": float(transition["C"][period]),
                "i": float(transition["i"][period]),
            }
            dist_row = by_period.loc[period]
            distribution_values = {
                "mean_mpc_centered": float(dist_row["mean_mpc"] - steady["mean_mpc"]),
                "share_low_liquidity_centered": float(dist_row["share_low_liquidity"] - steady["share_low_liquidity"]),
                "interest_exposure_centered": float(dist_row["interest_exposure"] - steady["interest_exposure"]),
            }
            for variable, value in {**aggregate_values, **distribution_values}.items():
                rows.append(
                    {
                        "shock": shock,
                        "period": period,
                        "variable": variable,
                        "response": value,
                        "response_per_unit": value / float(shock_size),
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    library = pd.DataFrame(rows)
    library.to_csv(output_dir / "shock_response_library.csv", index=False)
    spec = ShockLibrarySpec(
        shock_size=float(shock_size),
        horizon=int(config.shock_T),
        shocks=tuple(shocks),
        variables=VARIABLES,
        note=(
            "Отклики получены из HANK transition solver. Они используются как локальная HANK/SSJ-библиотека "
            "для генерации траекторий с неполитическими шоками."
        ),
    )
    (output_dir / "shock_response_library_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "steady_distributional_values.json").write_text(
        json.dumps(steady, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return library


def generate_stochastic_hank_paths(
    *,
    shock_library_csv: Path,
    steady_distributional_values_json: Path,
    output_dir: Path,
    trajectory_seeds: tuple[int, ...],
    shock_standard_deviations: dict[str, float] | None = None,
    shock_persistence: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Generate HANK/SSJ-implied paths from random non-policy shock sequences."""

    library = pd.read_csv(shock_library_csv)
    steady = json.loads(steady_distributional_values_json.read_text(encoding="utf-8"))
    shock_standard_deviations = shock_standard_deviations or {
        "rstar": 0.0007,
        "Z": 0.0007,
        "G": 0.0007,
        "sigma_z": 0.0005,
    }
    shock_persistence = shock_persistence or {
        "rstar": 0.65,
        "Z": 0.75,
        "G": 0.55,
        "sigma_z": 0.70,
    }

    responses = _response_arrays(library)
    horizon = max(array.size for shock_map in responses.values() for array in shock_map.values())
    rows: list[dict[str, object]] = []
    shock_rows: list[dict[str, object]] = []
    for seed in trajectory_seeds:
        rng = np.random.default_rng(int(seed))
        shock_paths = {
            shock: _ar1_path(
                rng=rng,
                horizon=horizon,
                sigma=float(shock_standard_deviations.get(shock, 0.0)),
                rho=float(shock_persistence.get(shock, 0.0)),
            )
            for shock in responses
        }
        values = {
            variable: np.zeros(horizon, dtype=float)
            for variable in VARIABLES
        }
        for shock, shock_path in shock_paths.items():
            for variable, response in responses[shock].items():
                values[variable] += _convolve_path(shock_path, response, horizon)
            for period, value in enumerate(shock_path):
                shock_rows.append(
                    {
                        "scenario": f"shock_path_{seed}",
                        "scenario_label": f"HANK/SSJ path {seed}",
                        "period": period,
                        "shock": shock,
                        "value": value,
                    }
                )

        for period in range(horizon):
            mean_mpc_centered = float(values["mean_mpc_centered"][period])
            low_liquidity_centered = float(values["share_low_liquidity_centered"][period])
            exposure_centered = float(values["interest_exposure_centered"][period])
            rows.append(
                {
                    "scenario": f"shock_path_{seed}",
                    "scenario_label": f"HANK/SSJ path {seed}",
                    "period": period,
                    "pi": float(values["pi"][period]),
                    "Y": float(values["Y"][period]),
                    "output_gap": float(values["output_gap"][period]),
                    "C": float(values["C"][period]),
                    "i": float(values["i"][period]),
                    "mean_mpc": float(steady["mean_mpc"] + mean_mpc_centered),
                    "share_low_liquidity": float(steady["share_low_liquidity"] + low_liquidity_centered),
                    "interest_exposure": float(steady["interest_exposure"] + exposure_centered),
                    "mean_mpc_centered": mean_mpc_centered,
                    "share_low_liquidity_centered": low_liquidity_centered,
                    "interest_exposure_centered": exposure_centered,
                    "pi_level": float(values["pi"][period]),
                    "Y_level": float(1.0 + values["Y"][period]),
                    "C_level": float(steady["C"] + values["C"][period]),
                    "i_level": float(steady["i"] + values["i"][period]),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = pd.DataFrame(rows)
    shocks = pd.DataFrame(shock_rows)
    paths.to_csv(output_dir / "hank_observables.csv", index=False)
    shocks.to_csv(output_dir / "shock_paths.csv", index=False)
    spec = StochasticPathSpec(
        source_shock_library=str(shock_library_csv),
        output_path=str(output_dir / "hank_observables.csv"),
        trajectory_seeds=tuple(int(seed) for seed in trajectory_seeds),
        shock_standard_deviations={key: float(value) for key, value in shock_standard_deviations.items()},
        shock_persistence={key: float(value) for key, value in shock_persistence.items()},
        note=(
            "Траектории построены свёрткой случайных последовательностей неполитических шоков "
            "с HANK transition responses. Самостоятельная редуцированная динамика не задаётся."
        ),
    )
    (output_dir / "stochastic_paths_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return paths


def _steady_distributional_values(ss, mpc: np.ndarray, config) -> dict[str, float]:
    distribution = stationary_distribution(ss)
    mesh = state_mesh(ss)
    masks = build_group_masks(ss, config, mpc=mpc)["groups"]
    mean_mpc = float(np.sum(distribution * mpc))
    return {
        "C": float(ss["C"]),
        "i": float(ss["i"]),
        "mean_mpc": mean_mpc,
        "share_low_liquidity": float(np.sum(distribution * masks["low_liquid"])),
        "interest_exposure": float(np.sum(distribution * mesh["b"] * mpc)),
    }


def _response_arrays(library: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    responses: dict[str, dict[str, np.ndarray]] = {}
    for (shock, variable), frame in library.groupby(["shock", "variable"], sort=False):
        responses.setdefault(str(shock), {})[str(variable)] = (
            frame.sort_values("period")["response_per_unit"].to_numpy(dtype=float)
        )
    return responses


def _ar1_path(*, rng: np.random.Generator, horizon: int, sigma: float, rho: float) -> np.ndarray:
    innovations = rng.normal(0.0, float(sigma), size=horizon)
    path = np.zeros(horizon, dtype=float)
    for period, innovation in enumerate(innovations):
        path[period] = innovation if period == 0 else float(rho) * path[period - 1] + innovation
    return path


def _convolve_path(shock_path: np.ndarray, response: np.ndarray, horizon: int) -> np.ndarray:
    full = np.convolve(shock_path, response)
    return full[:horizon]
