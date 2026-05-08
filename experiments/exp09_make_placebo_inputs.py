from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DISTRIBUTIONAL_FEATURES = {
    "observed_distribution": (
        "mean_mpc_obs",
        "low_liquidity_share_obs",
        "interest_exposure_obs",
    ),
    "filtered_distribution": (
        "E_mean_mpc",
        "E_low_liquidity_share",
        "E_interest_exposure",
    ),
    "filtered_distribution_mpc": ("E_mean_mpc",),
    "filtered_distribution_liquidity": ("E_low_liquidity_share",),
    "filtered_distribution_exposure": ("E_interest_exposure",),
}


@dataclass(frozen=True)
class PlaceboInputSpec:
    source_inputs: str
    output_dir: str
    seed: int
    modified_information_states: tuple[str, ...]
    modified_features: dict[str, tuple[str, ...]]
    outputs: dict[str, str]
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Build artificial distributional inputs for falsification tests.")
    parser.add_argument("--information-inputs", default="outputs/ssj/stochastic/information_state_inputs_long.csv")
    parser.add_argument("--output-dir", default="outputs/ssj/stochastic/placebo")
    parser.add_argument("--seed", type=int, default=2031)
    args = parser.parse_args()

    source = pd.read_csv(args.information_inputs)
    _require_columns(source)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    permuted = _permuted_distribution_inputs(source, seed=args.seed)
    fake = _fake_distribution_inputs(source, seed=args.seed + 1)

    permuted_path = output_dir / "information_state_inputs_permuted_distribution_long.csv"
    fake_path = output_dir / "information_state_inputs_fake_distribution_long.csv"
    permuted.to_csv(permuted_path, index=False)
    fake.to_csv(fake_path, index=False)

    spec = PlaceboInputSpec(
        source_inputs=args.information_inputs,
        output_dir=args.output_dir,
        seed=int(args.seed),
        modified_information_states=tuple(DISTRIBUTIONAL_FEATURES),
        modified_features=DISTRIBUTIONAL_FEATURES,
        outputs={
            "permuted_distribution": str(permuted_path),
            "fake_distribution": str(fake_path),
        },
        note=(
            "Меняются только распределительные признаки в observed_distribution и filtered_distribution. "
            "Агрегатные информационные наборы и ориентир полной информации остаются неизменными."
        ),
    )
    (output_dir / "placebo_inputs_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {permuted_path}")
    print(f"Wrote {fake_path}")


def _permuted_distribution_inputs(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result = frame.copy()
    for information_state, feature_names in DISTRIBUTIONAL_FEATURES.items():
        for feature_name in feature_names:
            mask = (result["information_state"] == information_state) & (result["feature_name"] == feature_name)
            subset = result.loc[mask, ["scenario", "period", "observation_seed", "value"]].copy()
            shuffled_parts = []
            for _, group in subset.groupby(["period", "observation_seed"], sort=False):
                values = group["value"].to_numpy(dtype=float).copy()
                rng.shuffle(values)
                changed = group.copy()
                changed["value"] = values
                shuffled_parts.append(changed)
            shuffled = pd.concat(shuffled_parts, ignore_index=False).sort_index()
            result.loc[mask, "value"] = shuffled["value"].to_numpy(dtype=float)
    return result


def _fake_distribution_inputs(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    result = frame.copy()
    scenarios = tuple(sorted(frame["scenario"].unique()))
    observation_seeds = tuple(sorted(frame["observation_seed"].unique()))
    periods = tuple(sorted(frame["period"].unique()))
    horizon = len(periods)

    for information_state, feature_names in DISTRIBUTIONAL_FEATURES.items():
        for feature_name in feature_names:
            mask = (frame["information_state"] == information_state) & (frame["feature_name"] == feature_name)
            values = frame.loc[mask, "value"].to_numpy(dtype=float)
            rho = _estimate_ar1(frame.loc[mask])
            scale = float(np.std(values, ddof=0))
            mean = float(np.mean(values))
            fake_values: dict[tuple[str, int, int], float] = {}
            for scenario in scenarios:
                for observation_seed in observation_seeds:
                    path = _fake_ar1_path(rng=rng, horizon=horizon, mean=mean, scale=scale, rho=rho)
                    for period, value in zip(periods, path):
                        fake_values[(str(scenario), int(observation_seed), int(period))] = float(value)
            result.loc[mask, "value"] = [
                fake_values[(str(row.scenario), int(row.observation_seed), int(row.period))]
                for row in result.loc[mask, ["scenario", "observation_seed", "period"]].itertuples(index=False)
            ]
    return result


def _estimate_ar1(subset: pd.DataFrame) -> float:
    x_lag: list[float] = []
    x_now: list[float] = []
    for _, group in subset.sort_values("period").groupby(["scenario", "observation_seed"], sort=False):
        values = group["value"].to_numpy(dtype=float)
        if values.size > 1:
            x_lag.extend(values[:-1])
            x_now.extend(values[1:])
    if len(x_lag) < 2 or np.std(x_lag) <= 1e-12 or np.std(x_now) <= 1e-12:
        return 0.0
    rho = float(np.corrcoef(np.asarray(x_lag), np.asarray(x_now))[0, 1])
    return float(np.clip(rho, -0.95, 0.95))


def _fake_ar1_path(
    *,
    rng: np.random.Generator,
    horizon: int,
    mean: float,
    scale: float,
    rho: float,
) -> np.ndarray:
    if scale <= 1e-14:
        return np.full(horizon, mean, dtype=float)
    innovation_scale = scale * np.sqrt(max(1.0 - rho**2, 1e-8))
    path = np.zeros(horizon, dtype=float)
    path[0] = rng.normal(mean, scale)
    for period in range(1, horizon):
        path[period] = mean + rho * (path[period - 1] - mean) + rng.normal(0.0, innovation_scale)
    return path


def _require_columns(frame: pd.DataFrame) -> None:
    required = {"scenario", "period", "observation_seed", "information_state", "feature_name", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input table is missing columns: {sorted(missing)}")


if __name__ == "__main__":
    main()
