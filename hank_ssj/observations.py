from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ObservationNoiseSpec:
    seeds: tuple[int, ...]
    noise_scale: float
    aggregate_noise_scale: float
    distribution_noise_scale: float
    variables: tuple[str, ...]
    scale_floor: float


def build_noisy_observations(
    *,
    observables_csv: Path,
    output_dir: Path,
    seeds: tuple[int, ...] = tuple(range(900, 950)),
    noise_scale: float = 1.0,
    aggregate_noise_scale: float | None = None,
    distribution_noise_scale: float | None = None,
    noise_reference_csv: Path | None = None,
    scale_floor: float = 1e-5,
) -> pd.DataFrame:
    r"""Build noisy observations \(o_t=Mq_t+\nu_t\) from HANK/SSJ observables."""

    panel = pd.read_csv(observables_csv)
    scale_panel = panel if noise_reference_csv is None else pd.read_csv(noise_reference_csv)
    observed_variables = (
        "pi",
        "Y",
        "C",
        "mean_mpc_centered",
        "share_low_liquidity_centered",
        "interest_exposure_centered",
    )
    aggregate_variables = {"pi", "Y", "C"}
    distribution_variables = set(observed_variables).difference(aggregate_variables)
    aggregate_noise_scale = noise_scale if aggregate_noise_scale is None else aggregate_noise_scale
    distribution_noise_scale = noise_scale if distribution_noise_scale is None else distribution_noise_scale
    missing = sorted(set(observed_variables).difference(panel.columns))
    if missing:
        raise ValueError(f"{observables_csv} is missing required columns: {missing}")
    scale_missing = sorted(set(observed_variables).difference(scale_panel.columns))
    if scale_missing:
        raise ValueError(f"{noise_reference_csv} is missing required columns: {scale_missing}")

    base_scales = {
        variable: max(float(scale_panel[variable].std(ddof=0)), scale_floor)
        for variable in observed_variables
    }
    observation_scales = {
        variable: base_scales[variable]
        * (aggregate_noise_scale if variable in aggregate_variables else distribution_noise_scale)
        for variable in observed_variables
    }
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        noisy = panel[["scenario", "scenario_label", "period"]].copy()
        noisy["observation_seed"] = seed
        for variable in observed_variables:
            shock = rng.normal(
                loc=0.0,
                scale=observation_scales[variable],
                size=len(panel),
            )
            noisy[f"{variable}_obs"] = panel[variable].to_numpy(dtype=float) + shock
        frames.append(noisy)

    observations = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output_dir / "hank_observations.csv", index=False)
    spec = ObservationNoiseSpec(
        seeds=seeds,
        noise_scale=noise_scale,
        aggregate_noise_scale=float(aggregate_noise_scale),
        distribution_noise_scale=float(distribution_noise_scale),
        variables=observed_variables,
        scale_floor=scale_floor,
    )
    spec_payload = {
        **asdict(spec),
        "base_scales": base_scales,
        "observation_scales": observation_scales,
        "source_observables": str(observables_csv),
        "noise_reference_observables": str(noise_reference_csv) if noise_reference_csv is not None else str(observables_csv),
    }
    (output_dir / "hank_observations_spec.json").write_text(
        json.dumps(spec_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return observations
