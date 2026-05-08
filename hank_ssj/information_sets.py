from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class InformationSetSpec:
    name: str
    inputs: tuple[str, ...]
    role: str


@dataclass(frozen=True)
class InformationStateInputSpec:
    source_observables: str
    source_observations: str
    source_filtered_states: str | None
    information_states: tuple[str, ...]
    note: str


INFORMATION_SETS = (
    InformationSetSpec(
        name="aggregate_only",
        inputs=("pi_obs", "Y_obs"),
        role="Базовый набор агрегатных наблюдений.",
    ),
    InformationSetSpec(
        name="aggregate_history",
        inputs=("pi_obs_lags", "Y_obs_lags", "i_lag"),
        role="Проверяет, может ли короткая история агрегатов заменить фильтрацию.",
    ),
    InformationSetSpec(
        name="filtered_aggregates",
        inputs=("E_pi", "E_Y", "E_C"),
        role="Ценность фильтрации агрегатного состояния.",
    ),
    InformationSetSpec(
        name="observed_distribution",
        inputs=("pi_obs", "Y_obs", "mean_mpc_obs", "low_liquidity_share_obs"),
        role="Ценность шумных распределительных индикаторов.",
    ),
    InformationSetSpec(
        name="filtered_distribution",
        inputs=("E_pi", "E_Y", "E_C", "E_mean_mpc", "E_low_liquidity_share", "E_interest_exposure"),
        role="Главный объект: предельная ценность распределительной информации сверх фильтрованных агрегатов.",
    ),
    InformationSetSpec(
        name="filtered_distribution_mpc",
        inputs=("E_pi", "E_Y", "E_C", "E_mean_mpc"),
        role="Ценность средней предельной склонности к потреблению сверх фильтрованных агрегатов.",
    ),
    InformationSetSpec(
        name="filtered_distribution_liquidity",
        inputs=("E_pi", "E_Y", "E_C", "E_low_liquidity_share"),
        role="Ценность доли низколиквидных домохозяйств сверх фильтрованных агрегатов.",
    ),
    InformationSetSpec(
        name="filtered_distribution_exposure",
        inputs=("E_pi", "E_Y", "E_C", "E_interest_exposure"),
        role="Ценность процентной экспозиции сверх фильтрованных агрегатов.",
    ),
    InformationSetSpec(
        name="full_information",
        inputs=("pi", "Y", "C", "mean_mpc", "low_liquidity_share", "interest_exposure"),
        role="Верхняя граница качества, не реалистичный информационный режим.",
    ),
)


def build_information_state_inputs(
    *,
    observables_csv: Path,
    observations_csv: Path,
    filtered_states_csv: Path | None = None,
    output_dir: Path,
) -> pd.DataFrame:
    """Build long-form inputs for policy rules from HANK/SSJ observations."""

    observables = pd.read_csv(observables_csv)
    observations = pd.read_csv(observations_csv)
    keys = ["scenario", "scenario_label", "period"]
    _require_columns(observables, {*keys, "pi", "Y", "C", "i", "mean_mpc_centered", "share_low_liquidity_centered", "interest_exposure_centered"}, observables_csv)
    _require_columns(observations, {*keys, "observation_seed", "pi_obs", "Y_obs", "C_obs", "mean_mpc_centered_obs", "share_low_liquidity_centered_obs", "interest_exposure_centered_obs"}, observations_csv)

    merged = observations.merge(
        observables[
            [
                *keys,
                "pi",
                "Y",
                "C",
                "i",
                "mean_mpc_centered",
                "share_low_liquidity_centered",
                "interest_exposure_centered",
            ]
        ],
        on=keys,
        how="inner",
        validate="many_to_one",
    ).sort_values(["scenario", "observation_seed", "period"])
    merged["i_lag"] = merged.groupby(["scenario", "observation_seed"])["i"].shift(1).fillna(0.0)
    merged["pi_obs_lag"] = merged.groupby(["scenario", "observation_seed"])["pi_obs"].shift(1).fillna(0.0)
    merged["Y_obs_lag"] = merged.groupby(["scenario", "observation_seed"])["Y_obs"].shift(1).fillna(0.0)
    if filtered_states_csv is not None:
        filtered = pd.read_csv(filtered_states_csv)
        _require_columns(
            filtered,
            {
                "scenario",
                "scenario_label",
                "period",
                "observation_seed",
                "E_pi",
                "E_Y",
                "E_C",
                "E_mean_mpc",
                "E_low_liquidity_share",
                "E_interest_exposure",
            },
            filtered_states_csv,
        )
        merged = merged.merge(
            filtered[
                [
                    "scenario",
                    "scenario_label",
                    "period",
                    "observation_seed",
                    "E_pi",
                    "E_Y",
                    "E_C",
                    "E_mean_mpc",
                    "E_low_liquidity_share",
                    "E_interest_exposure",
                ]
            ],
            on=["scenario", "scenario_label", "period", "observation_seed"],
            how="inner",
            validate="one_to_one",
        )

    rows: list[dict[str, object]] = []
    for record in merged.to_dict(orient="records"):
        base = {
            "scenario": record["scenario"],
            "scenario_label": record["scenario_label"],
            "period": int(record["period"]),
            "observation_seed": int(record["observation_seed"]),
        }
        _append_state(
            rows,
            base,
            "aggregate_only",
            {
                "pi_obs": record["pi_obs"],
                "Y_obs": record["Y_obs"],
            },
        )
        _append_state(
            rows,
            base,
            "aggregate_history",
            {
                "pi_obs": record["pi_obs"],
                "Y_obs": record["Y_obs"],
                "pi_obs_lag": record["pi_obs_lag"],
                "Y_obs_lag": record["Y_obs_lag"],
                "i_lag": record["i_lag"],
            },
        )
        _append_state(
            rows,
            base,
            "observed_distribution",
            {
                "pi_obs": record["pi_obs"],
                "Y_obs": record["Y_obs"],
                "C_obs": record["C_obs"],
                "mean_mpc_obs": record["mean_mpc_centered_obs"],
                "low_liquidity_share_obs": record["share_low_liquidity_centered_obs"],
                "interest_exposure_obs": record["interest_exposure_centered_obs"],
            },
        )
        if filtered_states_csv is not None:
            _append_state(
                rows,
                base,
                "filtered_aggregates",
                {
                    "E_pi": record["E_pi"],
                    "E_Y": record["E_Y"],
                    "E_C": record["E_C"],
                },
            )
            _append_state(
                rows,
                base,
                "filtered_distribution",
                {
                    "E_pi": record["E_pi"],
                    "E_Y": record["E_Y"],
                    "E_C": record["E_C"],
                    "E_mean_mpc": record["E_mean_mpc"],
                    "E_low_liquidity_share": record["E_low_liquidity_share"],
                    "E_interest_exposure": record["E_interest_exposure"],
                },
            )
            _append_state(
                rows,
                base,
                "filtered_distribution_mpc",
                {
                    "E_pi": record["E_pi"],
                    "E_Y": record["E_Y"],
                    "E_C": record["E_C"],
                    "E_mean_mpc": record["E_mean_mpc"],
                },
            )
            _append_state(
                rows,
                base,
                "filtered_distribution_liquidity",
                {
                    "E_pi": record["E_pi"],
                    "E_Y": record["E_Y"],
                    "E_C": record["E_C"],
                    "E_low_liquidity_share": record["E_low_liquidity_share"],
                },
            )
            _append_state(
                rows,
                base,
                "filtered_distribution_exposure",
                {
                    "E_pi": record["E_pi"],
                    "E_Y": record["E_Y"],
                    "E_C": record["E_C"],
                    "E_interest_exposure": record["E_interest_exposure"],
                },
            )
        _append_state(
            rows,
            base,
            "full_information",
            {
                "pi": record["pi"],
                "Y": record["Y"],
                "C": record["C"],
                "mean_mpc": record["mean_mpc_centered"],
                "low_liquidity_share": record["share_low_liquidity_centered"],
                "interest_exposure": record["interest_exposure_centered"],
            },
        )

    long = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(output_dir / "information_state_inputs_long.csv", index=False)
    spec = InformationStateInputSpec(
        source_observables=str(observables_csv),
        source_observations=str(observations_csv),
        source_filtered_states=str(filtered_states_csv) if filtered_states_csv is not None else None,
        information_states=tuple(sorted(long["information_state"].unique())),
        note="Входы правил собраны из HANK/SSJ-наблюдений, фильтрованных оценок и истинных HANK/SSJ-переменных.",
    )
    (output_dir / "information_state_inputs_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return long


def _append_state(
    rows: list[dict[str, object]],
    base: dict[str, object],
    information_state: str,
    features: dict[str, object],
) -> None:
    for feature_name, value in features.items():
        rows.append(
            {
                **base,
                "information_state": information_state,
                "feature_name": feature_name,
                "value": float(value),
            }
        )


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
