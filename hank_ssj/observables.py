from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ObservableSet:
    name: str
    variables: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class HankObservableBuildSpec:
    source_aggregate_paths: str
    source_distribution_paths: str
    output_variables: tuple[str, ...]
    interest_exposure_definition: str


AGGREGATE_OBSERVABLES = ObservableSet(
    name="aggregate_only",
    variables=("pi", "Y", "C"),
    description="Агрегатные макроэкономические индикаторы без распределительных статистик.",
)

DISTRIBUTIONAL_OBSERVABLES = ObservableSet(
    name="distributional",
    variables=("pi", "Y", "C", "mean_mpc", "low_liquidity_share", "interest_exposure"),
    description="Агрегаты плюс распределительные статистики, полученные из HANK-состояния.",
)

FULL_INFORMATION_OUTPUTS = ObservableSet(
    name="full_information",
    variables=DISTRIBUTIONAL_OBSERVABLES.variables,
    description="Верхняя граница: истинные переменные, порождённые HANK/SSJ, без шума наблюдения.",
)


def build_hank_observable_panel(*, hank_core_dir: Path, output_dir: Path) -> pd.DataFrame:
    r"""Build the HANK/SSJ observable panel used by imperfect-information experiments.

    The panel is built from HANK-generated transition paths.  It does not impose
    an additional law of motion; it only selects and renames HANK outputs into a
    compact sequence \(q_t\).
    """

    aggregate_path = hank_core_dir / "aggregate_paths.csv"
    distribution_path = hank_core_dir / "distribution_paths.csv"
    aggregate = pd.read_csv(aggregate_path)
    distribution = pd.read_csv(distribution_path)

    keys = ["scenario", "scenario_label", "period"]
    required_aggregate = {
        *keys,
        "pi_deviation",
        "Y_deviation",
        "output_gap_deviation",
        "C_deviation",
        "i_deviation",
        "pi_level",
        "Y_level",
        "C_level",
        "i_level",
    }
    required_distribution = {
        *keys,
        "mean_mpc",
        "share_low_liquidity",
        "mean_liquid_wealth",
    }
    _require_columns(aggregate, required_aggregate, aggregate_path)
    _require_columns(distribution, required_distribution, distribution_path)

    distribution_columns = [*keys, "mean_mpc", "share_low_liquidity", "mean_liquid_wealth"]
    uses_direct_exposure = "interest_exposure" in distribution.columns
    if uses_direct_exposure:
        distribution_columns.append("interest_exposure")

    frame = aggregate[
        [
            *keys,
            "pi_deviation",
            "Y_deviation",
            "output_gap_deviation",
            "C_deviation",
            "i_deviation",
            "pi_level",
            "Y_level",
            "C_level",
            "i_level",
        ]
    ].merge(
        distribution[distribution_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    frame = frame.rename(
        columns={
            "pi_deviation": "pi",
            "Y_deviation": "Y",
            "output_gap_deviation": "output_gap",
            "C_deviation": "C",
            "i_deviation": "i",
        }
    )
    exposure_definition = "direct HANK moment: sum_t D_t * b_t * MPC_t"
    if not uses_direct_exposure:
        frame["interest_exposure"] = frame["mean_liquid_wealth"] * frame["mean_mpc"]
        exposure_definition = "fallback proxy: mean_liquid_wealth * mean_mpc"
    for column in ("mean_mpc", "share_low_liquidity", "interest_exposure"):
        frame[f"{column}_centered"] = frame[column] - frame.groupby("scenario")[column].transform("first")

    ordered_columns = [
        "scenario",
        "scenario_label",
        "period",
        "pi",
        "Y",
        "output_gap",
        "C",
        "i",
        "mean_mpc",
        "share_low_liquidity",
        "interest_exposure",
        "mean_mpc_centered",
        "share_low_liquidity_centered",
        "interest_exposure_centered",
        "pi_level",
        "Y_level",
        "C_level",
        "i_level",
    ]
    frame = frame[ordered_columns].sort_values(["scenario", "period"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "hank_observables.csv", index=False)
    spec = HankObservableBuildSpec(
        source_aggregate_paths=str(aggregate_path),
        source_distribution_paths=str(distribution_path),
        output_variables=tuple(ordered_columns[3:14]),
        interest_exposure_definition=exposure_definition,
    )
    (output_dir / "hank_observables_spec.json").write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return frame


def _require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
