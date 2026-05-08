from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SSJArtifactSpec:
    """Метаданные для экспортированных якобианов метода последовательностей."""

    source: str
    horizon: int
    input_name: str
    note: str


def export_long_jacobian_to_npz(
    *,
    jacobian_csv: Path,
    output_path: Path,
    spec: SSJArtifactSpec,
) -> None:
    """Convert a long-form Jacobian CSV into a compressed matrix archive.

    The current HANK core writes Jacobians in long format with columns
    ``output``, ``input``, ``response_period``, ``shock_period`` and ``value``.
    The HANK/SSJ information experiment needs matrix artifacts that can be
    consumed by trajectory and observation builders.  This function performs
    only that data-format conversion; it does not relabel a policy shock
    Jacobian as an exogenous interest-rate-path Jacobian.
    """

    frame = pd.read_csv(jacobian_csv)
    required = {"output", "input", "response_period", "shock_period", "value"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Jacobian file is missing required columns: {sorted(missing)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for (input_name, output_name), group in frame.groupby(["input", "output"], sort=True):
        matrix = _matrix_from_long(group)
        arrays[f"J_{input_name}_{output_name}"] = matrix

    arrays["metadata_source"] = np.array(spec.source)
    arrays["metadata_input_name"] = np.array(spec.input_name)
    arrays["metadata_note"] = np.array(spec.note)
    arrays["metadata_horizon"] = np.array(spec.horizon)
    arrays["metadata_json_keys"] = np.array(sorted(asdict(spec).keys()))
    np.savez_compressed(output_path, **arrays)


def _matrix_from_long(group: pd.DataFrame) -> np.ndarray:
    response_periods = group["response_period"].astype(int)
    shock_periods = group["shock_period"].astype(int)
    matrix = np.zeros((int(response_periods.max()) + 1, int(shock_periods.max()) + 1), dtype=float)
    matrix[response_periods.to_numpy(), shock_periods.to_numpy()] = group["value"].to_numpy(dtype=float)
    return matrix
