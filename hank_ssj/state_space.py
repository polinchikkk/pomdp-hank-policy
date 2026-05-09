from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class StateSpaceSpec:
    """Joint linear state-space specification for the HANK/SSJ information problem."""

    state_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    A: np.ndarray
    Q: np.ndarray
    M: np.ndarray
    R: np.ndarray
    initial_mean: np.ndarray
    initial_cov: np.ndarray


def state_space_spec_to_jsonable(spec: StateSpaceSpec) -> dict[str, Any]:
    """Convert a state-space specification with numpy arrays to JSON-safe objects."""

    return {
        "state_names": list(spec.state_names),
        "observation_names": list(spec.observation_names),
        "A": _array_to_list(spec.A),
        "Q": _array_to_list(spec.Q),
        "M": _array_to_list(spec.M),
        "R": _array_to_list(spec.R),
        "initial_mean": _array_to_list(spec.initial_mean),
        "initial_cov": _array_to_list(spec.initial_cov),
    }


def spectral_radius(matrix: np.ndarray) -> float:
    """Return the largest absolute eigenvalue of a square matrix."""

    if matrix.size == 0:
        return 0.0
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def _array_to_list(array: np.ndarray) -> list[Any]:
    return np.asarray(array, dtype=float).tolist()
