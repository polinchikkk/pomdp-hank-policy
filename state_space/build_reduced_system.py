from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReducedStateSpec:
    aggregate_state: tuple[str, ...]
    distribution_state: tuple[str, ...]

    @property
    def full_state(self) -> tuple[str, ...]:
        return self.aggregate_state + self.distribution_state


def default_reduced_state_spec() -> ReducedStateSpec:
    return ReducedStateSpec(
        aggregate_state=("inflation_gap", "output_gap", "natural_rate_gap"),
        distribution_state=("mean_mpc", "low_liquidity_share"),
    )
