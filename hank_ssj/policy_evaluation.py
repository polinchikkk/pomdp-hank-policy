from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from policy.linear_rules import LinearRule


@dataclass(frozen=True)
class PolicyLossWeights:
    inflation: float = 1.0
    output_gap: float = 1.0
    consumption: float = 0.25
    rate_smoothing: float = 0.1


@dataclass(frozen=True)
class TrajectoryLoss:
    total_loss: float
    inflation_loss: float
    output_gap_loss: float
    consumption_loss: float
    rate_smoothing_loss: float
    stability_penalty: float


@dataclass(frozen=True)
class SSJPolicyEvaluationSpec:
    information_inputs: str
    hank_observables: str
    jacobians: str
    note: str


class HankSSJPolicyEnvironment:
    """Evaluate simple rate rules on HANK/SSJ-implied trajectories.

    The evaluator does not introduce a separate reduced economic law. It uses
    HANK/SSJ observables as the baseline path and the SSJ response to a monetary
    policy shock as a local projection for alternative rate paths.
    """

    def __init__(
        self,
        *,
        information_inputs: pd.DataFrame,
        observables: pd.DataFrame,
        jacobians: dict[str, np.ndarray],
        loss_weights: PolicyLossWeights | None = None,
        ridge: float = 1e-10,
        discount: float = 1.0,
        max_abs_rate: float = 0.05,
        max_abs_rate_change: float = 0.03,
        stability_penalty_weight: float = 1e4,
    ) -> None:
        self.loss_weights = PolicyLossWeights() if loss_weights is None else loss_weights
        self.discount = float(discount)
        self.max_abs_rate = float(max_abs_rate)
        self.max_abs_rate_change = float(max_abs_rate_change)
        self.stability_penalty_weight = float(stability_penalty_weight)

        self._features = _wide_features(information_inputs)
        self._feature_matrix_cache: dict[tuple[str, str, int, tuple[str, ...]], np.ndarray] = {}
        self._observables = _observable_paths(observables)
        self.scenarios = tuple(sorted({key[0] for key in self._observables}))
        self.seeds = tuple(sorted({key[2] for key in self._features}))
        self.periods = int(max(len(frame) for frame in self._observables.values()))
        self._effects = _rate_path_effects(jacobians, periods=self.periods, ridge=ridge)
        self._optimal_rate_cache: dict[str, np.ndarray] = {}
        self._ridge = float(ridge)

    @classmethod
    def from_files(
        cls,
        *,
        information_inputs_csv: Path,
        hank_observables_csv: Path,
        jacobians_npz: Path,
        **kwargs,
    ) -> "HankSSJPolicyEnvironment":
        information_inputs = pd.read_csv(information_inputs_csv)
        observables = pd.read_csv(hank_observables_csv)
        with np.load(jacobians_npz) as bundle:
            jacobians = {
                key: np.asarray(bundle[key], dtype=float)
                for key in bundle.files
                if key.startswith("J_")
            }
        return cls(
            information_inputs=information_inputs,
            observables=observables,
            jacobians=jacobians,
            **kwargs,
        )

    def feature_scales(
        self,
        *,
        policy: LinearRule,
        information_state: str,
        seeds: list[int],
    ) -> dict[str, float]:
        del policy
        frames = []
        seed_set = set(seeds)
        for (scenario, state, seed), frame in self._features.items():
            if state == information_state and seed in seed_set:
                frames.append(frame)
        if not frames:
            raise ValueError(f"No feature data for {information_state} and seeds {sorted(seed_set)}.")
        values = pd.concat(frames, ignore_index=True)
        return {
            name: float(max(values[name].std(ddof=0), 1e-5))
            for name in values.columns
            if name not in {"period"}
        }

    def simulate(self, *, policy: LinearRule, information_state: str, seed: int) -> TrajectoryLoss:
        losses = [
            self.simulate_scenario(
                policy=policy,
                information_state=information_state,
                scenario=scenario,
                seed=seed,
            )
            for scenario in self.scenarios
        ]
        return _mean_losses(losses)

    def simulate_scenario(
        self,
        *,
        policy: LinearRule,
        information_state: str,
        scenario: str,
        seed: int,
    ) -> TrajectoryLoss:
        feature_key = (scenario, information_state, int(seed))
        if feature_key not in self._features:
            raise ValueError(f"Missing features for {feature_key}.")
        observable_key = (scenario,)
        if observable_key not in self._observables:
            raise ValueError(f"Missing HANK/SSJ observables for {scenario}.")

        feature_matrix = self._feature_matrix(feature_key, policy.spec.feature_names)
        base = self._observables[observable_key].sort_values("period").reset_index(drop=True)
        periods = min(feature_matrix.shape[0], len(base), self.periods)
        feature_matrix = feature_matrix[:periods]
        base = base.iloc[:periods]

        policy_rate = np.zeros(periods, dtype=float)
        lagged_rate = 0.0
        deterministic_part = policy.intercept + feature_matrix @ np.asarray(policy.coefficients, dtype=float)
        for period in range(periods):
            policy_rate[period] = deterministic_part[period]
            if policy.spec.includes_lagged_rate:
                policy_rate[period] += policy.lagged_rate_weight * lagged_rate
            lagged_rate = float(policy_rate[period])

        baseline_rate = base["i"].to_numpy(dtype=float)
        rate_change = policy_rate - np.r_[0.0, policy_rate[:-1]]
        rate_deviation = policy_rate - baseline_rate

        pi = base["pi"].to_numpy(dtype=float) + self._effects["pi"][:periods, :periods] @ rate_deviation
        y = base["output_gap"].to_numpy(dtype=float) + self._effects["output_gap"][:periods, :periods] @ rate_deviation
        c = base["C"].to_numpy(dtype=float) + self._effects["C"][:periods, :periods] @ rate_deviation

        discounts = self.discount ** np.arange(periods)
        w = self.loss_weights
        inflation_loss = float(np.sum(discounts * w.inflation * pi**2))
        output_gap_loss = float(np.sum(discounts * w.output_gap * y**2))
        consumption_loss = float(np.sum(discounts * w.consumption * c**2))
        rate_smoothing_loss = float(np.sum(discounts * w.rate_smoothing * rate_change**2))
        stability_penalty = self._stability_penalty(policy_rate, rate_change)
        total = inflation_loss + output_gap_loss + consumption_loss + rate_smoothing_loss + stability_penalty
        return TrajectoryLoss(
            total_loss=total,
            inflation_loss=inflation_loss,
            output_gap_loss=output_gap_loss,
            consumption_loss=consumption_loss,
            rate_smoothing_loss=rate_smoothing_loss,
            stability_penalty=stability_penalty,
        )

    def feature_matrix(
        self,
        *,
        scenario: str,
        information_state: str,
        seed: int,
        feature_names: tuple[str, ...],
    ) -> np.ndarray:
        return self._feature_matrix((scenario, information_state, int(seed)), feature_names)

    def optimal_rate_path(self, *, scenario: str) -> np.ndarray:
        """Return the local SSJ-optimal rate path for a baseline HANK trajectory."""

        if scenario in self._optimal_rate_cache:
            return self._optimal_rate_cache[scenario]
        observable_key = (scenario,)
        if observable_key not in self._observables:
            raise ValueError(f"Missing HANK/SSJ observables for {scenario}.")
        base = self._observables[observable_key].sort_values("period").reset_index(drop=True)
        periods = min(len(base), self.periods)
        base = base.iloc[:periods]
        baseline_rate = base["i"].to_numpy(dtype=float)

        diff = np.eye(periods) - np.eye(periods, k=-1)
        diff[0, :] = 0.0
        diff[0, 0] = 1.0

        weights = self.loss_weights
        blocks = [
            np.sqrt(weights.inflation) * self._effects["pi"][:periods, :periods],
            np.sqrt(weights.output_gap) * self._effects["output_gap"][:periods, :periods],
            np.sqrt(weights.consumption) * self._effects["C"][:periods, :periods],
            np.sqrt(weights.rate_smoothing) * diff,
        ]
        target = [
            np.sqrt(weights.inflation)
            * (base["pi"].to_numpy(dtype=float) - self._effects["pi"][:periods, :periods] @ baseline_rate),
            np.sqrt(weights.output_gap)
            * (base["output_gap"].to_numpy(dtype=float) - self._effects["output_gap"][:periods, :periods] @ baseline_rate),
            np.sqrt(weights.consumption)
            * (base["C"].to_numpy(dtype=float) - self._effects["C"][:periods, :periods] @ baseline_rate),
            np.zeros(periods, dtype=float),
        ]
        design = np.vstack(blocks)
        offset = np.concatenate(target)
        rate = -np.linalg.solve(
            design.T @ design + self._ridge * np.eye(periods),
            design.T @ offset,
        )
        self._optimal_rate_cache[scenario] = rate
        return rate

    def _feature_matrix(
        self,
        feature_key: tuple[str, str, int],
        feature_names: tuple[str, ...],
    ) -> np.ndarray:
        cache_key = (*feature_key, feature_names)
        if cache_key in self._feature_matrix_cache:
            return self._feature_matrix_cache[cache_key]
        frame = self._features[feature_key].sort_values("period").reset_index(drop=True)
        missing = [name for name in feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing features for {feature_key}: {missing}")
        matrix = frame.loc[:, list(feature_names)].to_numpy(dtype=float)
        self._feature_matrix_cache[cache_key] = matrix
        return matrix

    def _stability_penalty(self, policy_rate: np.ndarray, rate_change: np.ndarray) -> float:
        rate_excess = np.maximum(np.abs(policy_rate) - self.max_abs_rate, 0.0)
        change_excess = np.maximum(np.abs(rate_change) - self.max_abs_rate_change, 0.0)
        return float(self.stability_penalty_weight * (np.sum(rate_excess**2) + np.sum(change_excess**2)))


def _wide_features(long: pd.DataFrame) -> dict[tuple[str, str, int], pd.DataFrame]:
    required = {"scenario", "period", "observation_seed", "information_state", "feature_name", "value"}
    missing = required.difference(long.columns)
    if missing:
        raise ValueError(f"Information input table is missing columns: {sorted(missing)}")
    wide = (
        long.pivot_table(
            index=["scenario", "information_state", "observation_seed", "period"],
            columns="feature_name",
            values="value",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["scenario", "information_state", "observation_seed", "period"])
    )
    result: dict[tuple[str, str, int], pd.DataFrame] = {}
    for (scenario, state, seed), frame in wide.groupby(["scenario", "information_state", "observation_seed"], sort=False):
        result[(str(scenario), str(state), int(seed))] = frame.drop(
            columns=["scenario", "information_state", "observation_seed"]
        ).reset_index(drop=True)
    return result


def _observable_paths(observables: pd.DataFrame) -> dict[tuple[str], pd.DataFrame]:
    required = {"scenario", "period", "pi", "output_gap", "C", "i"}
    missing = required.difference(observables.columns)
    if missing:
        raise ValueError(f"HANK/SSJ observable table is missing columns: {sorted(missing)}")
    result: dict[tuple[str], pd.DataFrame] = {}
    for scenario, frame in observables.groupby("scenario", sort=False):
        result[(str(scenario),)] = frame[["period", "pi", "output_gap", "C", "i"]].sort_values("period").reset_index(drop=True)
    return result


def _rate_path_effects(jacobians: dict[str, np.ndarray], *, periods: int, ridge: float) -> dict[str, np.ndarray]:
    rate_key = "J_monetary_policy_shock_i"
    if rate_key not in jacobians:
        raise ValueError(f"Jacobian bundle does not contain {rate_key}.")
    rate_response = jacobians[rate_key][:periods, :periods]
    normal = rate_response.T @ rate_response + float(ridge) * np.eye(periods)
    shock_from_rate = np.linalg.solve(normal, rate_response.T)

    effects: dict[str, np.ndarray] = {}
    for variable in ("pi", "output_gap", "C"):
        key = f"J_monetary_policy_shock_{variable}"
        if key not in jacobians:
            raise ValueError(f"Jacobian bundle does not contain {key}.")
        effects[variable] = jacobians[key][:periods, :periods] @ shock_from_rate
    return effects


def _mean_losses(losses: list[TrajectoryLoss]) -> TrajectoryLoss:
    if not losses:
        raise ValueError("Cannot average an empty loss list.")
    fields = TrajectoryLoss.__dataclass_fields__.keys()
    values = {
        field: float(np.mean([getattr(loss, field) for loss in losses]))
        for field in fields
    }
    return TrajectoryLoss(**values)
