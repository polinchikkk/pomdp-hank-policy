from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from hank_partial_info_baseline.config import HANKPartialInfoConfig, default_partial_info_config
from hank_partial_info_baseline.state_space import ReducedHANKStateSpaceModel


REGIME_NAMES = ("normal", "stress")


@dataclass(frozen=True)
class RegimeSwitchingConfig:
    output_dir: str = "outputs/hank_regime_switching_stage5"
    random_seed: int = 17
    lambda_y: float = 0.5
    lambda_i: float = 0.05
    regime_transition: tuple[tuple[float, float], tuple[float, float]] = (
        (0.96, 0.04),
        (0.10, 0.90),
    )
    moderate_gap_scale: float = 1.0
    strong_gap_scale: float = 1.75
    stress_inflation_row_factor: float = 1.10
    stress_output_row_factor: float = 1.15
    stress_low_liquidity_row_factor: float = 1.22
    stress_mean_mpc_row_factor: float = 1.18
    stress_inflation_control_factor: float = 1.20
    stress_output_control_factor: float = 1.35
    stress_low_liquidity_control_factor: float = 1.60
    stress_mean_mpc_control_factor: float = 1.45
    stress_macro_noise_factor: float = 1.15
    stress_distribution_noise_factor: float = 1.60
    output_to_inflation_link: float = 0.03
    output_to_low_liquidity_link: float = -0.08
    output_to_mean_mpc_link: float = -0.06
    partial_config: HANKPartialInfoConfig = field(default_factory=default_partial_info_config)

    def scenario_specs(self) -> list[dict]:
        partial_specs = {
            spec["name"]: spec
            for spec in self.partial_config.scenario_specs()
        }
        gap_specs = (
            ("moderate_gap", "Умеренный режимный разрыв", self.moderate_gap_scale),
            ("strong_gap", "Сильный режимный разрыв", self.strong_gap_scale),
        )
        info_names = ("macro_core", "thin_information")
        rows = []
        for info_name in info_names:
            info_spec = partial_specs[info_name]
            for gap_name, gap_label, gap_scale in gap_specs:
                rows.append({
                    "name": f"{info_name}_{gap_name}",
                    "label": f"{info_spec['label']} × {gap_label.lower()}",
                    "info_scenario_name": info_name,
                    "info_scenario_label": info_spec["label"],
                    "gap_name": gap_name,
                    "gap_label": gap_label,
                    "gap_scale": gap_scale,
                    "description": (
                        f"{info_spec['description']} При этом reduced-state HANK dynamics "
                        f"имеют скрытое переключение между режимами `{REGIME_NAMES[0]}` "
                        f"и `{REGIME_NAMES[1]}` с интенсивностью `{gap_label.lower()}`."
                    ),
                    "noisy_observations": tuple(info_spec["noisy_observations"]),
                    "known_exact": tuple(info_spec["known_exact"]),
                    "noise_scale": float(info_spec["noise_scale"]),
                })
        return rows

    def filter_spec_payload(self) -> dict:
        return {
            "filter_type": "interacting_multiple_model_kalman_filter",
            "regime_names": list(REGIME_NAMES),
            "regime_transition_matrix": [list(row) for row in self.regime_transition],
            "note": (
                "Policy instrument is known exactly; noisy macro releases are filtered with an "
                "IMM / switching Kalman filter over reduced-state HANK dynamics."
            ),
        }

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["scenario_specs"] = self.scenario_specs()
        return payload


@dataclass(frozen=True)
class RegimeSwitchingModel:
    state_names: tuple[str, ...]
    observation_names: tuple[str, ...]
    regime_names: tuple[str, ...]
    regime_transition_matrix: np.ndarray
    transition_matrices: np.ndarray
    control_loadings: np.ndarray
    process_noise_covariances: np.ndarray
    observation_matrix: np.ndarray
    steady_state_statistics: dict[str, float]
    base_model_training_summary: dict[str, float]
    gap_scale: float

    def state_index(self, name: str) -> int:
        return self.state_names.index(name)

    def observation_index(self, name: str) -> int:
        return self.observation_names.index(name)

    def num_regimes(self) -> int:
        return len(self.regime_names)

    def initial_state_mean(self) -> np.ndarray:
        return np.zeros(len(self.state_names), dtype=float)

    def stationary_regime_distribution(self) -> np.ndarray:
        matrix = np.asarray(self.regime_transition_matrix, dtype=float)
        eigvals, eigvecs = np.linalg.eig(matrix.T)
        index = int(np.argmin(np.abs(eigvals - 1.0)))
        stationary = np.real(eigvecs[:, index])
        stationary = stationary / np.sum(stationary)
        stationary = np.clip(stationary, 1.0e-10, None)
        return stationary / np.sum(stationary)

    def stationary_state_covariances(self) -> np.ndarray:
        covariances = []
        for regime_index in range(self.num_regimes()):
            transition = self.transition_matrices[regime_index]
            noise_cov = self.process_noise_covariances[regime_index]
            try:
                covariance = solve_discrete_lyapunov(transition, noise_cov)
            except Exception:
                covariance = np.diag(np.maximum(np.diag(noise_cov), 1.0e-8))
            covariance = 0.5 * (covariance + covariance.T)
            eigvals, eigvecs = np.linalg.eigh(covariance)
            eigvals = np.clip(eigvals, 1.0e-10, None)
            covariances.append(eigvecs @ np.diag(eigvals) @ eigvecs.T)
        return np.asarray(covariances, dtype=float)


def _gap_adjustment(base_factor: float, gap_scale: float) -> float:
    return 1.0 + gap_scale * (base_factor - 1.0)


def _stabilize_transition_matrix(matrix: np.ndarray, exogenous_dim: int) -> np.ndarray:
    candidate = np.asarray(matrix, dtype=float).copy()
    for _ in range(25):
        radius = float(np.max(np.abs(np.linalg.eigvals(candidate))))
        if radius < 0.985:
            break
        candidate[exogenous_dim:, :] *= 0.97
    return candidate


def build_regime_switching_model(
    reduced_model: ReducedHANKStateSpaceModel,
    config: RegimeSwitchingConfig,
    gap_scale: float,
) -> RegimeSwitchingModel:
    normal_transition = np.asarray(reduced_model.transition_matrix, dtype=float).copy()
    normal_control = np.asarray(reduced_model.control_loadings, dtype=float).copy()
    normal_noise = np.asarray(reduced_model.process_noise_cov, dtype=float).copy()
    observation_matrix = np.asarray(reduced_model.observation_matrix, dtype=float).copy()
    state_names = tuple(reduced_model.state_names)

    stress_transition = normal_transition.copy()
    stress_control = normal_control.copy()
    stress_noise = normal_noise.copy()

    idx_pi = state_names.index("inflation_gap")
    idx_output = state_names.index("output_gap")
    idx_low_liq = state_names.index("low_liquidity_gap")
    idx_mean_mpc = state_names.index("mean_mpc_gap")
    exogenous_dim = len(reduced_model.exogenous_state_names)

    for index, factor in (
        (idx_pi, _gap_adjustment(config.stress_inflation_row_factor, gap_scale)),
        (idx_output, _gap_adjustment(config.stress_output_row_factor, gap_scale)),
        (idx_low_liq, _gap_adjustment(config.stress_low_liquidity_row_factor, gap_scale)),
        (idx_mean_mpc, _gap_adjustment(config.stress_mean_mpc_row_factor, gap_scale)),
    ):
        stress_transition[index, :] *= factor

    stress_transition[idx_pi, idx_output] += config.output_to_inflation_link * gap_scale
    stress_transition[idx_low_liq, idx_output] += config.output_to_low_liquidity_link * gap_scale
    stress_transition[idx_mean_mpc, idx_output] += config.output_to_mean_mpc_link * gap_scale
    stress_transition = _stabilize_transition_matrix(stress_transition, exogenous_dim)

    for index, factor in (
        (idx_pi, _gap_adjustment(config.stress_inflation_control_factor, gap_scale)),
        (idx_output, _gap_adjustment(config.stress_output_control_factor, gap_scale)),
        (idx_low_liq, _gap_adjustment(config.stress_low_liquidity_control_factor, gap_scale)),
        (idx_mean_mpc, _gap_adjustment(config.stress_mean_mpc_control_factor, gap_scale)),
    ):
        stress_control[index] *= factor

    for index in range(exogenous_dim):
        stress_noise[index, index] *= _gap_adjustment(config.stress_macro_noise_factor, gap_scale)
    for index in (idx_low_liq, idx_mean_mpc):
        stress_noise[index, index] *= _gap_adjustment(config.stress_distribution_noise_factor, gap_scale)
    stress_noise += 1.0e-12 * np.eye(len(state_names), dtype=float)

    regime_transition_matrix = np.asarray(config.regime_transition, dtype=float)
    return RegimeSwitchingModel(
        state_names=state_names,
        observation_names=tuple(reduced_model.observation_names),
        regime_names=REGIME_NAMES,
        regime_transition_matrix=regime_transition_matrix,
        transition_matrices=np.stack([normal_transition, stress_transition], axis=0),
        control_loadings=np.stack([normal_control, stress_control], axis=0),
        process_noise_covariances=np.stack([normal_noise, stress_noise], axis=0),
        observation_matrix=observation_matrix,
        steady_state_statistics=dict(reduced_model.steady_state_statistics),
        base_model_training_summary=dict(reduced_model.training_summary),
        gap_scale=float(gap_scale),
    )


def regime_model_spec_payload(model: RegimeSwitchingModel, scenario: dict) -> dict:
    return {
        "stage": "stage5_regime_switching_reduced_hank",
        "regime_names": list(model.regime_names),
        "scenario_name": scenario["name"],
        "scenario_label": scenario["label"],
        "gap_name": scenario["gap_name"],
        "gap_scale": float(scenario["gap_scale"]),
        "regime_transition_matrix": model.regime_transition_matrix.tolist(),
        "state_names": list(model.state_names),
        "observation_names": list(model.observation_names),
        "info_scenario_name": scenario["info_scenario_name"],
        "noisy_observations": list(scenario["noisy_observations"]),
        "description": (
            "Two-regime reduced-state HANK overlay. Normal regime uses the stage-3 local linear "
            "approximation; stress regime amplifies policy transmission and distributional dynamics."
        ),
        "steady_state_statistics": model.steady_state_statistics,
        "base_model_training_summary": model.base_model_training_summary,
    }
