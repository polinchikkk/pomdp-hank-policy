"""Regime-switching reduced-state HANK baseline."""

from .pipeline import run_pipeline
from .regime_evaluation import evaluate_policy_under_regime_uncertainty, evaluate_regime_filter
from .regime_filter import SwitchingKalmanFilterResults, run_switching_kalman_filter
from .regime_model import (
    RegimeSwitchingConfig,
    RegimeSwitchingModel,
    build_regime_switching_model,
    regime_model_spec_payload,
)
from .regime_simulation import (
    RegimePolicyRun,
    generate_regime_observations,
    simulate_filtered_policy,
    simulate_full_information_policy,
    simulate_hidden_regimes,
)

# Backward-compatible alias with the old experimental package name.
run_stage4_pipeline = run_pipeline

__all__ = [
    "RegimePolicyRun",
    "RegimeSwitchingConfig",
    "RegimeSwitchingModel",
    "SwitchingKalmanFilterResults",
    "build_regime_switching_model",
    "evaluate_policy_under_regime_uncertainty",
    "evaluate_regime_filter",
    "generate_regime_observations",
    "regime_model_spec_payload",
    "run_pipeline",
    "run_stage4_pipeline",
    "run_switching_kalman_filter",
    "simulate_filtered_policy",
    "simulate_full_information_policy",
    "simulate_hidden_regimes",
]
