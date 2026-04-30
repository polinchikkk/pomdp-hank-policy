from __future__ import annotations

from regime_switching_baseline.regime_model import RegimeSwitchingConfig


def extreme_sticky_regime_config() -> RegimeSwitchingConfig:
    return RegimeSwitchingConfig(
        regime_transition=((0.975, 0.025), (0.02, 0.98)),
        moderate_gap_scale=1.0,
        strong_gap_scale=3.0,
        stress_inflation_row_factor=1.18,
        stress_output_row_factor=1.30,
        stress_low_liquidity_row_factor=1.45,
        stress_mean_mpc_row_factor=1.35,
        stress_inflation_control_factor=1.40,
        stress_output_control_factor=1.80,
        stress_low_liquidity_control_factor=2.50,
        stress_mean_mpc_control_factor=2.00,
        stress_macro_noise_factor=1.40,
        stress_distribution_noise_factor=2.30,
    )
