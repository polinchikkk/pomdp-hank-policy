from __future__ import annotations

from dataclasses import asdict, dataclass


STATE_NAMES = (
    "rstar_gap",
    "productivity_gap",
    "fiscal_gap",
    "inflation_gap",
    "output_gap",
    "low_liquidity_gap",
    "mean_mpc_gap",
)

NOISY_OBSERVATION_NAMES = (
    "pi",
    "output_gap",
    "C",
    "w",
    "N",
    "share_low_liquidity",
    "mean_mpc",
)

OBSERVATION_LABELS = {
    "pi": "Инфляция",
    "output_gap": "Разрыв выпуска",
    "C": "Потребление",
    "w": "Реальная заработная плата",
    "N": "Занятость",
    "share_low_liquidity": "Доля низколиквидных домохозяйств",
    "mean_mpc": "Средняя MPC",
    "i": "Номинальная ставка",
}


@dataclass(frozen=True)
class HANKPartialInfoConfig:
    output_dir: str = "outputs/hank_partial_info_stage3"
    horizon: int = 60
    response_horizon: int = 60
    training_periods: int = 1200
    training_burn_in: int = 120
    random_seed: int = 7
    impulse_scale: float = 0.001
    base_policy_shock_size: float = 0.001
    base_policy_shock_period: int = 0
    base_policy_shock_persistence: float = 0.0
    rstar_rho: float = 0.85
    productivity_rho: float = 0.80
    fiscal_rho: float = 0.70
    rstar_std: float = 0.00020
    productivity_std: float = 0.00050
    fiscal_std: float = 0.00035
    training_policy_std: float = 0.00050
    training_policy_rho: float = 0.25
    lambda_y: float = 0.5
    lambda_i: float = 0.05
    confidence_scale: float = 1.96

    def base_measurement_noise(self) -> dict[str, float]:
        return {
            "pi": 0.00015,
            "output_gap": 0.00040,
            "C": 0.00035,
            "w": 0.00030,
            "N": 0.00035,
            "share_low_liquidity": 0.00300,
            "mean_mpc": 0.00150,
        }

    def scenario_specs(self) -> list[dict]:
        return [
            {
                "name": "macro_core",
                "label": "Фильтрация: инфляция, выпуск и ставка",
                "description": (
                    "Базовый policy-relevant набор наблюдений: инфляция и разрыв выпуска "
                    "наблюдаются с шумом, ставка известна как инструмент политики."
                ),
                "noisy_observations": ("pi", "output_gap"),
                "known_exact": ("i",),
                "noise_scale": 1.0,
                "includes_distribution_stats": False,
            },
            {
                "name": "full_macro",
                "label": "Фильтрация: расширенный макронабор",
                "description": (
                    "К инфляции и выпуску добавлены потребление и реальная заработная плата."
                ),
                "noisy_observations": ("pi", "output_gap", "C", "w"),
                "known_exact": ("i",),
                "noise_scale": 1.0,
                "includes_distribution_stats": False,
            },
            {
                "name": "thin_information",
                "label": "Фильтрация: инфляция и ставка",
                "description": (
                    "Тонкий информационный набор: регулятор видит только инфляцию "
                    "и собственную ставку."
                ),
                "noisy_observations": ("pi",),
                "known_exact": ("i",),
                "noise_scale": 1.0,
                "includes_distribution_stats": False,
            },
            {
                "name": "high_noise",
                "label": "Фильтрация: высокий шум измерения",
                "description": (
                    "Базовый набор инфляция-выпуск-ставка, но с более высоким шумом "
                    "в доступных макронаблюдениях."
                ),
                "noisy_observations": ("pi", "output_gap"),
                "known_exact": ("i",),
                "noise_scale": 2.0,
                "includes_distribution_stats": False,
            },
            {
                "name": "distribution_augmented",
                "label": "Фильтрация: с распределительной статистикой",
                "description": (
                    "К инфляции, выпуску и ставке добавлены наблюдения доли "
                    "низколиквидных домохозяйств и средней MPC."
                ),
                "noisy_observations": ("pi", "output_gap", "share_low_liquidity", "mean_mpc"),
                "known_exact": ("i",),
                "noise_scale": 1.0,
                "includes_distribution_stats": True,
            },
        ]

    def filter_spec_payload(self) -> dict:
        return {
            "filter_type": "linear_gaussian_kalman_filter_with_known_policy_instrument",
            "state_names": list(STATE_NAMES),
            "noisy_observation_names": list(NOISY_OBSERVATION_NAMES),
            "base_measurement_noise_std": self.base_measurement_noise(),
            "confidence_band_scale": self.confidence_scale,
            "note": (
                "Policy rate is treated as an exact component of the policymaker information set. "
                "Noisy measurements are applied only to macro and distributional releases."
            ),
        }

    def policy_spec_payload(self, phi_pi: float, phi_y: float, rho_i: float) -> dict:
        return {
            "policy_name": "certainty_equivalent_taylor_rule_on_filtered_hank_state",
            "rule_formula": (
                "i_t = rho_i i_(t-1) + (1-rho_i)(rstar_hat_t + phi_pi*pi_hat_t + phi_y*y_gap_hat_t) + eps_t^i"
            ),
            "phi_pi": float(phi_pi),
            "phi_y": float(phi_y),
            "rho_i": float(rho_i),
            "loss_function": "L_t = pi_t^2 + lambda_y * y_gap_t^2 + lambda_i * (i_t - i_(t-1))^2",
            "lambda_y": float(self.lambda_y),
            "lambda_i": float(self.lambda_i),
        }

    def scenario_table_payload(self) -> list[dict]:
        base_noise = self.base_measurement_noise()
        rows = []
        for scenario in self.scenario_specs():
            rows.append({
                "scenario": scenario["label"],
                "observed_variables": ", ".join(
                    OBSERVATION_LABELS[name]
                    for name in scenario["noisy_observations"] + scenario["known_exact"]
                ),
                "noise_scale": scenario["noise_scale"],
                "uses_distribution_stats": scenario["includes_distribution_stats"],
                "base_noise": {
                    name: base_noise[name]
                    for name in scenario["noisy_observations"]
                },
            })
        return rows

    def model_spec_payload(self, hank_model_name: str) -> dict:
        return {
            "stage": "stage3_partial_information_full_hank",
            "source_model": hank_model_name,
            "description": (
                "Reduced-state local linear representation of the full two-asset HANK model, "
                "used to filter policy-relevant macro and distributional state components "
                "from noisy observables before applying a classical policy rule."
            ),
            "state_names": list(STATE_NAMES),
            "state_interpretation": {
                "rstar_gap": "Отклонение естественной ставки / агрегированного спросового фактора",
                "productivity_gap": "Скрытый фактор производительности",
                "fiscal_gap": "Скрытый фискальный фактор",
                "inflation_gap": "Отклонение инфляции от стационара",
                "output_gap": "Отклонение выпуска от стационара",
                "low_liquidity_gap": "Отклонение доли низколиквидных домохозяйств",
                "mean_mpc_gap": "Отклонение средней MPC",
            },
            "observation_names": list(NOISY_OBSERVATION_NAMES) + ["i"],
            "timing_note": (
                "Noisy macro releases enter the Kalman update contemporaneously; the policy rate "
                "is treated as an exactly known instrument chosen by the regulator."
            ),
        }

    def to_dict(self) -> dict:
        return asdict(self)


def default_partial_info_config() -> HANKPartialInfoConfig:
    return HANKPartialInfoConfig()
