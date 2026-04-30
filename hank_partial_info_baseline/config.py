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

INFORMATION_REGIME_LABELS = {
    "compact_macro_observations": "Сжатый макронабор",
    "macro_observations": "Базовые макроэкономические наблюдения",
    "thin_information": "Ограниченный информационный набор",
    "high_noise_macro_observations": "Базовые макронаблюдения с повышенным шумом",
    "distribution_signals": "Макронаблюдения и шумные распределительные сигналы",
    "full_information_upper_bound": "Полная информация",
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
                "label": "Сжатый макронабор: инфляция, выпуск и ставка",
                "description": (
                    "Сжатый макроэкономический режим: регулятор наблюдает только шумные "
                    "сигналы по инфляции и разрыву выпуска; ставка известна точно как "
                    "собственный инструмент."
                ),
                "noisy_observations": ("pi", "output_gap"),
                "known_exact": ("i",),
                "noise_scale": 1.0,
                "includes_distribution_stats": False,
                "information_regime": "compact_macro_observations",
                "information_regime_label": INFORMATION_REGIME_LABELS["compact_macro_observations"],
                "distribution_signal_lag": None,
                "uses_true_distribution_state": False,
            },
            {
                "name": "full_macro",
                "label": "Базовые макроэкономические наблюдения",
                "description": (
                    "Базовый макроэкономический режим: регулятор получает шумные "
                    "текущие сигналы по инфляции, разрыву выпуска, потреблению, "
                    "реальной заработной плате и занятости. Распределительные "
                    "характеристики напрямую не наблюдаются."
                ),
                "noisy_observations": ("pi", "output_gap", "C", "w", "N"),
                "known_exact": ("i",),
                "noise_scale": 1.0,
                "includes_distribution_stats": False,
                "information_regime": "macro_observations",
                "information_regime_label": INFORMATION_REGIME_LABELS["macro_observations"],
                "distribution_signal_lag": None,
                "uses_true_distribution_state": False,
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
                "information_regime": "thin_information",
                "information_regime_label": INFORMATION_REGIME_LABELS["thin_information"],
                "distribution_signal_lag": None,
                "uses_true_distribution_state": False,
            },
            {
                "name": "high_noise",
                "label": "Базовые макронаблюдения с повышенным шумом",
                "description": (
                    "Те же базовые макроэкономические наблюдения, но с более высоким "
                    "шумом измерения."
                ),
                "noisy_observations": ("pi", "output_gap", "C", "w", "N"),
                "known_exact": ("i",),
                "noise_scale": 2.0,
                "includes_distribution_stats": False,
                "information_regime": "high_noise_macro_observations",
                "information_regime_label": INFORMATION_REGIME_LABELS["high_noise_macro_observations"],
                "distribution_signal_lag": None,
                "uses_true_distribution_state": False,
            },
            {
                "name": "distribution_augmented",
                "label": "Макронаблюдения и шумные распределительные сигналы",
                "description": (
                    "К базовым макроэкономическим наблюдениям добавлены шумные "
                    "сигналы по доле низколиквидных домохозяйств и средней предельной "
                    "склонности к потреблению. Эти сигналы используются как наблюдаемые "
                    "статистики, а не как истинные скрытые состояния."
                ),
                "noisy_observations": (
                    "pi",
                    "output_gap",
                    "C",
                    "w",
                    "N",
                    "share_low_liquidity",
                    "mean_mpc",
                ),
                "known_exact": ("i",),
                "noise_scale": 1.0,
                "includes_distribution_stats": True,
                "information_regime": "distribution_signals",
                "information_regime_label": INFORMATION_REGIME_LABELS["distribution_signals"],
                "distribution_signal_lag": 0,
                "uses_true_distribution_state": False,
            },
        ]

    def article_information_regimes_payload(self) -> list[dict]:
        scenario_map = {
            spec["name"]: spec
            for spec in self.scenario_specs()
        }
        macro_spec = scenario_map["full_macro"]
        distribution_spec = scenario_map["distribution_augmented"]
        return [
            {
                "name": "macro_observations",
                "label": INFORMATION_REGIME_LABELS["macro_observations"],
                "role": "базовый режим",
                "scenario_name": macro_spec["name"],
                "noisy_observations": list(macro_spec["noisy_observations"]),
                "known_exact": list(macro_spec["known_exact"]),
                "distribution_signals": "нет",
                "note": (
                    "Регулятор использует только шумные макроэкономические сигналы. "
                    "Распределительные характеристики напрямую не наблюдаются."
                ),
            },
            {
                "name": "distribution_signals",
                "label": INFORMATION_REGIME_LABELS["distribution_signals"],
                "role": "режим с распределительными сигналами",
                "scenario_name": distribution_spec["name"],
                "noisy_observations": list(distribution_spec["noisy_observations"]),
                "known_exact": list(distribution_spec["known_exact"]),
                "distribution_signals": "шумные сигналы по ликвидности и средней склонности к потреблению",
                "note": (
                    "Распределительные показатели входят только как шумные наблюдаемые "
                    "сигналы и не подменяют истинное скрытое состояние."
                ),
            },
            {
                "name": "full_information_upper_bound",
                "label": INFORMATION_REGIME_LABELS["full_information_upper_bound"],
                "role": "верхняя граница",
                "scenario_name": "full_information",
                "true_state_names": list(STATE_NAMES),
                "distribution_signals": "истинное структурное и распределительное состояние",
                "note": (
                    "Используется только как верхняя граница: правило строится по истинному "
                    "состоянию, но потери сравниваются с теми же реализованными траекториями."
                ),
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
                "Ставка известна регулятору точно как собственный инструмент. "
                "Шум накладывается только на наблюдаемые макроэкономические и "
                "распределительные статистики."
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
                "information_regime": scenario["information_regime_label"],
                "observed_variables": ", ".join(
                    OBSERVATION_LABELS[name]
                    for name in scenario["noisy_observations"] + scenario["known_exact"]
                ),
                "noise_scale": scenario["noise_scale"],
                "uses_distribution_stats": scenario["includes_distribution_stats"],
                "distribution_signal_lag": scenario["distribution_signal_lag"],
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
                "Низкоразмерное локально-линейное представление полной двухактивной "
                "HANK-модели. Используется для восстановления скрытых компонент состояния "
                "по шумным наблюдаемым сигналам перед применением правила политики."
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
                "Наблюдаемые макроэкономические и распределительные сигналы входят в "
                "обновление фильтра с шумом измерения; ставка считается точно известным "
                "инструментом, который выбирает регулятор."
            ),
            "loss_evaluation_note": (
                "Правило политики использует только наблюдаемые сигналы или оценённое "
                "состояние, а функция потерь считается по истинным реализованным значениям "
                "инфляции и разрыва выпуска."
            ),
        }

    def to_dict(self) -> dict:
        return asdict(self)


def default_partial_info_config() -> HANKPartialInfoConfig:
    return HANKPartialInfoConfig()
