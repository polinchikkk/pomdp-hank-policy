from __future__ import annotations


INFORMATION_STATE_DESIGN_SCENARIO_NAMES = (
    "full_macro_moderate_gap",
    "full_macro_strong_gap",
    "distribution_augmented_moderate_gap",
    "distribution_augmented_strong_gap",
    "thin_information_moderate_gap",
    "thin_information_strong_gap",
)

INFORMATION_STATE_DESIGN_SCENARIO_LABELS = {
    "full_macro_moderate_gap": "Базовые макроэкономические наблюдения × умеренная различимость режимов",
    "full_macro_strong_gap": "Базовые макроэкономические наблюдения × высокая различимость режимов",
    "distribution_augmented_moderate_gap": "Макронаблюдения и шумные распределительные сигналы × умеренная различимость режимов",
    "distribution_augmented_strong_gap": "Макронаблюдения и шумные распределительные сигналы × высокая различимость режимов",
    "thin_information_moderate_gap": "Бедный набор наблюдений × умеренная различимость режимов",
    "thin_information_strong_gap": "Бедный набор наблюдений × высокая различимость режимов",
}


def information_state_design_scenario_names() -> tuple[str, ...]:
    return INFORMATION_STATE_DESIGN_SCENARIO_NAMES


def information_state_design_scenario_label(scenario_name: str) -> str:
    return INFORMATION_STATE_DESIGN_SCENARIO_LABELS[scenario_name]
