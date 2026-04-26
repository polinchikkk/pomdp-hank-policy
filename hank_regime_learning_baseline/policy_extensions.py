from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_learning_policy_baseline.ppo import train_ppo_policy
from hank_full_baseline.steady_state import solve_steady_state
from hank_full_baseline.transition import solve_transition
from hank_learning_policy_baseline.policies import BasePolicy, ClassicalFilteredRulePolicy, FullInformationRulePolicy
from hank_partial_info_baseline.state_space import fit_reduced_state_space
from regime_switching_baseline.regime_model import build_regime_switching_model

from .config import RegimeLearningConfig, RegimeLearningVariant
from .core_matrix import SCENARIO_LABELS
from .environment import RegimeSwitchingPolicyEnvironment, build_scenario_spec
from .evaluation import evaluate_policy_trace, simulate_policy_episode, summarize_training_history
from .pipeline import _evaluate_checkpoint_selection
from .tuning import default_universal_candidate_lookup, extreme_sticky_regime_config


SCENARIO_NAMES = (
    "macro_core_moderate_gap",
    "macro_core_strong_gap",
    "thin_information_moderate_gap",
    "thin_information_strong_gap",
)


POLICY_CLASS_SPECS = {
    "classical_filtered_rule": {
        "policy_label_short_ru": "Фиксированное правило Тейлора",
        "policy_label_ru": "Классическое правило по оценённому состоянию",
        "input_set": "filtered_taylor_state",
        "input_set_label_ru": "Оценённый тейлоровский набор",
        "functional_class": "fixed_linear",
        "functional_class_label_ru": "Фиксированная линейная форма",
        "parameter_selection": "calibrated",
        "parameter_selection_label_ru": "Калиброванные коэффициенты",
        "uses_distributional_variables": False,
        "uses_observation_history": False,
        "uses_rl_training": False,
    },
    "optimized_linear_estimated_state": {
        "policy_label_short_ru": "Оптимизированное линейное правило",
        "policy_label_ru": "Оптимизированное линейное правило по оценённому состоянию",
        "input_set": "filtered_taylor_state",
        "input_set_label_ru": "Оценённый тейлоровский набор",
        "functional_class": "tuned_linear",
        "functional_class_label_ru": "Линейная форма, выбранная по валидации",
        "parameter_selection": "validation_grid_search",
        "parameter_selection_label_ru": "Отбор по валидационным траекториям",
        "uses_distributional_variables": False,
        "uses_observation_history": False,
        "uses_rl_training": False,
    },
    "optimized_linear_extended_state": {
        "policy_label_short_ru": "Оптимизированное расширенное линейное",
        "policy_label_ru": "Оптимизированное линейное правило на расширенном оценённом состоянии",
        "input_set": "filtered_extended_state",
        "input_set_label_ru": "Расширенное оценённое состояние",
        "functional_class": "tuned_linear",
        "functional_class_label_ru": "Линейная форма, выбранная по валидации",
        "parameter_selection": "validation_coordinate_search",
        "parameter_selection_label_ru": "Координатный отбор по валидационным траекториям",
        "uses_distributional_variables": False,
        "uses_observation_history": False,
        "uses_rl_training": False,
    },
    "history_observables_rule": {
        "policy_label_short_ru": "Историческое правило по наблюдениям",
        "policy_label_ru": "Историческое правило по наблюдаемым переменным",
        "input_set": "observation_history",
        "input_set_label_ru": "Наблюдаемые переменные и их история",
        "functional_class": "history_linear",
        "functional_class_label_ru": "Линейное правило с внутренним сглаженным состоянием",
        "parameter_selection": "validation_grid_search",
        "parameter_selection_label_ru": "Отбор по валидационным траекториям",
        "uses_distributional_variables": False,
        "uses_observation_history": True,
        "uses_rl_training": False,
    },
    "calibrated_true_state_rule": {
        "policy_label_short_ru": "Калиброванное по истинному состоянию",
        "policy_label_ru": "Калиброванное правило по истинному состоянию",
        "input_set": "true_taylor_state",
        "input_set_label_ru": "Истинный тейлоровский набор",
        "functional_class": "fixed_linear",
        "functional_class_label_ru": "Фиксированная линейная форма",
        "parameter_selection": "calibrated",
        "parameter_selection_label_ru": "Калиброванные коэффициенты",
        "uses_distributional_variables": False,
        "uses_observation_history": False,
        "uses_rl_training": False,
    },
    "optimized_linear_true_state": {
        "policy_label_short_ru": "Оптимизированное по истинному состоянию",
        "policy_label_ru": "Оптимизированный линейный ориентир полной информации",
        "input_set": "true_taylor_state",
        "input_set_label_ru": "Истинный тейлоровский набор",
        "functional_class": "tuned_linear",
        "functional_class_label_ru": "Линейная форма, выбранная по валидации",
        "parameter_selection": "validation_grid_search",
        "parameter_selection_label_ru": "Отбор по валидационным траекториям",
        "uses_distributional_variables": False,
        "uses_observation_history": False,
        "uses_rl_training": False,
    },
    "ppo_filtered_taylor_state": {
        "policy_label_short_ru": "PPO на тейлоровском наборе",
        "policy_label_ru": "PPO-правило на оценённом тейлоровском наборе",
        "input_set": "filtered_taylor_state",
        "input_set_label_ru": "Оценённый тейлоровский набор",
        "functional_class": "ppo_residual",
        "functional_class_label_ru": "Нелинейное правило PPO с добавочной поправкой",
        "parameter_selection": "ppo_checkpoint_selection",
        "parameter_selection_label_ru": "Выбор сохранённого шага по валидационным траекториям",
        "uses_distributional_variables": False,
        "uses_observation_history": False,
        "uses_rl_training": True,
    },
    "ppo_filtered_extended_state": {
        "policy_label_short_ru": "PPO на расширенном состоянии",
        "policy_label_ru": "PPO-правило на расширенном оценённом состоянии",
        "input_set": "filtered_extended_state",
        "input_set_label_ru": "Расширенное оценённое состояние",
        "functional_class": "ppo_residual",
        "functional_class_label_ru": "Нелинейное правило PPO с добавочной поправкой",
        "parameter_selection": "ppo_checkpoint_selection",
        "parameter_selection_label_ru": "Выбор сохранённого шага по валидационным траекториям",
        "uses_distributional_variables": False,
        "uses_observation_history": False,
        "uses_rl_training": True,
    },
}


@dataclass(frozen=True)
class LinearRuleParameters:
    phi_pi: float
    phi_y: float
    rho_i: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class HistoryRuleParameters:
    phi_pi: float
    phi_y: float
    rho_i: float
    alpha: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ExtendedLinearRuleParameters:
    coeff_rstar: float
    coeff_productivity: float
    coeff_fiscal: float
    coeff_pi: float
    coeff_output: float
    coeff_stress: float
    rho_i: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class OptimizedEstimatedStateRulePolicy(BasePolicy):
    """Taylor-type linear rule selected on validation paths and applied to the filtered state."""

    def __init__(self, params: LinearRuleParameters) -> None:
        self.params = params

    def rate(self, observation: np.ndarray, info: dict) -> float:
        state_names = tuple(info["state_names"])
        state = np.asarray(info["filtered_state"], dtype=float)
        previous_rate = float(info["current_rate"])
        lower, upper = info["rate_bounds"]
        idx_rstar = state_names.index("rstar_gap")
        idx_pi = state_names.index("inflation_gap")
        idx_output = state_names.index("output_gap")
        target = (
            state[idx_rstar]
            + self.params.phi_pi * state[idx_pi]
            + self.params.phi_y * state[idx_output]
        )
        rate = self.params.rho_i * previous_rate + (1.0 - self.params.rho_i) * target
        return float(np.clip(rate, lower, upper))


class OptimizedTrueStateRulePolicy(BasePolicy):
    """Taylor-type linear rule selected on validation paths and applied to the true state."""

    def __init__(self, params: LinearRuleParameters) -> None:
        self.params = params

    def rate(self, observation: np.ndarray, info: dict) -> float:
        state_names = tuple(info["state_names"])
        state = np.asarray(info["true_state"], dtype=float)
        previous_rate = float(info["current_rate"])
        lower, upper = info["rate_bounds"]
        idx_rstar = state_names.index("rstar_gap")
        idx_pi = state_names.index("inflation_gap")
        idx_output = state_names.index("output_gap")
        target = (
            state[idx_rstar]
            + self.params.phi_pi * state[idx_pi]
            + self.params.phi_y * state[idx_output]
        )
        rate = self.params.rho_i * previous_rate + (1.0 - self.params.rho_i) * target
        return float(np.clip(rate, lower, upper))


class OptimizedExtendedStateRulePolicy(BasePolicy):
    """Linear rule on the expanded filtered state with regime probability."""

    def __init__(self, params: ExtendedLinearRuleParameters) -> None:
        self.params = params

    def rate(self, observation: np.ndarray, info: dict) -> float:
        state_names = tuple(info["state_names"])
        state = np.asarray(info["filtered_state"], dtype=float)
        previous_rate = float(info["current_rate"])
        stress_probability = float(info["stress_probability"])
        lower, upper = info["rate_bounds"]
        idx_rstar = state_names.index("rstar_gap")
        idx_productivity = state_names.index("productivity_gap")
        idx_fiscal = state_names.index("fiscal_gap")
        idx_pi = state_names.index("inflation_gap")
        idx_output = state_names.index("output_gap")
        target = (
            self.params.coeff_rstar * state[idx_rstar]
            + self.params.coeff_productivity * state[idx_productivity]
            + self.params.coeff_fiscal * state[idx_fiscal]
            + self.params.coeff_pi * state[idx_pi]
            + self.params.coeff_output * state[idx_output]
            + self.params.coeff_stress * stress_probability
        )
        rate = self.params.rho_i * previous_rate + (1.0 - self.params.rho_i) * target
        return float(np.clip(rate, lower, upper))


class HistoryObservationRulePolicy(BasePolicy):
    """Stateful linear rule that uses an exponentially smoothed history of observed releases."""

    def __init__(self, params: HistoryRuleParameters) -> None:
        self.params = params
        self._smoothed_pi = 0.0
        self._smoothed_output = 0.0
        self._initialized = False

    def reset(self) -> None:
        self._smoothed_pi = 0.0
        self._smoothed_output = 0.0
        self._initialized = False

    def rate(self, observation: np.ndarray, info: dict) -> float:
        observation_names = tuple(info["noisy_observation_names"])
        observations = np.asarray(info["current_observations"], dtype=float)
        previous_rate = float(info["current_rate"])
        lower, upper = info["rate_bounds"]
        obs_map = {name: float(value) for name, value in zip(observation_names, observations)}
        current_pi = obs_map.get("pi", obs_map.get("inflation_gap", 0.0))
        current_output = obs_map.get("output_gap", 0.0)
        if not self._initialized:
            self._smoothed_pi = current_pi
            self._smoothed_output = current_output
            self._initialized = True
        else:
            alpha = self.params.alpha
            self._smoothed_pi = alpha * self._smoothed_pi + (1.0 - alpha) * current_pi
            self._smoothed_output = alpha * self._smoothed_output + (1.0 - alpha) * current_output
        target = self.params.phi_pi * self._smoothed_pi + self.params.phi_y * self._smoothed_output
        rate = self.params.rho_i * previous_rate + (1.0 - self.params.rho_i) * target
        return float(np.clip(rate, lower, upper))


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _format_seed_span(seeds: tuple[int, ...]) -> str:
    if not seeds:
        return ""
    if len(seeds) == 1:
        return f"`{seeds[0]}`"
    return f"`{seeds[0]}`--`{seeds[-1]}`"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _write_extension_latex_tables(root: Path, comparison_summary: pd.DataFrame, selected_specs: pd.DataFrame) -> None:
    focus = comparison_summary[
        comparison_summary["comparison_name"].isin(
            [
                "linear_minus_classical",
                "extended_minus_linear",
                "ppo_taylor_minus_linear",
                "ppo_extended_minus_extended_linear",
                "linear_minus_history",
                "linear_minus_optimized_true_state",
            ]
        )
    ].copy()
    lines = [
        "\\begin{tabular}{p{0.32\\linewidth}p{0.24\\linewidth}rrrr}",
        "\\toprule",
        "Сценарий & Сравнение & $\\Delta J$ & 95\\% ДИ & Доля побед & $N$ \\\\",
        "\\midrule",
    ]
    for _, row in focus.iterrows():
        ci = f"[{row['ci_lower']:.4f}; {row['ci_upper']:.4f}]"
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["scenario_label"])),
                    _latex_escape(str(row["comparison_label"])),
                    f"{float(row['mean_delta_cumulative_loss']):.4f}",
                    _latex_escape(ci),
                    f"{float(row['win_rate']):.2f}",
                    f"{int(row['num_test_trajectories'])}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_stage6_policy_extensions_comparison.tex").write_text("\n".join(lines), encoding="utf-8")

    specs = selected_specs.copy()

    def _parameter_summary(row: pd.Series) -> str:
        policy_name = str(row["policy_name"])
        if policy_name in {"optimized_linear_estimated_state", "optimized_linear_true_state"}:
            return (
                f"$\\phi_\\pi={float(row['phi_pi']):.2f}$, "
                f"$\\phi_y={float(row['phi_y']):.3f}$, "
                f"$\\rho_i={float(row['rho_i']):.2f}$"
            )
        if policy_name == "optimized_linear_extended_state":
            return (
                f"$\\rho_i={float(row['rho_i']):.2f}$, "
                f"$c_r={float(row['coeff_rstar']):.2f}$, "
                f"$c_z={float(row['coeff_productivity']):.2f}$, "
                f"$c_f={float(row['coeff_fiscal']):.2f}$, "
                f"$c_\\pi={float(row['coeff_pi']):.2f}$, "
                f"$c_y={float(row['coeff_output']):.2f}$, "
                f"$c_p={float(row['coeff_stress']):.2f}$"
            )
        if policy_name == "history_observables_rule":
            return (
                f"$\\phi_\\pi={float(row['phi_pi']):.2f}$, "
                f"$\\phi_y={float(row['phi_y']):.3f}$, "
                f"$\\rho_i={float(row['rho_i']):.2f}$, "
                f"$\\alpha={float(row['alpha']):.2f}$"
            )
        if policy_name in {"ppo_filtered_taylor_state", "ppo_filtered_extended_state"}:
            return (
                f"запуск={int(row['training_seed'])}, "
                f"итерация={int(row['checkpoint_iteration'])}, "
                f"$V_{{проверка}}={float(row['validation_return']):.3f}$"
            )
        return ""

    lines = [
        "\\begin{tabular}{p{0.34\\linewidth}p{0.18\\linewidth}p{0.34\\linewidth}}",
        "\\toprule",
        "Сценарий & Правило & Выбранные параметры \\\\",
        "\\midrule",
    ]
    policy_labels = {
        "optimized_linear_estimated_state": "Линейное по оценке",
        "optimized_linear_extended_state": "Расширенное линейное",
        "ppo_filtered_taylor_state": "PPO на тейлоровском наборе",
        "ppo_filtered_extended_state": "PPO на расширенном состоянии",
        "history_observables_rule": "Историческое по наблюдениям",
        "optimized_linear_true_state": "Линейное по истинному состоянию",
    }
    for _, row in specs.iterrows():
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["scenario_label"])),
                    _latex_escape(policy_labels.get(str(row["policy_name"]), str(row["policy_name"]))),
                    _parameter_summary(row),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_stage6_policy_extensions_selected_rules.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_policy_rule_ablation_tables(
    root: Path,
    levels: pd.DataFrame,
    same_input: pd.DataFrame,
    input_ablation: pd.DataFrame,
    history_vs_filtered: pd.DataFrame,
    ppo_same_input: pd.DataFrame,
) -> None:
    policy_order = [
        "classical_filtered_rule",
        "optimized_linear_estimated_state",
        "ppo_filtered_taylor_state",
        "optimized_linear_extended_state",
        "ppo_filtered_extended_state",
        "history_observables_rule",
        "calibrated_true_state_rule",
        "optimized_linear_true_state",
    ]
    pivot = levels.pivot_table(
        index=["scenario_name", "scenario_label"],
        columns="policy_name",
        values="mean_cumulative_policy_loss",
        aggfunc="first",
    )
    available = [policy_name for policy_name in policy_order if policy_name in pivot.columns]
    pivot = pivot[available]
    labels = {
        "classical_filtered_rule": "Фиксированное правило Тейлора",
        "optimized_linear_estimated_state": "Оптимизированное линейное",
        "ppo_filtered_taylor_state": "PPO на тейлоровском наборе",
        "optimized_linear_extended_state": "Расширенное линейное",
        "ppo_filtered_extended_state": "PPO на расширенном состоянии",
        "history_observables_rule": "История наблюдений",
        "calibrated_true_state_rule": "Калиброванное истинное состояние",
        "optimized_linear_true_state": "Оптимизированное истинное состояние",
    }
    lines = [
        "\\begin{tabular}{p{0.30\\linewidth}" + "r" * len(available) + "}",
        "\\toprule",
        "Сценарий & " + " & ".join(_latex_escape(labels[policy_name]) for policy_name in available) + " \\\\",
        "\\midrule",
    ]
    for (_scenario_name, scenario_label), row in pivot.iterrows():
        lines.append(
            " & ".join(
                [_latex_escape(str(scenario_label))]
                + [f"{float(row[policy_name]):.4e}" for policy_name in available]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_policy_rule_ablation_levels.tex").write_text("\n".join(lines), encoding="utf-8")

    def _write_comparison_table(frame: pd.DataFrame, path: Path) -> None:
        if frame.empty:
            return
        lines = [
            "\\begin{tabular}{p{0.28\\linewidth}p{0.22\\linewidth}p{0.22\\linewidth}rr}",
            "\\toprule",
            "Сценарий & Что меняется & Что фиксировано & $\\Delta J$ & Доля побед \\\\",
            "\\midrule",
        ]
        for row in frame.to_dict(orient="records"):
            lines.append(
                " & ".join(
                    [
                        _latex_escape(str(row["scenario_label"])),
                        _latex_escape(str(row["what_changes"])),
                        _latex_escape(str(row["what_fixed"])),
                        f"{float(row['mean_delta_cumulative_loss']):.4e}",
                        f"{float(row['win_rate']):.2f}",
                    ]
                )
                + " \\\\"
            )
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        path.write_text("\n".join(lines), encoding="utf-8")

    _write_comparison_table(same_input, root / "table_same_input_comparisons.tex")
    _write_comparison_table(input_ablation, root / "table_input_set_ablation.tex")
    _write_comparison_table(history_vs_filtered, root / "table_history_vs_filtered_comparisons.tex")
    _write_comparison_table(ppo_same_input, root / "table_ppo_same_input_comparisons.tex")


def _write_ppo_seed_robustness_table(root: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    lines = [
        "\\begin{tabular}{p{0.20\\linewidth}p{0.15\\linewidth}rrrrrr}",
        "\\toprule",
        "Сценарий & Правило & Запуск & Итерация & Потеря на тесте & $\\Delta J$ к линейному правилу & Доля побед & Неустойчивые траектории \\\\",
        "\\midrule",
    ]
    short_labels = {
        "ppo_filtered_taylor_state": "PPO, тейлоровский вход",
        "ppo_filtered_extended_state": "PPO, расширенный вход",
    }
    for row in summary.to_dict(orient="records"):
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["scenario_label"])),
                    _latex_escape(short_labels.get(str(row["policy_name"]), str(row["policy_name"]))),
                    f"{int(row['training_seed'])}",
                    f"{int(row['checkpoint_iteration'])}",
                    f"{float(row['mean_test_cumulative_loss']):.4e}",
                    f"{float(row['mean_delta_vs_linear']):.4e}",
                    f"{float(row['win_rate_vs_linear']):.2f}",
                    f"{int(row['test_unstable_episodes'])}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    (root / "table_ppo_seed_robustness.tex").write_text("\n".join(lines), encoding="utf-8")


def _bootstrap_ci(values: np.ndarray, *, seed: int = 2026, draws: int = 4000) -> tuple[float, float]:
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _policy_class_spec_frame() -> pd.DataFrame:
    rows = []
    for policy_name, payload in POLICY_CLASS_SPECS.items():
        rows.append({"policy_name": policy_name, **payload})
    return pd.DataFrame(rows)


def _build_variant(*, scenario_name: str, input_mode: str, include_distributional_state: bool) -> RegimeLearningVariant:
    suffix_map = {
        "belief_state": "estimated_state",
        "filtered_taylor_state": "filtered_taylor_state",
        "filtered_extended_state": "filtered_extended_state",
        "raw_observations": "history_observables",
    }
    description_map = {
        "belief_state": "Selected linear rule on the filtered state.",
        "filtered_taylor_state": "PPO on the filtered Taylor-state input.",
        "filtered_extended_state": "PPO on the filtered extended-state input.",
        "raw_observations": "History-based rule on observed macro releases.",
    }
    suffix = suffix_map[input_mode]
    return RegimeLearningVariant(
        name=f"{scenario_name}_{suffix}",
        scenario_name=scenario_name,
        scenario_label=SCENARIO_LABELS[scenario_name],
        input_mode=input_mode,
        include_distributional_state=include_distributional_state,
        description=description_map[input_mode],
    )


@lru_cache(maxsize=1)
def _base_reduced_objects():
    hank_config = default_calibration()
    regime_config = extreme_sticky_regime_config()
    bundle = solve_steady_state(hank_config)
    reduced_model = fit_reduced_state_space(bundle, hank_config, regime_config.partial_config)
    return hank_config, regime_config, reduced_model


def _build_objects(
    scenario_name: str,
    *,
    input_mode: str,
    include_distributional_state: bool,
    validation_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
):
    candidate = default_universal_candidate_lookup()["larger_network"]
    hank_config, regime_config, reduced_model = _base_reduced_objects()
    config = RegimeLearningConfig(
        horizon=60,
        gamma=0.99,
        lambda_y=0.5,
        lambda_i=0.05,
        action_bound=candidate.action_bound,
        classical_policy_mode="switching",
        training_seeds=(11,),
        selection_seeds=validation_seeds,
        evaluation_seeds=test_seeds,
        regime_config=regime_config,
        ppo=candidate.ppo,
    )
    variant = _build_variant(
        scenario_name=scenario_name,
        input_mode=input_mode,
        include_distributional_state=include_distributional_state,
    )
    scenario_spec = build_scenario_spec(config, variant)
    regime_model = build_regime_switching_model(reduced_model, regime_config, scenario_spec.gap_scale)

    def env_factory():
        return RegimeSwitchingPolicyEnvironment(
            model=regime_model,
            regime_config=regime_config,
            scenario_spec=scenario_spec,
            phi_pi=hank_config.phi_pi,
            phi_y=hank_config.phi_y,
            rho_i=hank_config.rho_i,
        )

    return hank_config, scenario_spec, env_factory


def _mean_cumulative_loss(*, env_factory, scenario_spec, policy: BasePolicy, seeds: Iterable[int]) -> tuple[float, float, int]:
    losses = []
    volatilities = []
    unstable = 0
    for seed in seeds:
        trace = simulate_policy_episode(
            env_factory=env_factory,
            policy=policy,
            scenario_spec=scenario_spec,
            evaluation_seed=int(seed),
            policy_name="candidate_policy",
            policy_label="Candidate policy",
            training_seed=None,
        )
        losses.append(float(trace["loss"].sum()))
        volatilities.append(float(np.std(trace["policy_rate"].to_numpy(dtype=float))))
        values = trace[["true_inflation_gap", "true_output_gap", "policy_rate"]].to_numpy(dtype=float)
        unstable += int((not np.isfinite(values).all()) or np.any(np.abs(values) > np.array([0.06, 0.12, 0.06])[None, :]))
    return float(np.mean(losses)), float(np.mean(volatilities)), int(unstable)


def _select_linear_rule(*, scenario_name: str, validation_seeds: tuple[int, ...]) -> tuple[LinearRuleParameters, pd.DataFrame]:
    _hank_config, scenario_spec, env_factory = _build_objects(
        scenario_name,
        input_mode="filtered_taylor_state",
        include_distributional_state=False,
        validation_seeds=validation_seeds,
        test_seeds=validation_seeds,
    )
    rows = []
    for phi_pi in (1.0, 1.5, 2.0, 2.5, 3.0):
        for phi_y in (0.0, 0.125, 0.25, 0.5):
            for rho_i in (0.3, 0.5, 0.7, 0.85):
                params = LinearRuleParameters(phi_pi=phi_pi, phi_y=phi_y, rho_i=rho_i)
                loss, volatility, unstable = _mean_cumulative_loss(
                    env_factory=env_factory,
                    scenario_spec=scenario_spec,
                    policy=OptimizedEstimatedStateRulePolicy(params),
                    seeds=validation_seeds,
                )
                rows.append({
                    "scenario_name": scenario_name,
                    **params.to_dict(),
                    "validation_cumulative_loss": loss,
                    "validation_policy_volatility": volatility,
                    "validation_unstable_episodes": unstable,
                })
    grid = pd.DataFrame(rows).sort_values(
        ["validation_unstable_episodes", "validation_cumulative_loss", "validation_policy_volatility"]
    ).reset_index(drop=True)
    best = grid.iloc[0]
    return (
        LinearRuleParameters(phi_pi=float(best["phi_pi"]), phi_y=float(best["phi_y"]), rho_i=float(best["rho_i"])),
        grid,
    )


def _select_true_state_linear_rule(
    *,
    scenario_name: str,
    validation_seeds: tuple[int, ...],
) -> tuple[LinearRuleParameters, pd.DataFrame]:
    _hank_config, scenario_spec, env_factory = _build_objects(
        scenario_name,
        input_mode="filtered_taylor_state",
        include_distributional_state=False,
        validation_seeds=validation_seeds,
        test_seeds=validation_seeds,
    )
    rows = []
    for phi_pi in (1.0, 1.5, 2.0, 2.5, 3.0):
        for phi_y in (0.0, 0.125, 0.25, 0.5):
            for rho_i in (0.3, 0.5, 0.7, 0.85):
                params = LinearRuleParameters(phi_pi=phi_pi, phi_y=phi_y, rho_i=rho_i)
                loss, volatility, unstable = _mean_cumulative_loss(
                    env_factory=env_factory,
                    scenario_spec=scenario_spec,
                    policy=OptimizedTrueStateRulePolicy(params),
                    seeds=validation_seeds,
                )
                rows.append({
                    "scenario_name": scenario_name,
                    **params.to_dict(),
                    "validation_cumulative_loss": loss,
                    "validation_policy_volatility": volatility,
                    "validation_unstable_episodes": unstable,
                })
    grid = pd.DataFrame(rows).sort_values(
        ["validation_unstable_episodes", "validation_cumulative_loss", "validation_policy_volatility"]
    ).reset_index(drop=True)
    best = grid.iloc[0]
    return (
        LinearRuleParameters(phi_pi=float(best["phi_pi"]), phi_y=float(best["phi_y"]), rho_i=float(best["rho_i"])),
        grid,
    )


def _select_extended_state_linear_rule(
    *,
    scenario_name: str,
    validation_seeds: tuple[int, ...],
    anchor_params: LinearRuleParameters,
) -> tuple[ExtendedLinearRuleParameters, pd.DataFrame]:
    _hank_config, scenario_spec, env_factory = _build_objects(
        scenario_name,
        input_mode="filtered_extended_state",
        include_distributional_state=False,
        validation_seeds=validation_seeds,
        test_seeds=validation_seeds,
    )
    candidate_grid = {
        "coeff_rstar": (0.5, 1.0, 1.5),
        "coeff_productivity": (-0.5, -0.25, 0.0, 0.25, 0.5),
        "coeff_fiscal": (-0.5, -0.25, 0.0, 0.25, 0.5),
        "coeff_pi": (1.0, 1.5, 2.0, 2.5, 3.0),
        "coeff_output": (0.0, 0.125, 0.25, 0.5),
        "coeff_stress": (-0.5, -0.25, 0.0, 0.25, 0.5),
        "rho_i": (0.3, 0.5, 0.7, 0.85),
    }
    current = ExtendedLinearRuleParameters(
        coeff_rstar=1.0,
        coeff_productivity=0.0,
        coeff_fiscal=0.0,
        coeff_pi=anchor_params.phi_pi,
        coeff_output=anchor_params.phi_y,
        coeff_stress=0.0,
        rho_i=anchor_params.rho_i,
    )
    rows = []

    def evaluate(params: ExtendedLinearRuleParameters) -> tuple[float, float, int]:
        loss, volatility, unstable = _mean_cumulative_loss(
            env_factory=env_factory,
            scenario_spec=scenario_spec,
            policy=OptimizedExtendedStateRulePolicy(params),
            seeds=validation_seeds,
        )
        rows.append({
            "scenario_name": scenario_name,
            **params.to_dict(),
            "validation_cumulative_loss": loss,
            "validation_policy_volatility": volatility,
            "validation_unstable_episodes": unstable,
        })
        return loss, volatility, unstable

    best_loss, best_volatility, best_unstable = evaluate(current)
    for pass_id in range(3):
        improved = False
        for field_name, candidates in candidate_grid.items():
            best_field_params = current
            best_field_tuple = (best_unstable, best_loss, best_volatility)
            for candidate_value in candidates:
                candidate_dict = current.to_dict()
                candidate_dict[field_name] = float(candidate_value)
                params = ExtendedLinearRuleParameters(**candidate_dict)
                loss, volatility, unstable = evaluate(params)
                candidate_tuple = (unstable, loss, volatility)
                if candidate_tuple < best_field_tuple:
                    best_field_tuple = candidate_tuple
                    best_field_params = params
            if best_field_params != current:
                current = best_field_params
                best_unstable, best_loss, best_volatility = best_field_tuple
                improved = True
        if not improved:
            break
        rows[-1]["coordinate_pass_completed"] = pass_id + 1
    grid = pd.DataFrame(rows).drop_duplicates().sort_values(
        ["validation_unstable_episodes", "validation_cumulative_loss", "validation_policy_volatility"]
    ).reset_index(drop=True)
    best = grid.iloc[0]
    return (
        ExtendedLinearRuleParameters(
            coeff_rstar=float(best["coeff_rstar"]),
            coeff_productivity=float(best["coeff_productivity"]),
            coeff_fiscal=float(best["coeff_fiscal"]),
            coeff_pi=float(best["coeff_pi"]),
            coeff_output=float(best["coeff_output"]),
            coeff_stress=float(best["coeff_stress"]),
            rho_i=float(best["rho_i"]),
        ),
        grid,
    )


def _select_history_rule(*, scenario_name: str, validation_seeds: tuple[int, ...]) -> tuple[HistoryRuleParameters, pd.DataFrame]:
    _hank_config, scenario_spec, env_factory = _build_objects(
        scenario_name,
        input_mode="raw_observations",
        include_distributional_state=True,
        validation_seeds=validation_seeds,
        test_seeds=validation_seeds,
    )
    output_grid = (0.0, 0.125, 0.25) if "macro_core" in scenario_name else (0.0,)
    rows = []
    for phi_pi in (0.5, 1.0, 1.5, 2.0, 2.5):
        for phi_y in output_grid:
            for rho_i in (0.3, 0.5, 0.7, 0.85):
                for alpha in (0.2, 0.5, 0.8):
                    params = HistoryRuleParameters(phi_pi=phi_pi, phi_y=phi_y, rho_i=rho_i, alpha=alpha)
                    loss, volatility, unstable = _mean_cumulative_loss(
                        env_factory=env_factory,
                        scenario_spec=scenario_spec,
                        policy=HistoryObservationRulePolicy(params),
                        seeds=validation_seeds,
                    )
                    rows.append({
                        "scenario_name": scenario_name,
                        **params.to_dict(),
                        "validation_cumulative_loss": loss,
                        "validation_policy_volatility": volatility,
                        "validation_unstable_episodes": unstable,
                    })
    grid = pd.DataFrame(rows).sort_values(
        ["validation_unstable_episodes", "validation_cumulative_loss", "validation_policy_volatility"]
    ).reset_index(drop=True)
    best = grid.iloc[0]
    return (
        HistoryRuleParameters(
            phi_pi=float(best["phi_pi"]),
            phi_y=float(best["phi_y"]),
            rho_i=float(best["rho_i"]),
            alpha=float(best["alpha"]),
        ),
        grid,
    )


def _ppo_selection_key(entry: dict[str, float | int | str | BasePolicy]) -> tuple[float, float, int, float]:
    return (
        float(entry["selection_objective"]),
        float(entry["selection_rate_rmse"]),
        int(entry["selection_unstable_episodes"]),
        -float(entry["validation_return"]),
    )


def _select_ppo_policy(
    *,
    scenario_name: str,
    input_mode: str,
    include_distributional_state: bool,
    training_seeds: tuple[int, ...],
    validation_seeds: tuple[int, ...],
    policy_name: str,
    policy_label: str,
) -> tuple[BasePolicy, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    candidate = default_universal_candidate_lookup()["larger_network"]
    _hank_config, scenario_spec, env_factory = _build_objects(
        scenario_name,
        input_mode=input_mode,
        include_distributional_state=include_distributional_state,
        validation_seeds=validation_seeds,
        test_seeds=validation_seeds,
    )
    classical_policy = ClassicalFilteredRulePolicy(action_bound=scenario_spec.action_bound)
    classical_label = "Классическое правило по оценённому состоянию"

    training_histories = []
    best_entry = None
    best_by_seed: dict[int, dict[str, object]] = {}
    for training_seed in training_seeds:
        trained_policy, history, checkpoints = train_ppo_policy(
            env_factory=env_factory,
            ppo_config=candidate.ppo,
            action_bound=scenario_spec.action_bound,
            gamma=scenario_spec.gamma,
            training_seed=int(training_seed),
            label=f"{scenario_name}_{policy_name}",
        )
        training_histories.append(history)
        checkpoint_candidates = checkpoints or []
        checkpoint_candidates.append(
            type(
                "FinalCheckpoint",
                (),
                {
                    "iteration": int(history["iteration"].iloc[-1]) if not history.empty else -1,
                    "policy": trained_policy,
                    "validation_return": float(history["validation_return"].iloc[-1]) if not history.empty else 0.0,
                    "mean_episode_return": float(history["mean_episode_return"].iloc[-1]) if not history.empty else 0.0,
                },
            )()
        )
        for checkpoint in checkpoint_candidates:
            selection_summary = _evaluate_checkpoint_selection(
                env_factory=env_factory,
                policy=checkpoint.policy,
                classical_policy=classical_policy,
                classical_label=classical_label,
                scenario_spec=scenario_spec,
                selection_seeds=validation_seeds,
            )
            candidate_entry = {
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "policy_name": policy_name,
                "policy_label": policy_label,
                "training_seed": int(training_seed),
                "checkpoint_iteration": int(checkpoint.iteration),
                "validation_return": float(checkpoint.validation_return),
                "mean_episode_return": float(checkpoint.mean_episode_return),
                **selection_summary,
                "policy": checkpoint.policy,
            }
            if best_entry is None or _ppo_selection_key(candidate_entry) < _ppo_selection_key(best_entry):
                best_entry = candidate_entry
            seed_key = int(training_seed)
            if seed_key not in best_by_seed or _ppo_selection_key(candidate_entry) < _ppo_selection_key(best_by_seed[seed_key]):
                best_by_seed[seed_key] = candidate_entry

    if best_entry is None:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Failed to select PPO policy for {scenario_name=} and {policy_name=}.")

    training_history = pd.concat(training_histories, ignore_index=True)
    training_summary = summarize_training_history(training_history)
    selected_summary = pd.DataFrame(
        [
            {
                key: value
                for key, value in best_entry.items()
                if key != "policy"
            }
        ]
    )
    seed_selected_entries = [best_by_seed[seed] for seed in sorted(best_by_seed)]
    return best_entry["policy"], training_history, training_summary, selected_summary, seed_selected_entries


def _evaluate_ppo_seed_robustness(
    *,
    scenario_name: str,
    validation_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
    linear_params: LinearRuleParameters,
    extended_params: ExtendedLinearRuleParameters,
    true_state_params: LinearRuleParameters,
    ppo_taylor_seed_entries: list[dict[str, object]],
    ppo_extended_seed_entries: list[dict[str, object]],
) -> pd.DataFrame:
    _hank_config, taylor_spec, taylor_env_factory = _build_objects(
        scenario_name,
        input_mode="filtered_taylor_state",
        include_distributional_state=False,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )
    _hank_config, extended_spec, extended_env_factory = _build_objects(
        scenario_name,
        input_mode="filtered_extended_state",
        include_distributional_state=False,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )

    linear_policy = OptimizedEstimatedStateRulePolicy(linear_params)
    extended_policy = OptimizedExtendedStateRulePolicy(extended_params)
    true_state_policy = OptimizedTrueStateRulePolicy(true_state_params)

    rows: list[dict[str, object]] = []
    for evaluation_seed in test_seeds:
        true_state_trace = simulate_policy_episode(
            env_factory=taylor_env_factory,
            policy=true_state_policy,
            scenario_spec=taylor_spec,
            evaluation_seed=int(evaluation_seed),
            policy_name="optimized_linear_true_state",
            policy_label="Оптимизированный линейный ориентир полной информации",
            training_seed=None,
        )
        linear_trace = simulate_policy_episode(
            env_factory=taylor_env_factory,
            policy=linear_policy,
            scenario_spec=taylor_spec,
            evaluation_seed=int(evaluation_seed),
            policy_name="optimized_linear_estimated_state",
            policy_label="Оптимизированное линейное правило по оценённому состоянию",
            training_seed=None,
        )
        extended_trace = simulate_policy_episode(
            env_factory=extended_env_factory,
            policy=extended_policy,
            scenario_spec=extended_spec,
            evaluation_seed=int(evaluation_seed),
            policy_name="optimized_linear_extended_state",
            policy_label="Оптимизированное линейное правило на расширенном оценённом состоянии",
            training_seed=None,
        )

        def _collect(entry: dict[str, object], *, env_factory, scenario_spec, benchmark_trace: pd.DataFrame, benchmark_name: str, benchmark_label: str) -> None:
            trace = simulate_policy_episode(
                env_factory=env_factory,
                policy=entry["policy"],
                scenario_spec=scenario_spec,
                evaluation_seed=int(evaluation_seed),
                policy_name=str(entry["policy_name"]),
                policy_label=str(entry["policy_label"]),
                training_seed=int(entry["training_seed"]),
            )
            metrics, _path_frame = evaluate_policy_trace(
                policy_trace=trace,
                reference_trace=true_state_trace,
                scenario_spec=scenario_spec,
            )
            metrics.update(
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "checkpoint_iteration": int(entry["checkpoint_iteration"]),
                    "validation_return": float(entry["validation_return"]),
                    "mean_episode_return": float(entry["mean_episode_return"]),
                    "selection_objective": float(entry["selection_objective"]),
                    "selection_rate_rmse": float(entry["selection_rate_rmse"]),
                    "selection_unstable_episodes": int(entry["selection_unstable_episodes"]),
                    "benchmark_policy_name": benchmark_name,
                    "benchmark_policy_label": benchmark_label,
                    "benchmark_cumulative_policy_loss": float(benchmark_trace["loss"].sum()),
                }
            )
            rows.append(metrics)

        for entry in ppo_taylor_seed_entries:
            _collect(
                entry,
                env_factory=taylor_env_factory,
                scenario_spec=taylor_spec,
                benchmark_trace=linear_trace,
                benchmark_name="optimized_linear_estimated_state",
                benchmark_label="Оптимизированное линейное правило по оценённому состоянию",
            )
        for entry in ppo_extended_seed_entries:
            _collect(
                entry,
                env_factory=extended_env_factory,
                scenario_spec=extended_spec,
                benchmark_trace=extended_trace,
                benchmark_name="optimized_linear_extended_state",
                benchmark_label="Оптимизированное линейное правило на расширенном оценённом состоянии",
            )

    return pd.DataFrame(rows)


def _summarize_ppo_seed_robustness(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    if seed_metrics.empty:
        return pd.DataFrame()
    group_columns = [
        "scenario_name",
        "scenario_label",
        "policy_name",
        "policy_label",
        "training_seed",
        "checkpoint_iteration",
        "validation_return",
        "selection_objective",
        "selection_rate_rmse",
        "selection_unstable_episodes",
        "benchmark_policy_name",
        "benchmark_policy_label",
    ]
    rows = []
    for keys, frame in seed_metrics.groupby(group_columns, dropna=False):
        deltas = (
            frame["cumulative_policy_loss"].to_numpy(dtype=float)
            - frame["benchmark_cumulative_policy_loss"].to_numpy(dtype=float)
        )
        loss_values = frame["cumulative_policy_loss"].to_numpy(dtype=float)
        loss_ci_low, loss_ci_high = _bootstrap_ci(loss_values)
        delta_ci_low, delta_ci_high = _bootstrap_ci(deltas)
        key_map = dict(zip(group_columns, keys))
        rows.append(
            {
                **key_map,
                "mean_test_cumulative_loss": float(loss_values.mean()),
                "test_loss_ci_lower": loss_ci_low,
                "test_loss_ci_upper": loss_ci_high,
                "mean_delta_vs_linear": float(deltas.mean()),
                "delta_ci_lower": delta_ci_low,
                "delta_ci_upper": delta_ci_high,
                "win_rate_vs_linear": float(np.mean(deltas < 0.0)),
                "probability_of_degradation_vs_linear": float(np.mean(deltas > 0.0)),
                "test_unstable_episodes": int(frame["unstable"].sum()),
                "num_test_trajectories": int(frame.shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["scenario_name", "policy_name", "training_seed"]
    ).reset_index(drop=True)


def _evaluate_selected_rules(
    *,
    scenario_name: str,
    validation_seeds: tuple[int, ...],
    test_seeds: tuple[int, ...],
    linear_params: LinearRuleParameters,
    extended_params: ExtendedLinearRuleParameters,
    history_params: HistoryRuleParameters,
    true_state_params: LinearRuleParameters,
    ppo_taylor_policy: BasePolicy,
    ppo_extended_policy: BasePolicy,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _hank_config, taylor_spec, taylor_env_factory = _build_objects(
        scenario_name,
        input_mode="filtered_taylor_state",
        include_distributional_state=False,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )
    _hank_config, extended_spec, extended_env_factory = _build_objects(
        scenario_name,
        input_mode="filtered_extended_state",
        include_distributional_state=False,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )
    _hank_config, raw_spec, raw_env_factory = _build_objects(
        scenario_name,
        input_mode="raw_observations",
        include_distributional_state=True,
        validation_seeds=validation_seeds,
        test_seeds=test_seeds,
    )

    policy_metric_rows = []
    policy_path_frames = []
    classical_policy = ClassicalFilteredRulePolicy(action_bound=taylor_spec.action_bound)
    calibrated_true_policy = FullInformationRulePolicy(action_bound=taylor_spec.action_bound)
    linear_policy = OptimizedEstimatedStateRulePolicy(linear_params)
    extended_policy = OptimizedExtendedStateRulePolicy(extended_params)
    history_policy = HistoryObservationRulePolicy(history_params)
    true_state_policy = OptimizedTrueStateRulePolicy(true_state_params)

    for seed in test_seeds:
        true_state_trace = simulate_policy_episode(
            env_factory=taylor_env_factory,
            policy=true_state_policy,
            scenario_spec=taylor_spec,
            evaluation_seed=int(seed),
            policy_name="optimized_linear_true_state",
            policy_label="Оптимизированный линейный ориентир полной информации",
            training_seed=None,
        )
        policies = (
            (
                taylor_env_factory,
                taylor_spec,
                classical_policy,
                "classical_filtered_rule",
                "Классическое правило по оценённому состоянию",
            ),
            (
                taylor_env_factory,
                taylor_spec,
                linear_policy,
                "optimized_linear_estimated_state",
                "Оптимизированное линейное правило по оценённому состоянию",
            ),
            (
                taylor_env_factory,
                taylor_spec,
                ppo_taylor_policy,
                "ppo_filtered_taylor_state",
                "PPO-правило на оценённом тейлоровском наборе",
            ),
            (
                extended_env_factory,
                extended_spec,
                extended_policy,
                "optimized_linear_extended_state",
                "Оптимизированное линейное правило на расширенном оценённом состоянии",
            ),
            (
                extended_env_factory,
                extended_spec,
                ppo_extended_policy,
                "ppo_filtered_extended_state",
                "PPO-правило на расширенном оценённом состоянии",
            ),
            (
                raw_env_factory,
                raw_spec,
                history_policy,
                "history_observables_rule",
                "Историческое правило по наблюдаемым переменным",
            ),
            (
                taylor_env_factory,
                taylor_spec,
                calibrated_true_policy,
                "calibrated_true_state_rule",
                "Калиброванное правило по истинному состоянию",
            ),
            (
                taylor_env_factory,
                taylor_spec,
                true_state_policy,
                "optimized_linear_true_state",
                "Оптимизированный линейный ориентир полной информации",
            ),
        )
        for env_factory, scenario_spec, policy, policy_name, policy_label in policies:
            trace = true_state_trace if policy_name == "optimized_linear_true_state" else simulate_policy_episode(
                env_factory=env_factory,
                policy=policy,
                scenario_spec=scenario_spec,
                evaluation_seed=int(seed),
                policy_name=policy_name,
                policy_label=policy_label,
                training_seed=None,
            )
            metrics, path_frame = evaluate_policy_trace(
                policy_trace=trace,
                reference_trace=true_state_trace,
                scenario_spec=scenario_spec,
            )
            metrics["scenario_name"] = scenario_name
            metrics["scenario_label"] = SCENARIO_LABELS[scenario_name]
            policy_metric_rows.append(metrics)
            path_frame["scenario_name"] = scenario_name
            path_frame["scenario_label"] = SCENARIO_LABELS[scenario_name]
            policy_path_frames.append(path_frame)

    return pd.DataFrame(policy_metric_rows), pd.concat(policy_path_frames, ignore_index=True)


def _comparison_summary(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparison_specs = {
        "linear_minus_classical": ("optimized_linear_estimated_state", "classical_filtered_rule"),
        "extended_minus_linear": ("optimized_linear_extended_state", "optimized_linear_estimated_state"),
        "ppo_taylor_minus_linear": ("ppo_filtered_taylor_state", "optimized_linear_estimated_state"),
        "ppo_extended_minus_extended_linear": ("ppo_filtered_extended_state", "optimized_linear_extended_state"),
        "history_minus_classical": ("history_observables_rule", "classical_filtered_rule"),
        "linear_minus_history": ("optimized_linear_estimated_state", "history_observables_rule"),
        "linear_minus_optimized_true_state": ("optimized_linear_estimated_state", "optimized_linear_true_state"),
    }
    labels = {
        "linear_minus_classical": "Оптимизированное линейное минус классическое",
        "extended_minus_linear": "Расширенное линейное минус оптимизированное линейное",
        "ppo_taylor_minus_linear": "PPO на тейлоровском наборе минус оптимизированное линейное",
        "ppo_extended_minus_extended_linear": "PPO на расширенном состоянии минус расширенное линейное",
        "history_minus_classical": "Историческое по наблюдаемым минус классическое",
        "linear_minus_history": "Оценённое состояние минус история наблюдений",
        "linear_minus_optimized_true_state": "Оптимизированное линейное минус оптимизированное истинное состояние",
    }
    for scenario_name, frame in policy_metrics.groupby("scenario_name"):
        pivot = frame.pivot_table(index="evaluation_seed", columns="policy_name", values="cumulative_policy_loss", aggfunc="first")
        for comparison_name, (left, right) in comparison_specs.items():
            if left not in pivot or right not in pivot:
                continue
            deltas = pivot[left].to_numpy(dtype=float) - pivot[right].to_numpy(dtype=float)
            benchmark = pivot[right].to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_ci(deltas)
            rows.append({
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "comparison_name": comparison_name,
                "comparison_label": labels[comparison_name],
                "mean_delta_cumulative_loss": float(deltas.mean()),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "win_rate": float(np.mean(deltas < 0.0)),
                "probability_of_degradation": float(np.mean(deltas > 0.0)),
                "relative_improvement_pct": float(100.0 * np.mean(-deltas / benchmark)),
                "num_test_trajectories": int(deltas.size),
            })
    return pd.DataFrame(rows).sort_values(["scenario_name", "comparison_name"]).reset_index(drop=True)


def _policy_rule_ablation_summary(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    policy_specs = _policy_class_spec_frame().set_index("policy_name")
    for (scenario_name, scenario_label, policy_name, policy_label), frame in policy_metrics.groupby(
        ["scenario_name", "scenario_label", "policy_name", "policy_label"]
    ):
        values = frame["cumulative_policy_loss"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(values)
        row = {
            "scenario_name": scenario_name,
            "scenario_label": scenario_label,
            "policy_name": policy_name,
            "policy_label": policy_label,
            "mean_cumulative_policy_loss": float(values.mean()),
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "num_test_trajectories": int(values.size),
        }
        if policy_name in policy_specs.index:
            row.update(policy_specs.loc[policy_name].to_dict())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["scenario_name", "mean_cumulative_policy_loss"]).reset_index(drop=True)


def _paired_test_trajectory_losses(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        policy_metrics.pivot_table(
            index=["scenario_name", "scenario_label", "evaluation_seed"],
            columns="policy_name",
            values="cumulative_policy_loss",
            aggfunc="first",
        )
        .reset_index()
    )
    pivot.columns = [str(column) for column in pivot.columns]
    return pivot.sort_values(["scenario_name", "evaluation_seed"]).reset_index(drop=True)


def _same_input_comparisons(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        policy_metrics.pivot_table(
            index=["scenario_name", "scenario_label", "evaluation_seed"],
            columns="policy_name",
            values="cumulative_policy_loss",
            aggfunc="first",
        )
        .reset_index()
    )
    rows = []
    for scenario_name, frame in pivot.groupby("scenario_name"):
        if "classical_filtered_rule" not in frame or "optimized_linear_estimated_state" not in frame:
            continue
        deltas = (
            frame["optimized_linear_estimated_state"].to_numpy(dtype=float)
            - frame["classical_filtered_rule"].to_numpy(dtype=float)
        )
        benchmark = frame["classical_filtered_rule"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(deltas)
        rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": frame["scenario_label"].iloc[0],
                "left_policy": "optimized_linear_estimated_state",
                "right_policy": "classical_filtered_rule",
                "left_label": POLICY_CLASS_SPECS["optimized_linear_estimated_state"]["policy_label_ru"],
                "right_label": POLICY_CLASS_SPECS["classical_filtered_rule"]["policy_label_ru"],
                "what_changes": "Настройка коэффициентов при том же информационном наборе и той же линейной форме",
                "what_fixed": "Оценённое макроэкономическое состояние, линейное правило, одинаковый тестовый набор",
                "mean_delta_cumulative_loss": float(deltas.mean()),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "win_rate": float(np.mean(deltas < 0.0)),
                "probability_of_degradation": float(np.mean(deltas > 0.0)),
                "relative_improvement_pct": float(100.0 * np.mean(-deltas / benchmark)),
                "num_test_trajectories": int(deltas.size),
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def _input_set_ablation(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        policy_metrics.pivot_table(
            index=["scenario_name", "scenario_label", "evaluation_seed"],
            columns="policy_name",
            values="cumulative_policy_loss",
            aggfunc="first",
        )
        .reset_index()
    )
    rows = []
    for scenario_name, frame in pivot.groupby("scenario_name"):
        if "optimized_linear_extended_state" not in frame or "optimized_linear_estimated_state" not in frame:
            continue
        deltas = (
            frame["optimized_linear_extended_state"].to_numpy(dtype=float)
            - frame["optimized_linear_estimated_state"].to_numpy(dtype=float)
        )
        benchmark = frame["optimized_linear_estimated_state"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(deltas)
        rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": frame["scenario_label"].iloc[0],
                "left_policy": "optimized_linear_extended_state",
                "right_policy": "optimized_linear_estimated_state",
                "left_label": POLICY_CLASS_SPECS["optimized_linear_extended_state"]["policy_label_ru"],
                "right_label": POLICY_CLASS_SPECS["optimized_linear_estimated_state"]["policy_label_ru"],
                "what_changes": "Расширение входного набора при той же линейной форме",
                "what_fixed": "Оценённое состояние, линейное правило, одинаковый тестовый набор",
                "mean_delta_cumulative_loss": float(deltas.mean()),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "win_rate": float(np.mean(deltas < 0.0)),
                "probability_of_degradation": float(np.mean(deltas > 0.0)),
                "relative_improvement_pct": float(100.0 * np.mean(-deltas / benchmark)),
                "num_test_trajectories": int(deltas.size),
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def _history_vs_filtered_comparisons(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        policy_metrics.pivot_table(
            index=["scenario_name", "scenario_label", "evaluation_seed"],
            columns="policy_name",
            values="cumulative_policy_loss",
            aggfunc="first",
        )
        .reset_index()
    )
    rows = []
    for scenario_name, frame in pivot.groupby("scenario_name"):
        if "history_observables_rule" not in frame or "optimized_linear_estimated_state" not in frame:
            continue
        deltas = (
            frame["history_observables_rule"].to_numpy(dtype=float)
            - frame["optimized_linear_estimated_state"].to_numpy(dtype=float)
        )
        benchmark = frame["optimized_linear_estimated_state"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_ci(deltas)
        rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": frame["scenario_label"].iloc[0],
                "left_policy": "history_observables_rule",
                "right_policy": "optimized_linear_estimated_state",
                "left_label": POLICY_CLASS_SPECS["history_observables_rule"]["policy_label_ru"],
                "right_label": POLICY_CLASS_SPECS["optimized_linear_estimated_state"]["policy_label_ru"],
                "what_changes": "История наблюдений против явного оценивания состояния",
                "what_fixed": "Простое правило, отбор по одним валидационным траекториям, одинаковый тестовый набор",
                "mean_delta_cumulative_loss": float(deltas.mean()),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "win_rate": float(np.mean(deltas < 0.0)),
                "probability_of_degradation": float(np.mean(deltas > 0.0)),
                "relative_improvement_pct": float(100.0 * np.mean(-deltas / benchmark)),
                "num_test_trajectories": int(deltas.size),
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def _ppo_same_input_comparisons(policy_metrics: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        policy_metrics.pivot_table(
            index=["scenario_name", "scenario_label", "evaluation_seed"],
            columns="policy_name",
            values="cumulative_policy_loss",
            aggfunc="first",
        )
        .reset_index()
    )
    specs = [
        (
            "ppo_filtered_taylor_state",
            "optimized_linear_estimated_state",
            "Нелинейная PPO-архитектура при том же тейлоровском входе",
            "Оценённый тейлоровский набор, одинаковый тестовый набор",
        ),
        (
            "ppo_filtered_extended_state",
            "optimized_linear_extended_state",
            "Нелинейная PPO-архитектура при том же расширенном входе",
            "Расширенное оценённое состояние, одинаковый тестовый набор",
        ),
    ]
    rows = []
    for scenario_name, frame in pivot.groupby("scenario_name"):
        for left_policy, right_policy, what_changes, what_fixed in specs:
            if left_policy not in frame or right_policy not in frame:
                continue
            deltas = frame[left_policy].to_numpy(dtype=float) - frame[right_policy].to_numpy(dtype=float)
            benchmark = frame[right_policy].to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_ci(deltas)
            rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": frame["scenario_label"].iloc[0],
                "left_policy": left_policy,
                "right_policy": right_policy,
                "left_label": POLICY_CLASS_SPECS[left_policy]["policy_label_ru"],
                "right_label": POLICY_CLASS_SPECS[right_policy]["policy_label_ru"],
                "what_changes": "Нелинейное правило PPO при том же наборе переменных"
                if "ppo_" in left_policy
                else what_changes,
                "what_fixed": "Тот же расширенный вход, те же тестовые траектории"
                if left_policy == "ppo_filtered_extended_state"
                else (
                    "Тот же тейлоровский вход, те же тестовые траектории"
                    if left_policy == "ppo_filtered_taylor_state"
                    else what_fixed
                ),
                "mean_delta_cumulative_loss": float(deltas.mean()),
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "win_rate": float(np.mean(deltas < 0.0)),
                "probability_of_degradation": float(np.mean(deltas > 0.0)),
                    "relative_improvement_pct": float(100.0 * np.mean(-deltas / benchmark)),
                    "num_test_trajectories": int(deltas.size),
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario_name", "left_policy"]).reset_index(drop=True)


def _component_decomposition(policy_paths: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        policy_paths.groupby(["scenario_name", "scenario_label", "evaluation_seed", "policy_name"])[
            ["policy_loss", "inflation_gap", "output_gap", "policy_rate"]
        ]
        .agg(
            cumulative_policy_loss=("policy_loss", "sum"),
            inflation_component=("inflation_gap", lambda x: float(np.sum(np.square(x.to_numpy(dtype=float))))),
            output_component=("output_gap", lambda x: float(0.5 * np.sum(np.square(x.to_numpy(dtype=float))))),
            rate_component=("policy_rate", lambda x: float(0.05 * np.sum(np.square(np.diff(x.to_numpy(dtype=float), prepend=0.0))))),
        )
        .reset_index()
    )
    rows = []
    comparisons = {
        "linear_vs_classical": ("optimized_linear_estimated_state", "classical_filtered_rule"),
        "extended_vs_linear": ("optimized_linear_extended_state", "optimized_linear_estimated_state"),
        "ppo_taylor_vs_linear": ("ppo_filtered_taylor_state", "optimized_linear_estimated_state"),
        "ppo_extended_vs_extended_linear": ("ppo_filtered_extended_state", "optimized_linear_extended_state"),
        "linear_vs_history": ("optimized_linear_estimated_state", "history_observables_rule"),
    }
    for scenario_name, frame in grouped.groupby("scenario_name"):
        pivot = frame.pivot_table(
            index="evaluation_seed",
            columns="policy_name",
            values=["inflation_component", "output_component", "rate_component"],
            aggfunc="first",
        )
        for comparison_name, (left, right) in comparisons.items():
            if ("inflation_component", left) not in pivot or ("inflation_component", right) not in pivot:
                continue
            rows.append({
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "comparison_name": comparison_name,
                "delta_inflation_component": float((pivot[("inflation_component", left)] - pivot[("inflation_component", right)]).mean()),
                "delta_output_component": float((pivot[("output_component", left)] - pivot[("output_component", right)]).mean()),
                "delta_rate_component": float((pivot[("rate_component", left)] - pivot[("rate_component", right)]).mean()),
            })
    return pd.DataFrame(rows).sort_values(["scenario_name", "comparison_name"]).reset_index(drop=True)


def _plot_delta_intervals(summary: pd.DataFrame, path: Path) -> None:
    data = summary[
        summary["comparison_name"].isin(("linear_minus_classical", "extended_minus_linear", "linear_minus_history"))
    ].copy()
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    order = [scenario for scenario in SCENARIO_NAMES if scenario in set(data["scenario_name"])]
    comparisons = [
        ("linear_minus_classical", "Линейное по оценке vs классическое", "#0b6e4f", -0.2),
        ("extended_minus_linear", "Расширенное состояние vs узкий вход", "#3a86ff", 0.0),
        ("linear_minus_history", "Оценка состояния vs история", "#ca6702", 0.2),
    ]
    x = np.arange(len(order))
    for comparison_name, label, color, offset in comparisons:
        available = [scenario for scenario in order if scenario in set(data.loc[data["comparison_name"] == comparison_name, "scenario_name"])]
        if not available:
            continue
        frame = data[data["comparison_name"] == comparison_name].set_index("scenario_name").loc[available]
        positions = np.array([order.index(scenario) for scenario in available], dtype=float)
        means = frame["mean_delta_cumulative_loss"].to_numpy(dtype=float)
        lower = means - frame["ci_lower"].to_numpy(dtype=float)
        upper = frame["ci_upper"].to_numpy(dtype=float) - means
        ax.errorbar(positions + offset, means, yerr=np.vstack([lower, upper]), fmt="o", capsize=4, color=color, label=label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    short_labels = {
        "macro_core_moderate_gap": "Базовый\nмакронабор,\nумеренная",
        "macro_core_strong_gap": "Базовый\nмакронабор,\nвысокая",
        "thin_information_moderate_gap": "Ограниченный\nнабор,\nумеренная",
        "thin_information_strong_gap": "Ограниченный\nнабор,\nвысокая",
    }
    ax.set_xticklabels([short_labels[scenario] for scenario in order])
    ax.set_ylabel("$\\Delta J$; ниже нуля означает выигрыш")
    ax.set_title("Расширенные сравнения правил на тестовых траекториях")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_ppo_same_input_intervals(summary: pd.DataFrame, path: Path) -> None:
    data = summary[
        summary["comparison_name"].isin(("ppo_taylor_minus_linear", "ppo_extended_minus_extended_linear"))
    ].copy()
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    order = [scenario for scenario in SCENARIO_NAMES if scenario in set(data["scenario_name"])]
    comparisons = [
        ("ppo_taylor_minus_linear", "PPO и линейное правило, тейлоровский вход", "#7b2cbf", -0.12),
        ("ppo_extended_minus_extended_linear", "PPO и линейное правило, расширенный вход", "#118ab2", 0.12),
    ]
    x = np.arange(len(order))
    for comparison_name, label, color, offset in comparisons:
        frame = data[data["comparison_name"] == comparison_name]
        if frame.empty:
            continue
        frame = frame.set_index("scenario_name").loc[order]
        means = frame["mean_delta_cumulative_loss"].to_numpy(dtype=float)
        lower = means - frame["ci_lower"].to_numpy(dtype=float)
        upper = frame["ci_upper"].to_numpy(dtype=float) - means
        ax.errorbar(x + offset, means, yerr=np.vstack([lower, upper]), fmt="o", capsize=4, color=color, label=label)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    short_labels = {
        "macro_core_moderate_gap": "Базовый\nмакронабор,\nумеренная",
        "macro_core_strong_gap": "Базовый\nмакронабор,\nвысокая",
        "thin_information_moderate_gap": "Ограниченный\nнабор,\nумеренная",
        "thin_information_strong_gap": "Ограниченный\nнабор,\nвысокая",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([short_labels[scenario] for scenario in order])
    ax.set_ylabel("$\\Delta J$; ниже нуля означает преимущество PPO")
    ax.set_title("Сравнение PPO и линейного правила при том же входе")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _plot_component_decomposition(components: pd.DataFrame, path: Path) -> None:
    data = components[components["comparison_name"] == "linear_vs_classical"].copy()
    order = [scenario for scenario in SCENARIO_NAMES if scenario in set(data["scenario_name"])]
    data = data.set_index("scenario_name").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    x = np.arange(len(data))
    positive_bottom = np.zeros(len(data))
    negative_bottom = np.zeros(len(data))
    specs = [
        ("delta_inflation_component", "Инфляция", "#ca6702"),
        ("delta_output_component", "Разрыв выпуска", "#0b6e4f"),
        ("delta_rate_component", "Сглаживание ставки", "#4361ee"),
    ]
    for column, label, color in specs:
        values = data[column].to_numpy(dtype=float)
        bottoms = np.where(values >= 0.0, positive_bottom, negative_bottom)
        ax.bar(x, values, bottom=bottoms, color=color, width=0.58, label=label)
        positive_bottom += np.where(values >= 0.0, values, 0.0)
        negative_bottom += np.where(values < 0.0, values, 0.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    short_labels = {
        "macro_core_moderate_gap": "Базовый\nмакронабор,\nумеренная",
        "macro_core_strong_gap": "Базовый\nмакронабор,\nвысокая",
        "thin_information_moderate_gap": "Ограниченный\nнабор,\nумеренная",
        "thin_information_strong_gap": "Ограниченный\nнабор,\nвысокая",
    }
    ax.set_xticklabels([short_labels[scenario] for scenario in order])
    ax.set_ylabel("Вклад в $\\Delta J$")
    ax.set_title("Разложение выигрыша оптимизированного линейного правила")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _run_full_hank_projection(
    *,
    policy_paths: pd.DataFrame,
    output_dir: Path,
    scenario_names: tuple[str, ...],
    policy_names: tuple[str, ...],
    shock_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    rows = []
    path_rows = []
    for scenario_name in scenario_names:
        for policy_name in policy_names:
            subset = policy_paths[
                (policy_paths["scenario_name"] == scenario_name)
                & (policy_paths["policy_name"] == policy_name)
            ].copy()
            if subset.empty:
                continue
            shock_path = (
                subset.groupby("period")["policy_rate"].mean().sort_index().to_numpy(dtype=float)
            )
            shock_path = shock_path[: hank_config.shock_T]
            if shock_path.size < hank_config.shock_T:
                shock_path = np.pad(shock_path, (0, hank_config.shock_T - shock_path.size))
            transition = None
            scale_used = math.nan
            solver_error = ""
            for scale in shock_scales:
                try:
                    transition = solve_transition(bundle, {"monetary_policy_shock": scale * shock_path})
                    scale_used = float(scale)
                    break
                except Exception as exc:  # pragma: no cover - records numerical solver failures.
                    solver_error = f"{type(exc).__name__}: {exc}"
                    transition = None
            if transition is None:
                rows.append({
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": policy_name,
                    "solver_success": 0,
                    "scale_used": math.nan,
                    "solver_error": solver_error,
                    "mean_shock_abs": float(np.mean(np.abs(shock_path))),
                    "peak_shock_abs": float(np.max(np.abs(shock_path))),
                    "full_hank_cumulative_loss": math.nan,
                    "peak_inflation_abs": math.nan,
                    "peak_output_gap_abs": math.nan,
                    "peak_consumption_abs": math.nan,
                    "peak_rate_abs": math.nan,
                })
                continue
            pi = transition["pi"]
            output = transition["output_gap"]
            rate = transition["i"]
            loss = pi**2 + 0.5 * output**2 + 0.05 * np.square(np.diff(rate, prepend=0.0))
            rows.append({
                "scenario_name": scenario_name,
                "scenario_label": SCENARIO_LABELS[scenario_name],
                "policy_name": policy_name,
                "solver_success": 1,
                "scale_used": scale_used,
                "solver_error": "",
                "mean_shock_abs": float(np.mean(np.abs(shock_path))),
                "peak_shock_abs": float(np.max(np.abs(shock_path))),
                "full_hank_cumulative_loss": float(np.sum(loss)),
                "peak_inflation_abs": float(np.max(np.abs(pi))),
                "peak_output_gap_abs": float(np.max(np.abs(output))),
                "peak_consumption_abs": float(np.max(np.abs(transition["C"]))),
                "peak_rate_abs": float(np.max(np.abs(rate))),
            })
            for period in range(len(pi)):
                path_rows.append({
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": policy_name,
                    "period": int(period),
                    "mean_policy_rate_path_used_as_shock": float(scale_used * shock_path[period]),
                    "projection_scale_used": scale_used,
                    "inflation_gap": float(pi[period]),
                    "output_gap": float(output[period]),
                    "policy_rate": float(rate[period]),
                    "consumption": float(transition["C"][period]),
                    "period_loss": float(loss[period]),
                })
    metrics = pd.DataFrame(rows)
    paths = pd.DataFrame(path_rows)
    metrics.to_csv(output_dir / "full_hank_projection_metrics.csv", index=False)
    paths.to_csv(output_dir / "full_hank_projection_paths.csv", index=False)
    return metrics, paths


def run_full_hank_projection_from_policy_paths(
    *,
    input_dir: str = "outputs/hank_regime_learning_stage6_policy_extensions",
    output_dir: str | None = None,
    scenario_names: tuple[str, ...] = ("thin_information_strong_gap",),
    policy_names: tuple[str, ...] = (
        "classical_filtered_rule",
        "optimized_linear_estimated_state",
        "ppo_filtered_taylor_state",
        "optimized_linear_extended_state",
        "ppo_filtered_extended_state",
        "history_observables_rule",
        "optimized_linear_true_state",
    ),
) -> dict[str, pd.DataFrame]:
    root = Path(input_dir)
    out = Path(output_dir) if output_dir is not None else root
    out.mkdir(parents=True, exist_ok=True)
    policy_paths = pd.read_csv(root / "policy_paths.csv")
    metrics, paths = _run_full_hank_projection(
        policy_paths=policy_paths,
        output_dir=out,
        scenario_names=scenario_names,
        policy_names=policy_names,
    )
    lines = [
        "# Full-HANK projection для правил этапа 6",
        "",
        "Средние тестовые траектории ставки из reduced-state экспериментов передаются в полную HANK как траектории monetary-policy shock. Это не является полной оптимизацией правила в full HANK; это проверка согласованности направления результатов при пропуске выбранных траекторий через full-HANK transition solver.",
        "",
        "Если full-scale траектория не сходится в nonlinear solver, используется последовательное уменьшение амплитуды. Поэтому поле `scale_used` важно для интерпретации: значение ниже единицы означает, что full-HANK solver принимает только локальную версию соответствующей policy path.",
        "",
        "## Результаты",
        "",
    ]
    for row in metrics.to_dict(orient="records"):
        if int(row["solver_success"]) == 1:
            lines.append(
                f"- {row['scenario_label']}, `{row['policy_name']}`: scale `{row['scale_used']:.2f}`, "
                f"full-HANK cumulative loss `{row['full_hank_cumulative_loss']:.4e}`, "
                f"peak inflation `{row['peak_inflation_abs']:.4e}`, peak output gap `{row['peak_output_gap_abs']:.4e}`."
            )
        else:
            lines.append(
                f"- {row['scenario_label']}, `{row['policy_name']}`: solver не сошелся; последняя ошибка `{row['solver_error']}`."
            )
    (out / "report_full_hank_projection.md").write_text("\n".join(lines), encoding="utf-8")
    return {"full_hank_metrics": metrics, "full_hank_paths": paths}


def _load_stage6_linear_params(input_dir: str | Path) -> dict[str, dict[str, object]]:
    frame = pd.read_csv(Path(input_dir) / "selected_rule_specs.csv")
    params: dict[str, dict[str, object]] = {}
    for scenario_name in frame["scenario_name"].dropna().unique():
        scenario_frame = frame[frame["scenario_name"] == scenario_name]
        linear_row = scenario_frame[scenario_frame["policy_name"] == "optimized_linear_estimated_state"].iloc[0]
        extended_row = scenario_frame[scenario_frame["policy_name"] == "optimized_linear_extended_state"].iloc[0]
        true_row = scenario_frame[scenario_frame["policy_name"] == "optimized_linear_true_state"].iloc[0]
        params[str(scenario_name)] = {
            "linear_params": LinearRuleParameters(
                phi_pi=float(linear_row["phi_pi"]),
                phi_y=float(linear_row["phi_y"]),
                rho_i=float(linear_row["rho_i"]),
            ),
            "extended_params": ExtendedLinearRuleParameters(
                coeff_rstar=float(extended_row["coeff_rstar"]),
                coeff_productivity=float(extended_row["coeff_productivity"]),
                coeff_fiscal=float(extended_row["coeff_fiscal"]),
                coeff_pi=float(extended_row["coeff_pi"]),
                coeff_output=float(extended_row["coeff_output"]),
                coeff_stress=float(extended_row["coeff_stress"]),
                rho_i=float(extended_row["rho_i"]),
            ),
            "true_state_params": LinearRuleParameters(
                phi_pi=float(true_row["phi_pi"]),
                phi_y=float(true_row["phi_y"]),
                rho_i=float(true_row["rho_i"]),
            ),
        }
    return params


def run_ppo_seed_robustness_check(
    *,
    base_input_dir: str = "outputs/hank_regime_learning_stage6_policy_extensions",
    output_dir: str = "outputs/hank_regime_learning_stage6_ppo_seed_robustness",
    ppo_training_seeds: tuple[int, ...] = (11, 22, 33),
    validation_seeds: tuple[int, ...] = tuple(range(500, 510)),
    test_seeds: tuple[int, ...] = tuple(range(900, 950)),
    scenario_names: tuple[str, ...] = SCENARIO_NAMES,
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    linear_params_map = _load_stage6_linear_params(base_input_dir)
    _save_json(
        root / "ppo_seed_robustness_spec.json",
        {
            "base_input_dir": str(base_input_dir),
            "ppo_training_seeds": list(ppo_training_seeds),
            "validation_seeds": list(validation_seeds),
            "test_seeds": list(test_seeds),
            "scenario_names": list(scenario_names),
            "note": "Проверка устойчивости отрицательного результата для PPO при сравнении с линейным правилом на том же наборе входных переменных.",
        },
    )

    history_frames = []
    training_summary_frames = []
    selection_rows = []
    seed_metric_frames = []

    for scenario_name in scenario_names:
        params = linear_params_map[scenario_name]
        _overall_taylor_policy, taylor_history, taylor_training_summary, taylor_selected, taylor_seed_entries = _select_ppo_policy(
            scenario_name=scenario_name,
            input_mode="filtered_taylor_state",
            include_distributional_state=False,
            training_seeds=ppo_training_seeds,
            validation_seeds=validation_seeds,
            policy_name="ppo_filtered_taylor_state",
            policy_label="PPO-правило на оценённом тейлоровском наборе",
        )
        _overall_extended_policy, extended_history, extended_training_summary, extended_selected, extended_seed_entries = _select_ppo_policy(
            scenario_name=scenario_name,
            input_mode="filtered_extended_state",
            include_distributional_state=False,
            training_seeds=ppo_training_seeds,
            validation_seeds=validation_seeds,
            policy_name="ppo_filtered_extended_state",
            policy_label="PPO-правило на расширенном оценённом состоянии",
        )
        history_frames.extend([taylor_history, extended_history])
        training_summary_frames.extend([taylor_training_summary, extended_training_summary])
        selection_rows.extend([taylor_selected, extended_selected])
        seed_metric_frames.append(
            _evaluate_ppo_seed_robustness(
                scenario_name=scenario_name,
                validation_seeds=validation_seeds,
                test_seeds=test_seeds,
                linear_params=params["linear_params"],
                extended_params=params["extended_params"],
                true_state_params=params["true_state_params"],
                ppo_taylor_seed_entries=taylor_seed_entries,
                ppo_extended_seed_entries=extended_seed_entries,
            )
        )

    ppo_training_history = pd.concat(history_frames, ignore_index=True)
    ppo_training_seed_summary = pd.concat(training_summary_frames, ignore_index=True)
    ppo_selection_summary = pd.concat(selection_rows, ignore_index=True)
    ppo_seed_test_metrics = pd.concat(seed_metric_frames, ignore_index=True)
    ppo_seed_robustness = _summarize_ppo_seed_robustness(ppo_seed_test_metrics)

    overview_rows = []
    for (scenario_name, policy_name), frame in ppo_seed_robustness.groupby(["scenario_name", "policy_name"]):
        overview_rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": frame["scenario_label"].iloc[0],
                "policy_name": policy_name,
                "policy_label": frame["policy_label"].iloc[0],
                "num_training_seeds": int(frame["training_seed"].nunique()),
                "best_mean_delta_vs_linear": float(frame["mean_delta_vs_linear"].min()),
                "worst_mean_delta_vs_linear": float(frame["mean_delta_vs_linear"].max()),
                "best_win_rate_vs_linear": float(frame["win_rate_vs_linear"].max()),
                "max_test_unstable_episodes": int(frame["test_unstable_episodes"].max()),
            }
        )
    overview = pd.DataFrame(overview_rows).sort_values(["scenario_name", "policy_name"]).reset_index(drop=True)

    ppo_training_history.to_csv(root / "ppo_training_history.csv", index=False)
    ppo_training_seed_summary.to_csv(root / "ppo_training_seed_summary.csv", index=False)
    ppo_selection_summary.to_csv(root / "ppo_selection_summary.csv", index=False)
    ppo_seed_test_metrics.to_csv(root / "ppo_seed_test_metrics.csv", index=False)
    ppo_seed_robustness.to_csv(root / "ppo_seed_robustness_summary.csv", index=False)
    overview.to_csv(root / "ppo_seed_robustness_overview.csv", index=False)
    _write_ppo_seed_robustness_table(root, ppo_seed_robustness)

    lines = [
        "# Проверка устойчивости отрицательного результата для PPO",
        "",
        f"Проверка выполнена на `{len(ppo_training_seeds)}` запусках PPO: {_format_seed_span(ppo_training_seeds)}.",
        f"Для каждого запуска выбирался лучший сохранённый шаг по валидационным траекториям {_format_seed_span(validation_seeds)}.",
        f"Затем правило проверялось на `{len(test_seeds)}` независимых тестовых траекториях.",
        "",
        "Сравнение проводится только с линейным правилом на том же наборе входных переменных.",
        "",
        "## Краткий вывод",
        "",
    ]
    for _, row in overview.iterrows():
        lines.append(
            f"- {row['scenario_label']}, {row['policy_label']}: "
            f"лучшее значение `ΔJ` к линейному правилу равно `{row['best_mean_delta_vs_linear']:.4e}`, "
            f"худшее `{row['worst_mean_delta_vs_linear']:.4e}`, "
            f"наибольшая доля побед над линейным правилом `{row['best_win_rate_vs_linear']:.2f}`."
        )
    lines.extend(
        [
            "",
            "Положительное значение `ΔJ` означает, что PPO хуже линейного правила.",
            "",
            "## Что содержится в таблице",
            "",
            "- `Запуск` — номер запуска PPO.",
            "- `Итерация` — выбранный шаг обучения по валидационным траекториям.",
            "- `Потеря на тесте` — средняя накопленная потеря на независимых тестовых траекториях.",
            "- `ΔJ к линейному правилу` — средняя разность между PPO и линейным правилом на том же входе.",
            "- `Неустойчивые траектории` — число тестовых траекторий, на которых возникала неустойчивость.",
        ]
    )
    (root / "report_ppo_seed_robustness.md").write_text("\n".join(lines), encoding="utf-8")

    return {
        "ppo_training_history": ppo_training_history,
        "ppo_training_seed_summary": ppo_training_seed_summary,
        "ppo_selection_summary": ppo_selection_summary,
        "ppo_seed_test_metrics": ppo_seed_test_metrics,
        "ppo_seed_robustness": ppo_seed_robustness,
        "ppo_seed_robustness_overview": overview,
    }


def run_policy_extension_experiments(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_policy_extensions",
    ppo_training_seeds: tuple[int, ...] = (11, 22),
    validation_seeds: tuple[int, ...] = tuple(range(500, 510)),
    test_seeds: tuple[int, ...] = tuple(range(900, 950)),
    scenario_names: tuple[str, ...] = SCENARIO_NAMES,
    run_full_hank_projection: bool = False,
    full_hank_scenarios: tuple[str, ...] = ("thin_information_strong_gap",),
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    figures_dir = root / "figures"
    root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _save_json(
        root / "stage6_policy_extensions_spec.json",
        {
            "ppo_candidate_name": "larger_network",
            "ppo_training_seeds": list(ppo_training_seeds),
            "ppo_config": default_universal_candidate_lookup()["larger_network"].ppo.to_dict(),
            "validation_seeds": list(validation_seeds),
            "test_seeds": list(test_seeds),
            "scenario_names": list(scenario_names),
            "run_full_hank_projection": run_full_hank_projection,
            "full_hank_scenarios": list(full_hank_scenarios),
            "note": (
                "Правила отбираются на валидационных траекториях и оцениваются на независимых тестовых траекториях. "
                "Историческое правило использует внутреннее экспоненциально сглаженное состояние наблюдений."
            ),
        },
    )

    grid_frames = []
    selected_rows = []
    metrics_frames = []
    path_frames = []
    ppo_history_frames = []
    ppo_training_summary_frames = []
    ppo_selection_rows = []
    ppo_seed_metric_frames = []
    for scenario_name in scenario_names:
        linear_params, linear_grid = _select_linear_rule(scenario_name=scenario_name, validation_seeds=validation_seeds)
        extended_params, extended_grid = _select_extended_state_linear_rule(
            scenario_name=scenario_name,
            validation_seeds=validation_seeds,
            anchor_params=linear_params,
        )
        history_params, history_grid = _select_history_rule(scenario_name=scenario_name, validation_seeds=validation_seeds)
        true_state_params, true_state_grid = _select_true_state_linear_rule(
            scenario_name=scenario_name,
            validation_seeds=validation_seeds,
        )
        ppo_taylor_policy, ppo_taylor_history, ppo_taylor_training_summary, ppo_taylor_selected, ppo_taylor_seed_entries = _select_ppo_policy(
            scenario_name=scenario_name,
            input_mode="filtered_taylor_state",
            include_distributional_state=False,
            training_seeds=ppo_training_seeds,
            validation_seeds=validation_seeds,
            policy_name="ppo_filtered_taylor_state",
            policy_label="PPO-правило на оценённом тейлоровском наборе",
        )
        ppo_extended_policy, ppo_extended_history, ppo_extended_training_summary, ppo_extended_selected, ppo_extended_seed_entries = _select_ppo_policy(
            scenario_name=scenario_name,
            input_mode="filtered_extended_state",
            include_distributional_state=False,
            training_seeds=ppo_training_seeds,
            validation_seeds=validation_seeds,
            policy_name="ppo_filtered_extended_state",
            policy_label="PPO-правило на расширенном оценённом состоянии",
        )
        linear_grid["rule_family"] = "optimized_linear_estimated_state"
        extended_grid["rule_family"] = "optimized_linear_extended_state"
        history_grid["rule_family"] = "history_observables_rule"
        true_state_grid["rule_family"] = "optimized_linear_true_state"
        grid_frames.extend([linear_grid, extended_grid, history_grid, true_state_grid])
        ppo_history_frames.extend([ppo_taylor_history, ppo_extended_history])
        ppo_training_summary_frames.extend([ppo_taylor_training_summary, ppo_extended_training_summary])
        ppo_selection_rows.extend([ppo_taylor_selected, ppo_extended_selected])
        selected_rows.extend(
            [
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": "optimized_linear_estimated_state",
                    **linear_params.to_dict(),
                },
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": "optimized_linear_extended_state",
                    **extended_params.to_dict(),
                },
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": "history_observables_rule",
                    **history_params.to_dict(),
                },
                {
                    "scenario_name": scenario_name,
                    "scenario_label": SCENARIO_LABELS[scenario_name],
                    "policy_name": "optimized_linear_true_state",
                    **true_state_params.to_dict(),
                },
                ppo_taylor_selected.iloc[0].to_dict(),
                ppo_extended_selected.iloc[0].to_dict(),
            ]
        )
        metrics, paths = _evaluate_selected_rules(
            scenario_name=scenario_name,
            validation_seeds=validation_seeds,
            test_seeds=test_seeds,
            linear_params=linear_params,
            extended_params=extended_params,
            history_params=history_params,
            true_state_params=true_state_params,
            ppo_taylor_policy=ppo_taylor_policy,
            ppo_extended_policy=ppo_extended_policy,
        )
        metrics_frames.append(metrics)
        path_frames.append(paths)
        ppo_seed_metric_frames.append(
            _evaluate_ppo_seed_robustness(
                scenario_name=scenario_name,
                validation_seeds=validation_seeds,
                test_seeds=test_seeds,
                linear_params=linear_params,
                extended_params=extended_params,
                true_state_params=true_state_params,
                ppo_taylor_seed_entries=ppo_taylor_seed_entries,
                ppo_extended_seed_entries=ppo_extended_seed_entries,
            )
        )

    selection_grid = pd.concat(grid_frames, ignore_index=True)
    selected_specs = pd.DataFrame(selected_rows)
    policy_metrics = pd.concat(metrics_frames, ignore_index=True)
    policy_paths = pd.concat(path_frames, ignore_index=True)
    ppo_training_history = pd.concat(ppo_history_frames, ignore_index=True)
    ppo_training_seed_summary = pd.concat(ppo_training_summary_frames, ignore_index=True)
    ppo_selection_summary = pd.concat(ppo_selection_rows, ignore_index=True)
    ppo_seed_test_metrics = pd.concat(ppo_seed_metric_frames, ignore_index=True)
    ppo_seed_robustness = _summarize_ppo_seed_robustness(ppo_seed_test_metrics)
    comparison_summary = _comparison_summary(policy_metrics)
    components = _component_decomposition(policy_paths)
    policy_levels = _policy_rule_ablation_summary(policy_metrics)
    paired_losses = _paired_test_trajectory_losses(policy_metrics)
    same_input = _same_input_comparisons(policy_metrics)
    input_ablation = _input_set_ablation(policy_metrics)
    history_vs_filtered = _history_vs_filtered_comparisons(policy_metrics)
    ppo_same_input = _ppo_same_input_comparisons(policy_metrics)
    policy_class_spec = _policy_class_spec_frame()

    selection_grid.to_csv(root / "selection_grid_results.csv", index=False)
    selected_specs.to_csv(root / "selected_rule_specs.csv", index=False)
    policy_metrics.to_csv(root / "policy_metrics.csv", index=False)
    policy_paths.to_csv(root / "policy_paths.csv", index=False)
    ppo_training_history.to_csv(root / "ppo_training_history.csv", index=False)
    ppo_training_seed_summary.to_csv(root / "ppo_training_seed_summary.csv", index=False)
    ppo_selection_summary.to_csv(root / "ppo_selection_summary.csv", index=False)
    ppo_seed_test_metrics.to_csv(root / "ppo_seed_test_metrics.csv", index=False)
    ppo_seed_robustness.to_csv(root / "ppo_seed_robustness_summary.csv", index=False)
    comparison_summary.to_csv(root / "comparison_summary.csv", index=False)
    components.to_csv(root / "component_decomposition.csv", index=False)
    policy_levels.to_csv(root / "policy_rule_ablation_summary.csv", index=False)
    paired_losses.to_csv(root / "paired_test_trajectory_losses.csv", index=False)
    same_input.to_csv(root / "same_input_comparisons.csv", index=False)
    input_ablation.to_csv(root / "input_set_ablation.csv", index=False)
    history_vs_filtered.to_csv(root / "history_vs_filtered_comparisons.csv", index=False)
    ppo_same_input.to_csv(root / "ppo_same_input_comparisons.csv", index=False)
    (root / "policy_class_spec.json").write_text(
        json.dumps(policy_class_spec.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_extension_latex_tables(root, comparison_summary, selected_specs)
    _write_policy_rule_ablation_tables(root, policy_levels, same_input, input_ablation, history_vs_filtered, ppo_same_input)
    _write_ppo_seed_robustness_table(root, ppo_seed_robustness)

    _plot_delta_intervals(comparison_summary, figures_dir / "fig_01_extended_delta_intervals")
    _plot_component_decomposition(components, figures_dir / "fig_02_extended_component_decomposition")
    _plot_ppo_same_input_intervals(comparison_summary, figures_dir / "fig_03_ppo_same_input_intervals")

    full_hank_metrics = pd.DataFrame()
    full_hank_paths = pd.DataFrame()
    if run_full_hank_projection:
        full_hank_metrics, full_hank_paths = _run_full_hank_projection(
            policy_paths=policy_paths,
            output_dir=root,
            scenario_names=full_hank_scenarios,
            policy_names=(
                "classical_filtered_rule",
                "optimized_linear_estimated_state",
                "ppo_filtered_taylor_state",
                "optimized_linear_extended_state",
                "ppo_filtered_extended_state",
                "history_observables_rule",
                "optimized_linear_true_state",
            ),
        )

    report_lines = [
        "# Расширенные эксперименты этапа 6",
        "",
        "## Что добавлено",
        "",
        "- Оптимизированное линейное правило по оценённому состоянию.",
        "- PPO-правило на том же оценённом тейлоровском наборе.",
        "- Оптимизированное линейное правило на расширенном оценённом состоянии.",
        "- PPO-правило на том же расширенном оценённом состоянии.",
        "- Историческое правило по наблюдаемым переменным с внутренним сглаженным состоянием.",
        "- Оптимизированный линейный ориентир по истинному состоянию.",
        f"- PPO обучается на запусках {_format_seed_span(ppo_training_seeds)} с выбором сохранённого шага по валидационным траекториям.",
        f"- Отбор выполнен на валидационных траекториях {_format_seed_span(validation_seeds)}.",
        f"- Итоговая проверка выполнена на `{len(test_seeds)}` независимых тестовых траекториях.",
        "",
        "## Что именно изолируют сравнения",
        "",
        "- `classical_filtered_rule` и `optimized_linear_estimated_state` используют один и тот же узкий тейлоровский вход: оценённые компоненты `r^*_t`, `\\pi_t`, `y_t` и лагированную ставку. Их сравнение изолирует эффект настройки коэффициентов при той же линейной форме и том же информационном наборе.",
        "- `ppo_filtered_taylor_state` использует тот же узкий тейлоровский вход, что и `optimized_linear_estimated_state`. Их сравнение показывает, даёт ли PPO самостоятельный выигрыш при том же входном наборе.",
        "- `optimized_linear_extended_state` использует расширенное оценённое состояние: `r^*_t`, производственный фактор, фискальный фактор, инфляцию, выпуск, вероятность стрессового режима и лагированную ставку. Его сравнение с узким линейным правилом изолирует эффект расширения входного набора при той же линейной форме.",
        "- `ppo_filtered_extended_state` использует тот же расширенный вход, что и `optimized_linear_extended_state`. Их сравнение даёт такую же проверку на более широком наборе переменных.",
        "- `history_observables_rule` использует другой вход: наблюдаемые переменные и их сглаженную историю. Его сравнение с правилом по оценённому состоянию не является чистым тестом фильтрации или нелинейности; здесь одновременно меняются набор входных переменных и способ формирования состояния.",
        "- `optimized_linear_true_state` является честно отобранным линейным ориентиром при истинном состоянии на том же узком тейлоровском наборе. Это не общая верхняя граница по всем классам правил, а сопоставимое линейное правило при полной информации.",
        "- В этом обновлённом блоке правила `ppo_filtered_taylor_state` и `ppo_filtered_extended_state` действительно обучаются, поэтому сравнение с линейным правилом при том же входе уже позволяет сделать прямой вывод о PPO.",
        "",
        "## Главный вывод",
        "",
        "Результаты показывают, что фиксированное узкое правило Тейлора является слишком слабым ориентиром. Значительная часть выигрыша достигается уже простым правилом того же линейного класса при тех же входных переменных, если коэффициенты выбраны по валидационным траекториям. Дополнительная проверка на расширенном оценённом состоянии позволяет отдельно измерить выигрыш от более богатого входного набора. Следовательно, выигрыш нельзя приписывать только более гибкой форме правила: существенную роль играют настройка и выбор входного представления.",
        "",
        "Отдельное сравнение PPO с сильными линейными правилами при том же входе интерпретируется только по парам `ppo_filtered_taylor_state` vs `optimized_linear_estimated_state` и `ppo_filtered_extended_state` vs `optimized_linear_extended_state`. Именно эти пары отвечают на вопрос, даёт ли PPO самостоятельный выигрыш сверх уже оптимизированного линейного правила на том же входе.",
        "",
        "Отдельно сохраняется результат о правиле по наблюдаемым переменным с использованием истории: в текущей сетке оно сравнивается с правилом по оценённому состоянию как альтернативная архитектура информационного использования, а не как чистый тест PPO или фильтрации.",
        "",
        "## Попарные сравнения",
        "",
    ]
    for row in comparison_summary.to_dict(orient="records"):
        if row["comparison_name"] in {
            "linear_minus_classical",
            "extended_minus_linear",
            "ppo_taylor_minus_linear",
            "ppo_extended_minus_extended_linear",
            "history_minus_classical",
            "linear_minus_history",
            "linear_minus_optimized_true_state",
        }:
            report_lines.append(
                f"- {row['scenario_label']}, {row['comparison_label']}: "
                f"delta `{row['mean_delta_cumulative_loss']:.4e}`, "
                f"95% ДИ `[{row['ci_lower']:.4e}; {row['ci_upper']:.4e}]`, "
                f"доля побед `{row['win_rate']:.2f}`."
            )
    if not ppo_seed_robustness.empty:
        report_lines.extend(["", "## Устойчивость отрицательного результата для PPO", ""])
        for (policy_name, policy_label), frame in ppo_seed_robustness.groupby(["policy_name", "policy_label"]):
            best_win_rate = float(frame["win_rate_vs_linear"].max())
            best_delta = float(frame["mean_delta_vs_linear"].min())
            worst_delta = float(frame["mean_delta_vs_linear"].max())
            report_lines.append(
                f"- {policy_label}: проверено `{int(frame['training_seed'].nunique())}` запусков; "
                f"лучшая средняя разность к линейному правилу `{best_delta:.4e}`, "
                f"худшая `{worst_delta:.4e}`, "
                f"максимальная доля побед над линейным правилом `{best_win_rate:.2f}`."
            )
    if run_full_hank_projection:
        report_lines.extend(
            [
                "",
                "## Full-HANK projection",
                "",
                "Средние тестовые траектории ставки из reduced-state экспериментов дополнительно переданы в полную HANK как экзогенные траектории monetary-policy shock. Это не заменяет полную оптимизацию в HANK, но служит проверкой того, не исчезает ли различие между правилами при пропуске через full-HANK transition solver.",
            ]
        )
    (root / "report_stage6_policy_extensions.md").write_text("\n".join(report_lines), encoding="utf-8")

    if not ppo_seed_robustness.empty:
        robustness_lines = [
            "# Проверка устойчивости отрицательного результата для PPO",
            "",
            f"Проверка выполнена на `{len(ppo_training_seeds)}` запусках PPO: {_format_seed_span(ppo_training_seeds)}.",
            f"Для каждого запуска выбирался лучший сохранённый шаг по валидационным траекториям {_format_seed_span(validation_seeds)}, после чего правило проверялось на `{len(test_seeds)}` независимых тестовых траекториях.",
            "",
            "Сравнение проводится только с линейным правилом на том же наборе входных переменных.",
            "",
            "## Краткий вывод",
            "",
        ]
        for (policy_name, policy_label), frame in ppo_seed_robustness.groupby(["policy_name", "policy_label"]):
            best_row = frame.sort_values("mean_delta_vs_linear").iloc[0]
            worst_row = frame.sort_values("mean_delta_vs_linear").iloc[-1]
            robustness_lines.append(
                f"- {policy_label}: ни один из проверенных запусков не дал устойчивого преимущества над линейным правилом. "
                f"Лучший запуск `{int(best_row['training_seed'])}` дал среднюю разность `{float(best_row['mean_delta_vs_linear']):.4e}`, "
                f"худший запуск `{int(worst_row['training_seed'])}` дал `{float(worst_row['mean_delta_vs_linear']):.4e}`."
            )
        robustness_lines.extend(["", "## Что смотреть в таблице", ""])
        robustness_lines.extend(
            [
                "- `Запуск` — номер запуска PPO.",
                "- `Итерация` — выбранный шаг обучения по валидационным траекториям.",
                "- `Потеря на тесте` — средняя накопленная потеря на независимых тестовых траекториях.",
                "- `ΔJ к линейному правилу` — разность между PPO и линейным правилом на том же входе; положительное значение означает, что PPO хуже.",
                "- `Неустойчивые траектории` — число тестовых траекторий, на которых возникала неустойчивость.",
            ]
        )
        (root / "report_ppo_seed_robustness.md").write_text("\n".join(robustness_lines), encoding="utf-8")

    return {
        "selection_grid": selection_grid,
        "selected_specs": selected_specs,
        "policy_metrics": policy_metrics,
        "policy_paths": policy_paths,
        "comparison_summary": comparison_summary,
        "components": components,
        "policy_levels": policy_levels,
        "paired_losses": paired_losses,
        "same_input": same_input,
        "input_ablation": input_ablation,
        "history_vs_filtered": history_vs_filtered,
        "ppo_same_input": ppo_same_input,
        "policy_class_spec": policy_class_spec,
        "ppo_training_history": ppo_training_history,
        "ppo_training_seed_summary": ppo_training_seed_summary,
        "ppo_selection_summary": ppo_selection_summary,
        "ppo_seed_test_metrics": ppo_seed_test_metrics,
        "ppo_seed_robustness": ppo_seed_robustness,
        "full_hank_metrics": full_hank_metrics,
        "full_hank_paths": full_hank_paths,
    }
