from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InformationStateSpec:
    name: str
    label: str
    variables: tuple[str, ...]
    role: str


def default_information_states() -> tuple[InformationStateSpec, ...]:
    return (
        InformationStateSpec(
            name="aggregate_only",
            label="Только агрегаты",
            variables=("observed_inflation_gap", "observed_output_gap"),
            role="Базовый набор макроэкономических индикаторов.",
        ),
        InformationStateSpec(
            name="filtered_aggregates",
            label="Оценённые агрегаты",
            variables=("filtered_inflation_gap", "filtered_output_gap", "filtered_natural_rate_gap"),
            role="Проверяет ценность фильтрации агрегатного состояния.",
        ),
        InformationStateSpec(
            name="distributional",
            label="Агрегаты и распределение",
            variables=(
                "filtered_inflation_gap",
                "filtered_output_gap",
                "filtered_natural_rate_gap",
                "filtered_mean_mpc",
                "filtered_low_liquidity_share",
            ),
            role="Главный объект: ценность распределительных статистик.",
        ),
        InformationStateSpec(
            name="distributional_mpc",
            label="Агрегаты и средняя MPC",
            variables=(
                "filtered_inflation_gap",
                "filtered_output_gap",
                "filtered_natural_rate_gap",
                "filtered_mean_mpc",
            ),
            role="Проверяет отдельную ценность средней MPC.",
        ),
        InformationStateSpec(
            name="distributional_liquidity",
            label="Агрегаты и доля низколиквидных",
            variables=(
                "filtered_inflation_gap",
                "filtered_output_gap",
                "filtered_natural_rate_gap",
                "filtered_low_liquidity_share",
            ),
            role="Проверяет отдельную ценность доли низколиквидных домохозяйств.",
        ),
        InformationStateSpec(
            name="full_information",
            label="Полная информация",
            variables=(
                "true_inflation_gap",
                "true_output_gap",
                "true_natural_rate_gap",
                "true_mean_mpc",
                "true_low_liquidity_share",
            ),
            role="Верхняя граница, а не реалистичный информационный набор.",
        ),
    )
