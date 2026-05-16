# Карта текущей структуры

Основная ветка оставляет только линию работы:

```text
HANK / метод последовательностей
+ неполная наблюдаемость
+ предельная ценность распределительной информации
```

Единая рамка проекта: ценность распределительной информации для денежно-кредитной политики при
неполной наблюдаемости в локальной HANK/SSJ-среде.

## Рабочие папки

- `hank/` — экономическое HANK-ядро.
- `hank/audit.py` — аудит стационарного распределения, policy residuals, market clearing и nonlinear transition solver residuals.
- `hank_ssj/` — интерфейс HANK/SSJ-артефактов и информационных состояний.
- `policy/` — интерпретируемые правила ставки, сравнение потерь и статистический вывод.
- `experiments/` — эксперименты HANK/SSJ-линии.
- `docs/` — постановка, дорожная карта и план проверки.
- `article/` — черновик статьи.
- `outputs/hank_core/` — текущие HANK-артефакты.
- `outputs/ssj/` — текущие SSJ-артефакты.

## Текущие эксперименты

- `experiments/exp00_hank_core_audit.py` — аудит HANK-ядра перед SSJ/VOI-слоем.
- `experiments/exp01_ssj_irfs.py` — экспорт якобиана HANK в матричный SSJ-формат.
- `experiments/exp02_build_hank_observables.py` — сборка \(q_t\) из HANK-выходов.
- `experiments/exp03_build_observations.py` — построение шумных наблюдений.
- `experiments/exp04_filter_states.py` — построение фильтрованных состояний.
- `experiments/exp05_build_information_inputs.py` — сборка входов информационных состояний.
- `experiments/exp06_build_shock_library.py` — HANK-библиотека откликов на неполитические шоки.
- `experiments/exp07_generate_stochastic_hank_paths.py` — стохастические HANK/SSJ-траектории.
- `experiments/exp08_main_voi.py` — основной расчёт ценности информационных состояний.
- `experiments/exp09_make_placebo_inputs.py` — входы для проверок с искусственными распределительными статистиками.
- `experiments/exp10_summarize_placebo_tests.py` — сводка проверок с искусственными распределительными статистиками.
- `experiments/exp11_noise_sensitivity.py` — чувствительность ценности распределительной информации к шуму наблюдений.
- `experiments/exp12_build_article_figures.py` — фигуры для текущего черновика статьи.
- `experiments/exp13_no_distributional_signal.py` — проверка, в которой агрегатные траектории сохраняются, а распределительный сигнал выключается.
- `experiments/exp14_distributional_signal_strength.py` — чувствительность к силе распределительного сигнала при фиксированном масштабе шума наблюдений.
- `experiments/exp15_loss_decomposition.py` — разложение снижения потерь по компонентам функции потерь.
- `experiments/exp16_income_risk_calibration.py` — пересчёт HANK/SSJ-пайплайна при разных калибровках доходного риска.
- `experiments/exp17_trajectory_count_robustness.py` — проверка устойчивости основного результата к числу стохастических HANK/SSJ-траекторий.
- `experiments/exp18_income_risk_shock_source.py` — проверка с временным шоком доходного риска как отдельным HANK/SSJ-источником распределительной динамики.
- `experiments/exp19_liquid_wedge_channel_calibration.py` — пересчёт HANK/SSJ-пайплайна при разных значениях клина ликвидной доходности.
- `experiments/exp20_joint_kalman_filter.py` — совместный линейный фильтр Калмана для всех переменных состояния.
- `experiments/exp21_main_voi_joint_filter.py` — основной расчёт ценности информации на совместном фильтре.
- `experiments/exp22_closed_loop_evaluation.py` — локальная проверка с обратной связью без переобучения правил.
- `experiments/exp22_mechanism_optimal_rate_projection.py` — механизм через прогноз локально SSJ-оптимальной ставки.
- `experiments/exp23_distributional_identification_battery.py` — идентификационные проверки против механического расширения числа признаков.
- `experiments/exp24_robustness_comparative_statics.py` — сравнительная статика робастности по экономическим вопросам.
- `experiments/exp25_policy_class_robustness.py` — проверка устойчивости результата к классу правила.
- `experiments/exp26_information_frontier.py` — информационная граница, предельные выигрыши и устойчивость рангов информационных уровней.
- `experiments/exp28_optimizer_noise_audit.py` — аудит численного шума оптимизации правил.
- `experiments/exp29_large_sample_joint_filter.py` — large-sample split с отдельными shock seed, observation seed и кластерной статистикой.
- `experiments/exp30_closed_loop_distributional_ssj.py` — closed-loop проверка с прямыми распределительными HANK/SSJ-откликами ставки.
- `experiments/exp31_validate_ssj_jacobians.py` — конечная разностная проверка локальной линейной аппроксимации HANK/SSJ.
- `experiments/exp32_null_distribution_channel.py` — negative control: нулевой распределительный канал и ложноположительная частота.
- `experiments/exp33_known_distribution_channel.py` — positive control: восстановление искусственно заданного распределительного канала.
- `experiments/exp34_distributional_feature_decomposition.py` — разложение MVOI по отдельным распределительным признакам, Shapley и компонентам потерь.
- `experiments/exp35_mechanism_residualized_crossfit.py` — cross-fit residual mechanism: остаточная распределительная информация после фильтрованных агрегатов.
- `experiments/exp36_lqg_information_oracle.py` — LQG/LQR Riccati-ориентир для совместной линейной state-space задачи.
- `experiments/exp39_mechanism_dashboard.py` — финальный mechanism dashboard с cross-fitting по shock seed, двумя целями и bins выигрыша распределительной информации.

## Общий статистический слой

- `policy/inference.py` — парный bootstrap, кластерный bootstrap, wild-bootstrap, sign-flip проверка, перестановочная проверка и поправка Бенджамини--Хохберга.
- `policy/lqg_oracle.py` — full-state LQR, partial-information LQG и оценка ограниченных правил в той же линейной state-space системе.
- `hank_ssj/distributional_jacobians.py` — построение прямых распределительных откликов HANK transition solver на траекторию ставки.
