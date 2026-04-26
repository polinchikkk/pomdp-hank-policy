# pomdp-hank-policy

Текущая рабочая ветка проекта сфокусирована на денежно-кредитной политике в полной двухактивной HANK-среде при неполной наблюдаемости. Главная рамка проекта: полная HANK-модель используется как истинная экономика, а низкоразмерный слой состояния выступает как интерфейс политики, то есть как представление той информации, которую регулятор может восстановить по наблюдаемым данным и использовать для выбора ставки.

Поэтому низкоразмерное состояние здесь не трактуется как слабая замена полной HANK. Напротив, это отдельный объект исследования: какая сжатая информация об агрегированных шоках, распределительных состояниях и скрытых режимах достаточна для денежно-кредитной политики в полной HANK-среде.

На `main` оставлена только актуальная HANK-линия:
- [hank_full_baseline](/Users/polinazosimova/pomdp-hank-policy/hank_full_baseline)
- [hank_partial_info_baseline](/Users/polinazosimova/pomdp-hank-policy/hank_partial_info_baseline)
- [hank_learning_policy_baseline](/Users/polinazosimova/pomdp-hank-policy/hank_learning_policy_baseline)
- [regime_switching_baseline](/Users/polinazosimova/pomdp-hank-policy/regime_switching_baseline)
- [hank_regime_learning_baseline](/Users/polinazosimova/pomdp-hank-policy/hank_regime_learning_baseline)
- [scripts/run_hank.py](/Users/polinazosimova/pomdp-hank-policy/scripts/run_hank.py)
- [scripts/run_hank_partial_info.py](/Users/polinazosimova/pomdp-hank-policy/scripts/run_hank_partial_info.py)
- [scripts/run_hank_learning.py](/Users/polinazosimova/pomdp-hank-policy/scripts/run_hank_learning.py)
- [scripts/run_hank_regime.py](/Users/polinazosimova/pomdp-hank-policy/scripts/run_hank_regime.py)
- [scripts/run_hank_regime_learning.py](/Users/polinazosimova/pomdp-hank-policy/scripts/run_hank_regime_learning.py)
- [scripts/run_hank_regime_core_matrix.py](/Users/polinazosimova/pomdp-hank-policy/scripts/run_hank_regime_core_matrix.py)
- [outputs/hank_policy_stage2](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_policy_stage2)
- [outputs/hank_partial_info_stage3](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_partial_info_stage3)
- [outputs/hank_learning_stage4](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_learning_stage4)
- [outputs/hank_regime_switching_stage5](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_regime_switching_stage5)
- [outputs/hank_regime_learning_stage6_core_matrix](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_regime_learning_stage6_core_matrix)
- [outputs/hank_regime_learning_stage6_universal_tuning](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_regime_learning_stage6_universal_tuning)
- [outputs/hank_regime_learning_stage6_reduced_state_validation](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_regime_learning_stage6_reduced_state_validation)

## Быстрый запуск

Установка зависимостей:

```bash
python3 -m pip install -r requirements.txt
```

Полный запуск HANK baseline:

```bash
python3 scripts/run_hank.py
```

Запуск HANK baseline с неполной информацией:

```bash
python3 scripts/run_hank_partial_info.py
```

Запуск learning-based policy layer поверх partial-information HANK:

```bash
python3 scripts/run_hank_learning.py
```

Запуск regime-switching HANK baseline:

```bash
python3 scripts/run_hank_regime.py
```

Запуск regime-learning experiments:

```bash
python3 scripts/run_hank_regime_learning.py
```

Сборка чистой матрицы монетарных сравнений для основного текста:

```bash
python3 scripts/run_hank_regime_core_matrix.py
```

Валидация редуцированного представления состояния:

```bash
python3 scripts/run_hank_regime_reduced_state_validation.py --run-full-hank-projection
```

По умолчанию результаты сохраняются в:

- `outputs/hank_policy_stage2`
- `outputs/hank_partial_info_stage3`
- `outputs/hank_learning_stage4`
- `outputs/hank_regime_switching_stage5`
- `outputs/hank_regime_learning_stage6_core_matrix`
- `outputs/hank_regime_learning_stage6_universal_tuning`
- `outputs/hank_regime_learning_stage6_reduced_state_validation`

## Что считает текущий baseline

- стационарное равновесие полной two-asset HANK-модели;
- распределение домохозяйств по ликвидному и неликвидному богатству;
- функции политики домохозяйств;
- переходную динамику после монетарного шока;
- импульсные отклики агрегатов;
- групповые и распределительные эффекты;
- validation и robustness-блок по MPC, WHtM и household-side calibration.

## Что считает partial-information baseline

- низкоразмерное представление скрытого состояния полной HANK-среды как интерфейс политики;
- синтетические траектории скрытого состояния и шумных макроэкономических наблюдений;
- фильтрацию скрытого состояния;
- классическую схему `наблюдения -> фильтрация -> правило`;
- сравнение с benchmark при полной информации по функции потерь, отклонению ставки и распределительным метрикам.

## Что считает learning-based policy baseline

- continuous-action residual PPO поверх того же reduced HANK state-space и того же фильтра;
- сравнение `классическая схема фильтрации и фиксированного правила` против `фильтрации и обучаемого правила`;
- основные сценарии: `macro_core`, `full_macro`, `thin_information`, `high_noise`, `distribution_augmented`;
- абляции: `оценённое состояние + неопределённость`, `наблюдаемые переменные`, `без распределительных компонент состояния`;
- policy, macro and distributional evaluation уже на полном HANK transition solver.

## Что считают regime-switching расширения

- stage 5: скрытые режимы поверх reduced-state HANK с переключающимся фильтром и классическим правилом;
- stage 6: обучаемые правила по отфильтрованному состоянию и по наблюдениям в HANK-среде со скрытыми режимами;
- основной результат stage 6 для основного текста теперь собран в чистую матрицу монетарных сравнений: `полная информация, фиксированное правило`, `фильтрация, фиксированное правило`, `фильтрация, гибкое правило`, `наблюдаемые переменные, гибкое правило`;
- главный вывод этой матрицы: гибкость правила по отфильтрованному состоянию особенно важна при тонком информационном наборе, тогда как в сценариях с более богатым набором наблюдений различие между правилами по наблюдаемым переменным и по отфильтрованному состоянию невелико.
- отдельный validation package проверяет, что редуцированное состояние является содержательно и прогностически значимым интерфейсом политики: для него собраны таблица экономической интерпретации компонент, out-of-sample forecast sufficiency tests и проверка сохранения ранжирования правил через full-HANK projection.

## Ключевые артефакты

- `outputs/hank_policy_stage2/model_spec.json`
- `outputs/hank_policy_stage2/policy_config.json`
- `outputs/hank_policy_stage2/scenario_config.json`
- `outputs/hank_policy_stage2/group_definition_spec.json`
- `outputs/hank_policy_stage2/steady_state_aggregates.json`
- `outputs/hank_policy_stage2/diagnostics_summary.json`
- `outputs/hank_policy_stage2/aggregate_paths.csv`
- `outputs/hank_policy_stage2/distribution_paths.csv`
- `outputs/hank_policy_stage2/group_paths.csv`
- `outputs/hank_policy_stage2/group_profiles.csv`
- `outputs/hank_policy_stage2/group_consumption_irfs.csv`
- `outputs/hank_policy_stage2/group_income_irfs.csv`
- `outputs/hank_policy_stage2/channel_decomposition.csv`
- `outputs/hank_policy_stage2/mpc_validation.csv`
- `outputs/hank_policy_stage2/transfer_mpc_validation.csv`
- `outputs/hank_policy_stage2/mpc_measure_spec.json`
- `outputs/hank_policy_stage2/wealthy_htm_sensitivity.csv`
- `outputs/hank_policy_stage2/reference_spec.json`
- `outputs/hank_policy_stage2/sequence_jacobian_reference_parameters.csv`
- `outputs/hank_policy_stage2/sequence_jacobian_reference_summary.csv`
- `outputs/hank_policy_stage2/household_robustness_summary.csv`
- `outputs/hank_policy_stage2/household_robustness_group_peaks.csv`
- `outputs/hank_policy_stage2/report_stage2_hank_policy.md`

Для partial-information HANK:

- `outputs/hank_partial_info_stage3/model_spec.json`
- `outputs/hank_partial_info_stage3/filter_spec.json`
- `outputs/hank_partial_info_stage3/policy_spec.json`
- `outputs/hank_partial_info_stage3/scenario_spec.json`
- `outputs/hank_partial_info_stage3/reduced_state_space.json`
- `outputs/hank_partial_info_stage3/true_state_paths.csv`
- `outputs/hank_partial_info_stage3/filtered_state_paths.csv`
- `outputs/hank_partial_info_stage3/observations.csv`
- `outputs/hank_partial_info_stage3/aggregate_paths.csv`
- `outputs/hank_partial_info_stage3/distribution_stats.csv`
- `outputs/hank_partial_info_stage3/group_paths.csv`
- `outputs/hank_partial_info_stage3/filter_metrics.csv`
- `outputs/hank_partial_info_stage3/policy_metrics.csv`
- `outputs/hank_partial_info_stage3/report_stage3_partial_information_hank.md`

Для stage 4 learning-based policy:

- `outputs/hank_learning_stage4/model_spec.json`
- `outputs/hank_learning_stage4/filter_spec.json`
- `outputs/hank_learning_stage4/policy_spec.json`
- `outputs/hank_learning_stage4/scenario_spec.json`
- `outputs/hank_learning_stage4/reduced_state_space.json`
- `outputs/hank_learning_stage4/training_history.csv`
- `outputs/hank_learning_stage4/training_seed_summary.csv`
- `outputs/hank_learning_stage4/policy_metrics.csv`
- `outputs/hank_learning_stage4/policy_comparison.csv`
- `outputs/hank_learning_stage4/aggregate_paths.csv`
- `outputs/hank_learning_stage4/distribution_stats.csv`
- `outputs/hank_learning_stage4/group_paths.csv`
- `outputs/hank_learning_stage4/report_stage4_learning_policy_hank.md`

Для stage 5 regime-switching baseline:

- `outputs/hank_regime_switching_stage5/filter_metrics.csv`
- `outputs/hank_regime_switching_stage5/policy_metrics.csv`
- `outputs/hank_regime_switching_stage5/regime_paths.csv`
- `outputs/hank_regime_switching_stage5/report_stage5_regime_switching_hank.md`

Для stage 6 tuned regime-learning results:

- `outputs/hank_regime_learning_stage6_universal_tuning/candidate_summary.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/scenario_results.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/best_candidate_core_map.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/best_candidate_delta_loss_matrix.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/best_vs_baseline_by_scenario.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/best_candidate_seed_win_rates.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/report_universal_tuning.md`

Для stage 6 summary и diagnostics:

- `outputs/hank_regime_learning_stage6_core_matrix/main_policy_matrix.csv`
- `outputs/hank_regime_learning_stage6_core_matrix/core_headline_table.csv`
- `outputs/hank_regime_learning_stage6_core_matrix/core_value_summary_table.csv`
- `outputs/hank_regime_learning_stage6_core_matrix/core_comparisons.csv`
- `outputs/hank_regime_learning_stage6_core_matrix/loss_component_decomposition.csv`
- `outputs/hank_regime_learning_stage6_core_matrix/stage6_core_text_blocks.tex`
- `outputs/hank_regime_learning_stage6_core_matrix/table_stage6_core_value_summary.tex`
- `outputs/hank_regime_learning_stage6_core_matrix/table_stage6_core_pairwise_summary.tex`
- `outputs/hank_regime_learning_stage6_core_matrix/final_evidence_package.md`
- `outputs/hank_regime_learning_stage6_policy_extensions/comparison_summary.csv`
- `outputs/hank_regime_learning_stage6_policy_extensions/selected_rule_specs.csv`
- `outputs/hank_regime_learning_stage6_policy_extensions/component_decomposition.csv`
- `outputs/hank_regime_learning_stage6_policy_extensions/full_hank_projection_metrics.csv`
- `outputs/hank_regime_learning_stage6_policy_extensions/table_stage6_policy_extensions_comparison.tex`
- `outputs/hank_regime_learning_stage6_policy_extensions/table_stage6_policy_extensions_selected_rules.tex`
- `outputs/hank_regime_learning_stage6_policy_extensions/report_stage6_policy_extensions.md`
- `outputs/hank_regime_learning_stage6_policy_extensions/report_full_hank_projection.md`
- `outputs/hank_regime_learning_stage6_policy_extensions/article_strength_check.md`
- `outputs/hank_regime_learning_stage6_reduced_state_validation/state_component_interpretation.csv`
- `outputs/hank_regime_learning_stage6_reduced_state_validation/forecast_sufficiency_summary.csv`
- `outputs/hank_regime_learning_stage6_reduced_state_validation/policy_ranking_validation.csv`
- `outputs/hank_regime_learning_stage6_reduced_state_validation/policy_pairwise_ranking_validation.csv`
- `outputs/hank_regime_learning_stage6_reduced_state_validation/table_state_component_interpretation.tex`
- `outputs/hank_regime_learning_stage6_reduced_state_validation/table_forecast_sufficiency.tex`
- `outputs/hank_regime_learning_stage6_reduced_state_validation/table_policy_ranking_validation.tex`
- `outputs/hank_regime_learning_stage6_reduced_state_validation/reduced_state_validation_text_blocks.tex`
- `outputs/hank_regime_learning_stage6_reduced_state_validation/report_reduced_state_validation.md`
- `outputs/hank_regime_learning_stage6_core_matrix/report_stage6_core_matrix.md`
- `outputs/hank_regime_learning_stage6_summary/stage6_thesis.txt`
- `outputs/hank_regime_learning_stage6_summary/stage6_summary_table.csv`
- `outputs/hank_regime_learning_stage6_summary/table_stage6_summary.tex`
- `outputs/hank_regime_learning_stage6_summary/stage6_text_blocks.tex`
- `outputs/hank_regime_learning_stage6_diagnostics/delta_loss_intervals.csv`
- `outputs/hank_regime_learning_stage6_diagnostics/stage6_metric_interpretation_table.csv`
- `outputs/hank_regime_learning_stage6_diagnostics/table_stage6_metric_interpretation.tex`
- `outputs/hank_regime_learning_stage6_diagnostics/report_stage6_diagnostics.md`

Основные таблицы лежат в:

- `outputs/hank_policy_stage2/tables/`

Основные figures лежат в:

- `outputs/hank_policy_stage2/figures/`

## Структура репозитория

- `hank_full_baseline/`
  Полный HANK pipeline: calibration, steady state, household solver, sequence-space Jacobian, transition dynamics, plots, tables и robustness.
- `hank_partial_info_baseline/`
  Reduced-state partial-observability HANK pipeline: state-space approximation, фильтр, information scenarios, policy diagnostics и article-ready plots/tables. Этот слой трактуется как интерфейс политики, а не как замена полной HANK-экономики.
- `hank_learning_policy_baseline/`
  Learning-based policy layer for partial-information HANK: PPO trainer, residual policy environment, scenario evaluation и article-ready outputs.
- `regime_switching_baseline/`
  Stage-5 regime-switching HANK overlay: скрытые режимы, переключающийся фильтр, classical benchmark и regime diagnostics.
- `hank_regime_learning_baseline/`
  Stage-6 regime-learning layer: обучаемые правила по наблюдениям и по отфильтрованному состоянию, проверки архитектурных ошибок, переноса на новые среды и интерпретационные diagnostics.
- `scripts/run_hank.py`
  One-command запуск полного HANK baseline.
- `scripts/run_hank_partial_info.py`
  One-command запуск partial-information HANK baseline.
- `scripts/run_hank_learning.py`
  One-command запуск stage-4 learning-based policy baseline.
- `scripts/run_hank_regime.py`
  One-command запуск stage-5 regime-switching baseline.
- `scripts/run_hank_regime_learning.py`
  One-command запуск stage-6 regime-learning experiments.
- `scripts/run_hank_regime_policy_extensions.py`
  One-command запуск stage-6 extension checks: оптимизированное линейное правило по оценённому состоянию, историческое правило по наблюдаемым переменным и увеличенный набор test trajectories.
- `scripts/run_hank_regime_full_hank_projection.py`
  Проверка selected stage-6 policy-rate paths через full-HANK transition solver.
- `scripts/run_hank_regime_reduced_state_validation.py`
  Проверка редуцированного состояния как интерфейса политики: экономическая интерпретация компонент, прогнозная достаточность и сохранение ранжирования правил при full-HANK projection.
- `outputs/hank_policy_stage2/`
  Канонический набор результатов для текущей рабочей ветки.
- `outputs/hank_partial_info_stage3/`
  Канонический набор результатов для HANK baseline с неполной информацией.
- `outputs/hank_learning_stage4/`
  Канонический набор результатов для stage 4: learning-based policy layer в полной HANK при неполной информации.
- `outputs/hank_regime_switching_stage5/`
  Канонический набор результатов для stage 5: regime-switching HANK baseline.
- `outputs/hank_regime_learning_stage6_universal_tuning/`
  Канонический набор результатов для stage 6: tuned regime-learning comparison против misspecified classical benchmark.
- `outputs/hank_regime_learning_stage6_policy_extensions/`
  Дополнительный stage-6 validation layer: selected linear rule, history-based observable rule, 50 held-out test trajectories и full-HANK projection для top policy paths.
- `outputs/hank_regime_learning_stage6_reduced_state_validation/`
  Обоснование редуцированного представления состояния: component interpretation, forecast sufficiency и policy ranking validation.

## Архивные ветки

Старые линии проекта специально вынесены из `main`, чтобы не мешать текущей HANK-разработке.

- `archive/hank-legacy`
  Legacy HANK-постановки и переходные partial-information HANK материалы.
- `archive/pre-hank-only-main`
  Предыдущая широкая версия проекта с baseline этапами 1–7, RL-блоками и не-HANK артефактами.

## Текущий фокус

Текущая main-line логика уже собрана как последовательность:
- stage 2: классическая денежно-кредитная политика при полной информации в полной HANK;
- stage 3: неполная наблюдаемость и classical `filter -> rule`, где низкоразмерное состояние является интерфейсом политики;
- stage 4: learning-based policy layer на той же HANK-среде и той же информационной структуре;
- stage 5: hidden regime-switching HANK under partial information;
- stage 6: tuned regime-learning policy и карта условий, где обучаемое правило получает преимущество над жестко заданной или ошибочно специфицированной классической архитектурой.

Следующие расширения должны идти уже от этой рамки: полная HANK как истинная экономика, низкоразмерное представление состояния как интерфейс политики, скрытые режимы и информационные искажения как условия, при которых обучаемая политика может иметь добавочную ценность.
