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

Запуск полной HANK-модели:

```bash
python3 scripts/run_hank.py
```

Запуск версии с неполной наблюдаемостью:

```bash
python3 scripts/run_hank_partial_info.py
```

Запуск обучаемого правила поверх версии с неполной наблюдаемостью:

```bash
python3 scripts/run_hank_learning.py
```

Запуск версии со скрытыми режимами:

```bash
python3 scripts/run_hank_regime.py
```

Запуск сравнений правил при скрытых режимах:

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

## Что считает базовая полная модель

- стационарное равновесие полной two-asset HANK-модели;
- распределение домохозяйств по ликвидному и неликвидному богатству;
- функции политики домохозяйств;
- переходную динамику после монетарного шока;
- импульсные отклики агрегатов;
- групповые и распределительные эффекты;
- проверочный блок по MPC, WHtM и параметрам сектора домохозяйств.

## Что считает версия с неполной наблюдаемостью

- низкоразмерное представление скрытого состояния полной HANK-среды как интерфейс политики;
- синтетические траектории скрытого состояния и шумных макроэкономических наблюдений;
- фильтрацию скрытого состояния;
- классическую схему `наблюдения -> фильтрация -> правило`;
- сравнение с правилом при полной информации по функции потерь, отклонению ставки и распределительным показателям.

## Что считает версия с обучаемым правилом

- PPO с непрерывным выбором ставки поверх того же редуцированного состояния и того же фильтра;
- сравнение `классическая схема фильтрации и фиксированного правила` против `фильтрации и обучаемого правила`;
- основные сценарии: `macro_core`, `full_macro`, `thin_information`, `high_noise`, `distribution_augmented`;
- абляции: `оценённое состояние + неопределённость`, `наблюдаемые переменные`, `без распределительных компонент состояния`;
- проверка траекторий ставки, макропеременных и распределительных показателей через переходный решатель полной HANK-модели.

## Что считают расширения со скрытыми режимами

- stage 5: скрытые режимы поверх редуцированного состояния HANK с переключающимся фильтром и классическим правилом;
- stage 6: обучаемые правила по отфильтрованному состоянию и по наблюдениям в HANK-среде со скрытыми режимами;
- основной результат stage 6 для основного текста теперь собран в чистую матрицу монетарных сравнений: `полная информация, фиксированное правило`, `фильтрация, фиксированное правило`, `фильтрация, гибкое правило`, `наблюдаемые переменные, гибкое правило`;
- главный вывод этой матрицы: гибкость правила по отфильтрованному состоянию особенно важна при тонком информационном наборе, тогда как в сценариях с более богатым набором наблюдений различие между правилами по наблюдаемым переменным и по отфильтрованному состоянию невелико.
- отдельный проверочный блок показывает, что редуцированное состояние является содержательно и прогностически значимым представлением для задачи политики: для него собраны таблица экономической интерпретации компонент, проверка качества прогноза и проверка сохранения ранжирования правил через HANK-переход.

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

Для версии с неполной наблюдаемостью:

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

Для stage 4 с обучаемым правилом:

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

Для stage 5 со скрытыми режимами:

- `outputs/hank_regime_switching_stage5/filter_metrics.csv`
- `outputs/hank_regime_switching_stage5/policy_metrics.csv`
- `outputs/hank_regime_switching_stage5/regime_paths.csv`
- `outputs/hank_regime_switching_stage5/report_stage5_regime_switching_hank.md`

Для stage 6 с подбором правил:

- `outputs/hank_regime_learning_stage6_universal_tuning/candidate_summary.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/scenario_results.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/best_candidate_core_map.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/best_candidate_delta_loss_matrix.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/best_vs_baseline_by_scenario.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/best_candidate_seed_win_rates.csv`
- `outputs/hank_regime_learning_stage6_universal_tuning/report_universal_tuning.md`

Для stage 6: основные таблицы и проверочные материалы:

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

Основные рисунки лежат в:

- `outputs/hank_policy_stage2/figures/`

## Структура репозитория

- `hank_full_baseline/`
  Полная HANK-модель: калибровка, стационарное состояние, решение задачи домохозяйств, якобианы, переходная динамика, графики и таблицы.
- `hank_partial_info_baseline/`
  Версия с неполной наблюдаемостью: редуцированное состояние, фильтр, информационные сценарии и сравнение правил. Этот слой трактуется как представление состояния для задачи политики, а не как замена полной HANK-модели.
- `hank_learning_policy_baseline/`
  Блок обучаемого правила для версии с неполной наблюдаемостью: обучение PPO, оценка по сценариям и итоговые таблицы.
- `regime_switching_baseline/`
  Блок со скрытыми режимами: скрытые режимы, переключающийся фильтр, классическое правило и диагностические материалы.
- `hank_regime_learning_baseline/`
  Этап 6: сравнение правил по наблюдаемым переменным и по оценённому состоянию, проверка ошибок спецификации, проверка переноса на новые среды и интерпретационные таблицы.
- `scripts/run_hank.py`
  Однокомандный запуск полной HANK-модели.
- `scripts/run_hank_partial_info.py`
  Однокомандный запуск версии с неполной наблюдаемостью.
- `scripts/run_hank_learning.py`
  Однокомандный запуск этапа 4 с обучаемым правилом.
- `scripts/run_hank_regime.py`
  Однокомандный запуск этапа 5 со скрытыми режимами.
- `scripts/run_hank_regime_learning.py`
  Однокомандный запуск этапа 6 со скрытыми режимами.
- `scripts/run_hank_regime_policy_extensions.py`
  Однокомандный запуск дополнительных проверок этапа 6: оптимизированные линейные правила, правило по наблюдаемым переменным с историей и расширенный набор тестовых траекторий.
- `scripts/run_hank_regime_full_hank_projection.py`
  Проверка выбранных траекторий ставки из этапа 6 через переходный решатель полной HANK-модели.
- `scripts/run_hank_regime_reduced_state_validation.py`
  Проверка редуцированного состояния как представления для задачи политики: экономическая интерпретация компонент, прогнозная достаточность и сохранение ранжирования правил при HANK-переходе.
- `outputs/hank_policy_stage2/`
  Канонический набор результатов для текущей рабочей ветки.
- `outputs/hank_partial_info_stage3/`
  Канонический набор результатов для HANK-модели с неполной наблюдаемостью.
- `outputs/hank_learning_stage4/`
  Канонический набор результатов для этапа 4: обучаемое правило в полной HANK-модели при неполной наблюдаемости.
- `outputs/hank_regime_switching_stage5/`
  Канонический набор результатов для этапа 5: версия со скрытыми режимами.
- `outputs/hank_regime_learning_stage6_universal_tuning/`
  Канонический набор результатов для этапа 6: сравнение настроенных правил со стандартным классическим правилом.
- `outputs/hank_regime_learning_stage6_policy_extensions/`
  Дополнительные проверки этапа 6: настроенные линейные правила, правило по наблюдаемым переменным с историей, 50 независимых тестовых траекторий и проверка выбранных траекторий через HANK-переход.
- `outputs/hank_regime_learning_stage6_reduced_state_validation/`
  Обоснование редуцированного представления состояния: экономическая интерпретация компонент, проверка качества прогноза и проверка сохранения ранжирования правил.

## Архивные ветки

Старые линии проекта специально вынесены из `main`, чтобы не мешать текущей HANK-разработке.

- `archive/hank-legacy`
  Ранние HANK-постановки и переходные материалы по версии с неполной наблюдаемостью.
- `archive/pre-hank-only-main`
  Предыдущая широкая версия проекта с этапами 1–7, блоками обучения и не-HANK артефактами.

## Текущий фокус

Текущая рабочая логика уже собрана как последовательность:
- stage 2: классическая денежно-кредитная политика при полной информации в полной HANK;
- stage 3: неполная наблюдаемость и классическая схема `фильтр -> правило`, где низкоразмерное состояние служит представлением для задачи политики;
- stage 4: обучаемое правило на той же HANK-среде и при той же информационной структуре;
- stage 5: HANK-модель при неполной наблюдаемости со скрытыми режимами;
- stage 6: настроенные правила при скрытых режимах и карта условий, в которых более гибкая схема даёт выигрыш относительно жёстко заданного классического правила.

Следующие расширения должны идти уже от этой рамки: полная HANK как истинная экономика, низкоразмерное представление состояния как рабочее представление для задачи политики, скрытые режимы и информационные искажения как условия, при которых обучаемая политика может иметь добавочную ценность.
