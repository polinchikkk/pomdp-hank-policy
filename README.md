# pomdp-hank-policy

Текущая рабочая ветка проекта сфокусирована на полной two-asset HANK-модели и классической денежно-кредитной политике в этой среде.

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
- [outputs/hank_policy_stage2](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_policy_stage2)
- [outputs/hank_partial_info_stage3](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_partial_info_stage3)
- [outputs/hank_learning_stage4](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_learning_stage4)
- [outputs/hank_regime_switching_stage5](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_regime_switching_stage5)
- [outputs/hank_regime_learning_stage6_universal_tuning](/Users/polinazosimova/pomdp-hank-policy/outputs/hank_regime_learning_stage6_universal_tuning)

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

По умолчанию результаты сохраняются в:

- `outputs/hank_policy_stage2`
- `outputs/hank_partial_info_stage3`
- `outputs/hank_learning_stage4`
- `outputs/hank_regime_switching_stage5`
- `outputs/hank_regime_learning_stage6_universal_tuning`

## Что считает текущий baseline

- стационарное равновесие полной two-asset HANK-модели;
- распределение домохозяйств по ликвидному и неликвидному богатству;
- функции политики домохозяйств;
- переходную динамику после монетарного шока;
- импульсные отклики агрегатов;
- групповые и распределительные эффекты;
- validation и robustness-блок по MPC, WHtM и household-side calibration.

## Что считает partial-information baseline

- reduced hidden-state представление полной HANK-среды;
- synthetic trajectories скрытого состояния и noisy macro observables;
- Kalman-filter восстановление hidden state;
- classical `filter -> rule` policy поверх оценённого состояния;
- сравнение с full-information HANK benchmark по loss, rate-gap и распределительным метрикам.

## Что считает learning-based policy baseline

- continuous-action residual PPO поверх того же reduced HANK state-space и того же Kalman filter;
- comparison `classical filter + fixed rule` vs `filtering + learned policy`;
- main scenarios: `macro_core`, `full_macro`, `thin_information`, `high_noise`, `distribution_augmented`;
- ablations: `filtered state + uncertainty`, `raw observations`, `without distributional state`;
- policy, macro and distributional evaluation уже на полном HANK transition solver.

## Что считают regime-switching расширения

- stage 5: hidden regime-switching overlay поверх reduced-state HANK с switching filter и classical policy benchmark;
- stage 6: raw-observation и tuned learning-based policy в regime-switching HANK;
- основной stage-6 артефакт на `main` теперь не ранние search-раны, а universal tuning result с лучшим кандидатом `larger_network`.

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

Основные таблицы лежат в:

- `outputs/hank_policy_stage2/tables/`

Основные figures лежат в:

- `outputs/hank_policy_stage2/figures/`

## Структура репозитория

- `hank_full_baseline/`
  Полный HANK pipeline: calibration, steady state, household solver, sequence-space Jacobian, transition dynamics, plots, tables и robustness.
- `hank_partial_info_baseline/`
  Reduced-state partial-observability HANK pipeline: state-space approximation, Kalman filter, information scenarios, policy diagnostics и article-ready plots/tables.
- `hank_learning_policy_baseline/`
  Learning-based policy layer for partial-information HANK: PPO trainer, residual policy environment, scenario evaluation и article-ready outputs.
- `regime_switching_baseline/`
  Stage-5 regime-switching HANK overlay: hidden regimes, switching filter, classical benchmark и regime diagnostics.
- `hank_regime_learning_baseline/`
  Stage-6 regime-learning layer: raw-observation and filtered-state policy experiments, universal tuning and scenario comparison.
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

## Архивные ветки

Старые линии проекта специально вынесены из `main`, чтобы не мешать текущей HANK-разработке.

- `archive/hank-legacy`
  Legacy HANK-постановки и переходные partial-information HANK материалы.
- `archive/pre-hank-only-main`
  Предыдущая широкая версия проекта с baseline этапами 1–7, RL-блоками и не-HANK артефактами.

## Текущий фокус

Текущая main-line логика уже собрана как последовательность:
- stage 2: full-information classical policy в полной HANK;
- stage 3: reduced-state partial observability и classical `filter -> rule`;
- stage 4: learning-based policy layer на той же HANK-среде и той же информационной структуре.
- stage 5: regime-switching HANK under partial information;
- stage 6: tuned regime-learning policy и карта условий, где RL получает преимущество над misspecified classical benchmark.

Следующие расширения должны идти уже от этого канонического stage-4 baseline, а не от старых pre-HANK линий проекта.
