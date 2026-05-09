# План работы: ценность распределительной информации для денежно-кредитной политики при неполной наблюдаемости в локальной HANK/SSJ-среде

Центральная формула работы:

```text
HANK-модель
+ равновесные отклики метода последовательностей
+ неполная наблюдаемость
+ предельная ценность распределительной информации
```

## 1. HANK-ядро

Цель: получить стационарное состояние, переходные траектории и распределительные статистики.

Текущие артефакты:

- `outputs/hank_core/calibration.json`;
- `outputs/hank_core/steady_state_aggregates.json`;
- `outputs/hank_core/jacobian_summary.csv`;
- `outputs/hank_core/distribution_paths.csv`;
- `outputs/hank_core/tables/table_00_calibration.tex`;
- `outputs/hank_core/tables/table_00_steady_state_moments.tex`.

Команда:

```bash
python3 scripts/run_hank.py --output-dir outputs/hank_core
```

## 2. Якобианы метода последовательностей

Цель: получить равновесные отклики агрегатных и распределительных переменных.

Текущий экспорт:

- `outputs/ssj/jacobians.npz`.

Команда:

```bash
python3 experiments/exp01_ssj_irfs.py --hank-core-dir outputs/hank_core --output-dir outputs/ssj
```

Дополнительно построена локальная библиотека откликов на неполитические шоки:

- `outputs/ssj/stochastic/shock_response_library.csv`;
- `outputs/ssj/stochastic/shock_paths.csv`;
- `outputs/ssj/stochastic/hank_observables.csv`.

Команды:

```bash
python3 experiments/exp06_build_shock_library.py --output-dir outputs/ssj/stochastic
python3 experiments/exp07_generate_stochastic_hank_paths.py --output-dir outputs/ssj/stochastic --num-trajectories 50
```

Следующие расширения:

- якобиан по траектории ставки;
- якобиан по шоку доходного риска;
- разложение ценности отдельных распределительных статистик.

## 3. Наблюдаемые переменные

Цель: построить вектор
\[
q_t=(\pi_t,y_t,C_t,\overline{\kappa}_t,\ell_t,\xi_t^r)
\]
и наблюдения
\[
o_t=Mq_t+\nu_t.
\]

Текущие артефакты:

- `outputs/ssj/hank_observables.csv`;
- `outputs/ssj/hank_observables_spec.json`;
- `outputs/ssj/hank_observations.csv`;
- `outputs/ssj/hank_observations_spec.json`.
- `outputs/ssj/filtered_states.csv`;
- `outputs/ssj/filtered_states_spec.json`;
- `outputs/ssj/filter_quality.csv`.

Команды:

```bash
python3 experiments/exp02_build_hank_observables.py --hank-core-dir outputs/hank_core --output-dir outputs/ssj
python3 experiments/exp03_build_observations.py --observables-csv outputs/ssj/hank_observables.csv --output-dir outputs/ssj
python3 experiments/exp04_filter_states.py --observables-csv outputs/ssj/hank_observables.csv --observations-csv outputs/ssj/hank_observations.csv --observations-spec outputs/ssj/hank_observations_spec.json --output-dir outputs/ssj
```

Информационные состояния:

- \(s_t^{obs-agg}\): агрегатные наблюдения;
- \(s_t^{hist-agg}\): история агрегатных наблюдений;
- \(s_t^{filt-agg}\): оценённые агрегаты;
- \(s_t^{obs-dist}\): наблюдаемые распределительные сигналы;
- \(s_t^{filt-dist}\): оценённые агрегаты и распределительные статистики;
- \(s_t^{full}\): полная информация как ориентир.

Текущий артефакт входов правил:

- `outputs/ssj/information_state_inputs_long.csv`;
- `outputs/ssj/information_state_inputs_spec.json`.

Команда:

```bash
python3 experiments/exp05_build_information_inputs.py --observables-csv outputs/ssj/hank_observables.csv --observations-csv outputs/ssj/hank_observations.csv --filtered-states-csv outputs/ssj/filtered_states.csv --output-dir outputs/ssj
```

Фильтрованные состояния уже включают `filtered_aggregates` и `filtered_distribution`.

Для стохастических траекторий используется тот же слой:

```bash
python3 experiments/exp03_build_observations.py --observables-csv outputs/ssj/stochastic/hank_observables.csv --output-dir outputs/ssj/stochastic --num-seeds 12
python3 experiments/exp04_filter_states.py --observables-csv outputs/ssj/stochastic/hank_observables.csv --observations-csv outputs/ssj/stochastic/hank_observations.csv --observations-spec outputs/ssj/stochastic/hank_observations_spec.json --output-dir outputs/ssj/stochastic
python3 experiments/exp05_build_information_inputs.py --observables-csv outputs/ssj/stochastic/hank_observables.csv --observations-csv outputs/ssj/stochastic/hank_observations.csv --filtered-states-csv outputs/ssj/stochastic/filtered_states.csv --output-dir outputs/ssj/stochastic
```

## 4. Правила политики

Базовое правило:
\[
i_t=\rho_i i_{t-1}+\theta^\top s_t.
\]

Для каждого информационного состояния коэффициенты подбираются отдельно.

Ограничения:

- \(\rho_i\in[0,1)\);
- реакция на инфляцию и выпуск не должна иметь неэкономический знак;
- волатильность ставки ограничивается сверху.

## 5. Главный эксперимент

Главная метрика:
\[
MVOI_{dist}=J(s^{filt-agg})-J(s^{filt-dist}).
\]

Дополнительно:

- \(VOI_{dist}=J(s^{obs-agg})-J(s^{filt-dist})\);
- разрыв до ориентира полной информации;
- доля сокращения измеренного разрыва до ориентира полной информации;
- доверительные интервалы;
- доля выигрышных траекторий.

Текущий основной прогон с совместным фильтром Калмана:

- `outputs/ssj/stochastic/state_space/kalman_filtered_states.csv`;
- `outputs/ssj/stochastic/state_space/filter_quality_joint.csv`;
- `outputs/ssj/stochastic/main_voi_joint_filter/main_voi_summary.csv`;
- `outputs/ssj/stochastic/main_voi_joint_filter/pairwise_value_of_information.csv`;
- `outputs/ssj/stochastic/main_voi_joint_filter/full_information_gap.csv`;
- `outputs/ssj/stochastic/main_voi_joint_filter/report_main_voi.md`.

Команда:

```bash
python3 experiments/exp21_main_voi_joint_filter.py
```

Текущий результат:

- фильтрованные агрегаты: средние потери `0.000433`;
- фильтрованные агрегаты + MPC: средние потери `0.000375`;
- фильтрованные агрегаты + доля низколиквидных домохозяйств: средние потери `0.000372`;
- фильтрованные агрегаты + процентная экспозиция: средние потери `0.000373`;
- все фильтрованные распределительные показатели: средние потери `0.000378`;
- предельная ценность всего распределительного блока: `0.000055`;
- доля выигрышных траекторий для всего блока: `0.587`;
- число тестовых траекторий: `300`.

Интерпретация после технического апгрейда: распределительная информация имеет положительную
предельную ценность сверх фильтрованных агрегатов, но эффект умеренный и должен подтверждаться
идентификационными проверками, а не только таблицей потерь.

## 6. Когда распределительная информация важна

Варьировать:

- шум агрегатных наблюдений;
- шум распределительных наблюдений;
- силу шока доходного риска;
- ликвидность домохозяйств;
- силу распределительного канала.

Этот блок был посчитан для старого скалярного фильтра и требует повторения на совместном фильтре:

- `outputs/ssj/stochastic/noise_sensitivity/noise_sensitivity_summary.csv`;
- `outputs/ssj/stochastic/noise_sensitivity/report_noise_sensitivity.md`.

Команда:

```bash
python3 experiments/exp11_noise_sensitivity.py --scales 0.5,2.0 --num-candidates 90
```

Старые численные выводы не переносим в основной текст без повторного расчёта, потому что новая
основная спецификация использует совместный фильтр Калмана.

## 7. Разложение распределительной информации

Сравнить добавление:

- средней предельной склонности к потреблению;
- доли низколиквидных домохозяйств;
- процентной экспозиции;
- всех статистик вместе.

Текущий результат:

- средняя предельная склонность к потреблению: снижение потерь `0.000058`;
- доля низколиквидных домохозяйств: снижение потерь `0.000061`;
- процентная экспозиция: снижение потерь `0.000061`;
- все распределительные статистики вместе: снижение потерь `0.000055`.

## 8. Идентификационная батарея

Цель: отделить содержательную распределительную информацию от механического увеличения числа
признаков в правиле.

Артефакты:

- `outputs/ssj/stochastic/identification_battery/identification_battery_summary.csv`;
- `outputs/ssj/stochastic/identification_battery/identification_battery_trajectory_losses.csv`;
- `outputs/ssj/stochastic/identification_battery/identification_feature_diagnostics.csv`;
- `outputs/ssj/stochastic/identification_battery/table_identification_battery.tex`;
- `outputs/ssj/stochastic/identification_battery/report_identification_battery.md`.

Команда:

```bash
python3 experiments/exp23_distributional_identification_battery.py
```

Текущий результат:

- фактическая распределительная информация: снижение потерь `0.000038`;
- искусственные признаки с похожей статистикой: `-0.000018`;
- перемешивание между сценариями: около нуля, `0.000003`;
- перемешивание времени внутри сценария: около нуля, `0.000002`;
- запаздывающие признаки: `-0.000027`;
- будущие сдвинутые признаки: `-0.000030`;
- остаточная распределительная информация: `0.000055`.

Главная строка -- остаточная распределительная информация. Она показывает, что эффект не
исчезает после удаления части распределительных признаков, линейно объясняемой фильтрованными
агрегатами.

## 9. Сравнительная статика робастности

Цель: собрать проверки устойчивости не как список таблиц, а как ответы на три экономических
вопроса:

- когда распределительная информация полезна;
- когда она не полезна;
- через какой компонент функции потерь проходит выигрыш.

Артефакты:

- `outputs/ssj/stochastic/robustness_comparative_statics/robustness_comparative_statics.csv`;
- `outputs/ssj/stochastic/robustness_comparative_statics/robustness_questions_summary.csv`;
- `outputs/ssj/stochastic/robustness_comparative_statics/loss_channel_summary.csv`;
- `outputs/ssj/stochastic/robustness_comparative_statics/report_robustness_comparative_statics.md`;
- `article/figures/fig_robustness_comparative_statics.pdf`.

Команда:

```bash
python3 experiments/exp24_robustness_comparative_statics.py
```

Содержательный вывод:

- при росте шума агрегатов MVOI растёт;
- при росте шума распределительных сигналов MVOI падает;
- при нулевом распределительном сигнале эффект исчезает;
- при слабом liquid wedge эффект отрицателен или близок к нулю;
- при сильном liquid wedge эффект становится положительным;
- основной компонент выигрыша -- снижение потерь по разрыву выпуска.

Оговорка: часть sensitivity-прогонов была рассчитана в старой скалярной фильтровой
спецификации. Они используются как направленная сравнительная статика; основная оценка статьи
остаётся joint-filter спецификацией.

Интерпретация после совместного фильтра: отдельные статистики полезны, но весь распределительный
блок вместе не даёт автоматического улучшения.

## 10. Устойчивость к классу правила

Цель: проверить, что ценность распределительной информации не является особенностью одного
базового линейного правила.

Сравниваются:

- правило типа Тейлора;
- Тейлор + потребление;
- Тейлор + распределительные показатели;
- правило с ограничениями на знаки;
- оптимизированное линейное правило;
- Ridge-регуляризация;
- LASSO/ElasticNet;
- малое нелинейное правило как проверка для приложения.

Артефакты:

- `outputs/ssj/stochastic/policy_class_robustness/policy_class_robustness_summary.csv`;
- `outputs/ssj/stochastic/policy_class_robustness/policy_class_robustness_trajectory_losses.csv`;
- `outputs/ssj/stochastic/policy_class_robustness/policy_class_fitted_rules.csv`;
- `outputs/ssj/stochastic/policy_class_robustness/table_policy_class_robustness.tex`;
- `outputs/ssj/stochastic/policy_class_robustness/report_policy_class_robustness.md`;
- `article/figures/fig_policy_class_robustness.pdf`.

Команда:

```bash
python3 experiments/exp25_policy_class_robustness.py
```

Содержательный вывод: распределительная информация снижает потери в Taylor-like правилах,
правиле с ограничениями на знаки, оптимизированном линейном правиле и регуляризованных линейных
правилах. Малое нелинейное правило в текущей реализации не усиливает результат и остаётся только
приложенческой проверкой.

## 11. Информационная граница

Цель: заменить набор отдельных дискретных сравнений общей границей ценности информации.

Уровни:

- 0: текущие агрегаты;
- 1: история агрегатов;
- 2: фильтрованные агрегаты;
- 3: лучший одиночный распределительный сигнал;
- 4: все фильтрованные распределительные сигналы;
- 5: полная информация.

Артефакты:

- `outputs/ssj/stochastic/information_frontier/information_frontier.csv`;
- `outputs/ssj/stochastic/information_frontier/information_frontier_marginal_values.csv`;
- `outputs/ssj/stochastic/information_frontier/information_state_ranking.csv`;
- `outputs/ssj/stochastic/information_frontier/information_state_ranking_stability.csv`;
- `outputs/ssj/stochastic/information_frontier/table_information_state_ranking.tex`;
- `outputs/ssj/stochastic/information_frontier/table_information_marginal_values.tex`;
- `outputs/ssj/stochastic/information_frontier/report_information_frontier.md`;
- `article/figures/fig_information_frontier.pdf`;
- `article/figures/fig_information_marginal_values.pdf`.

Команда:

```bash
python3 experiments/exp26_information_frontier.py
```

Содержательный вывод: основная граница показывает большой выигрыш от фильтрации агрегатов и
дополнительный выигрыш от одного распределительного сигнала. Все распределительные сигналы вместе
не обязаны доминировать лучший одиночный сигнал, поэтому результат трактуется как ценность
конкретной policy-relevant информации, а не как автоматическое преимущество большего числа
переменных.

## 8. Проверки с искусственными статистиками

Проверить:

- случайную статистику с похожей дисперсией и персистентностью;
- перемешанные распределительные ряды;
- калибровку без распределительного канала.

Текущие артефакты:

- `outputs/ssj/stochastic/placebo/placebo_summary.csv`;
- `outputs/ssj/stochastic/placebo/report_placebo_tests.md`.

Команды:

```bash
python3 experiments/exp09_make_placebo_inputs.py --information-inputs outputs/ssj/stochastic/information_state_inputs_long.csv --output-dir outputs/ssj/stochastic/placebo
python3 experiments/exp08_main_voi.py --information-inputs outputs/ssj/stochastic/placebo/information_state_inputs_permuted_distribution_long.csv --hank-observables outputs/ssj/stochastic/hank_observables.csv --jacobians outputs/ssj/jacobians.npz --output-dir outputs/ssj/stochastic/placebo/main_voi_permuted --validation-seeds 900:905 --test-seeds 906:911
python3 experiments/exp08_main_voi.py --information-inputs outputs/ssj/stochastic/placebo/information_state_inputs_fake_distribution_long.csv --hank-observables outputs/ssj/stochastic/hank_observables.csv --jacobians outputs/ssj/jacobians.npz --output-dir outputs/ssj/stochastic/placebo/main_voi_fake --validation-seeds 900:905 --test-seeds 906:911
python3 experiments/exp10_summarize_placebo_tests.py
```

## 9. Фигуры для статьи

Текущие фигуры:

- `article/figures/fig_main_information_states.pdf`;
- `article/figures/fig_distributional_effect_evidence.pdf`;
- `article/figures/fig_artificial_distribution_checks.pdf`;
- `article/figures/fig_distributional_signal_strength.pdf`;
- `article/figures/fig_income_risk_calibration.pdf`;
- `article/figures/fig_loss_component_decomposition.pdf`;
- `article/figures/fig_noise_sensitivity.pdf`.
- `article/figures/fig_additional_robustness.pdf`.

Команда:

```bash
python3 experiments/exp12_build_article_figures.py
```

## 10. Устойчивость к числу HANK/SSJ-траекторий

Цель: проверить, не держится ли основной результат на малом числе стохастических путей.

Текущие артефакты:

- `outputs/ssj/stochastic/trajectory_count_robustness/trajectory_count_robustness_summary.csv`;
- `outputs/ssj/stochastic/trajectory_count_robustness/table_trajectory_count_robustness.tex`;
- `outputs/ssj/stochastic/trajectory_count_robustness/report_trajectory_count_robustness.md`.

Команда:

```bash
python3 experiments/exp17_trajectory_count_robustness.py --path-counts 100 --num-candidates 120
```

Этот блок относится к старому скалярному фильтру. После перехода к совместному фильтру его нужно
повторить, если проверка по числу траекторий остаётся в статье.

## 11. Проверка без распределительного сигнала

Цель: проверить, исчезает ли предельная ценность распределительной информации, если агрегатные
HANK/SSJ-траектории сохранить, а динамику распределительных статистик выключить.

Текущие артефакты:

- `outputs/ssj/stochastic/no_distributional_signal/no_distributional_signal_summary.csv`;
- `outputs/ssj/stochastic/no_distributional_signal/report_no_distributional_signal.md`.

Команда:

```bash
python3 experiments/exp13_no_distributional_signal.py --num-candidates 90
```

Этот блок относится к старому скалярному фильтру. Его нельзя использовать как основной
falsification-тест до повторения на совместном фильтре.

## 12. Чувствительность к силе распределительного сигнала

Цель: проверить, растёт ли предельная ценность распределительной информации при усилении
распределительных отклонений относительно фиксированного масштаба шума измерения.

Текущие артефакты:

- `outputs/ssj/stochastic/distributional_signal_strength/distributional_signal_strength_summary.csv`;
- `outputs/ssj/stochastic/distributional_signal_strength/report_distributional_signal_strength.md`.

Команда:

```bash
python3 experiments/exp14_distributional_signal_strength.py --factors 0,0.5,1,2 --num-candidates 90
```

Этот блок относится к старому скалярному фильтру и должен быть пересчитан, если остаётся в
центральной статье.

## 13. Разложение потерь по компонентам

Цель: показать, через какие компоненты функции потерь проходит выигрыш от распределительной
информации.

Текущие артефакты:

- `outputs/ssj/stochastic/main_voi/loss_component_decomposition.csv`;
- `outputs/ssj/stochastic/main_voi/report_loss_component_decomposition.md`.

Команда:

```bash
python3 experiments/exp15_loss_decomposition.py
```

Текущий результат для предельной ценности распределительной информации:

- инфляция: снижение `0.000011`;
- разрыв выпуска: снижение `0.000098`;
- потребление: снижение `0.000009`;
- сглаживание ставки: почти без изменений.

Интерпретация: основной выигрыш связан со снижением потерь по разрыву выпуска, а не с большей
волатильностью ставки.

## 14. Чувствительность к доходному риску HANK

Цель: проверить, сохраняется ли предельная ценность распределительной информации, если заново
пересчитать HANK steady state и SSJ-отклики при разных значениях доходного риска.

Текущие артефакты:

- `outputs/ssj/income_risk_calibration/income_risk_calibration_summary.csv`;
- `outputs/ssj/income_risk_calibration/report_income_risk_calibration.md`.

Команда:

```bash
python3 experiments/exp16_income_risk_calibration.py --sigma-z-values 0.84,0.88,0.92 --num-trajectories 30 --num-candidates 70
```

Этот блок пока остаётся кандидатом на повторный robustness-прогон с совместным фильтром.

## 15. Шок доходного риска как отдельный источник динамики

Цель: добавить временный шок `sigma_z` прямо в HANK transition solver и проверить, сохраняется ли
главный результат, когда распределительная динамика имеет отдельный HANK-источник.

Текущие артефакты:

- `outputs/ssj/stochastic/income_risk_shock_source/income_risk_shock_source_summary.csv`;
- `outputs/ssj/stochastic/income_risk_shock_source/table_income_risk_shock_source.tex`;
- `outputs/ssj/stochastic/income_risk_shock_source/report_income_risk_shock_source.md`.

Команда:

```bash
python3 experiments/exp18_income_risk_shock_source.py --num-trajectories 50 --num-candidates 90
```

Старые численные значения относятся к скалярному фильтру. Блок нужно повторить на совместном
фильтре, если он остаётся в статье.

## 16. Чувствительность к клину ликвидной доходности

Цель: проверить, зависит ли ценность распределительной информации от силы распределительного
канала в HANK-калибровке.

Текущие артефакты:

- `outputs/ssj/liquid_wedge_channel/liquid_wedge_channel_summary.csv`;
- `outputs/ssj/liquid_wedge_channel/table_liquid_wedge_channel.tex`;
- `outputs/ssj/liquid_wedge_channel/report_liquid_wedge_channel.md`.

Команда:

```bash
python3 experiments/exp19_liquid_wedge_channel_calibration.py --omega-values 0.005,0.01,0.015,0.02 --num-trajectories 30 --num-candidates 70
```

Старые численные значения относятся к скалярному фильтру. Блок нужно повторить на совместном
фильтре, если чувствительность к клину ликвидной доходности остаётся в статье.

## 17. Closed-loop локальная проекция

Цель: проверить, сохраняются ли выводы, если альтернативная ставка меняет не только итоговые
переменные функции потерь, но и контрфактические наблюдения и фильтрованные признаки, на которые
правило реагирует в следующих периодах.

Команда:

```bash
python3 experiments/exp22_closed_loop_evaluation.py
```

Текущие артефакты:

- `outputs/ssj/stochastic/closed_loop/main_voi_closed_loop_summary.csv`;
- `outputs/ssj/stochastic/closed_loop/convergence_diagnostics.csv`;
- `outputs/ssj/stochastic/closed_loop/report_closed_loop.md`.

Текущий вывод: в closed-loop режиме преимущество полного фильтрованного распределительного
состояния над фильтрованными агрегатами становится слабым и статистически неустойчивым. Это
ослабляет старый сильный claim, но делает работу честнее: распределительная информация видна как
локальный информационный сигнал, а не как уже доказанное преимущество полностью согласованной
политики.

Ограничение: в текущем `jacobians.npz` нет прямых SSJ-якобианов для средней MPC, доли
низколиквидных домохозяйств и процентной экспозиции. Для них closed-loop проверка использует
локальную регрессионную проекцию на агрегатные SSJ-эффекты. Следующий сильный технический шаг --
достроить прямые распределительные SSJ-отклики ставки.

## 18. Механизм через локально SSJ-оптимальную ставку

Цель: доказать, что распределительные признаки полезны не только по таблице потерь, а потому что
помогают предсказывать локально SSJ-оптимальную реакцию ставки.

Команда:

```bash
python3 experiments/exp22_mechanism_optimal_rate_projection.py
```

Текущие артефакты:

- `outputs/ssj/stochastic/mechanism_optimal_rate_projection/mechanism_optimal_rate_projection.csv`;
- `outputs/ssj/stochastic/mechanism_optimal_rate_projection/table_mechanism_optimal_rate_projection.tex`;
- `outputs/ssj/stochastic/mechanism_optimal_rate_projection/mechanism_residual_projection.csv`;
- `outputs/ssj/stochastic/mechanism_optimal_rate_projection/mechanism_transmission_projection.csv`;
- `article/figures/fig_mechanism_distribution_transmission.pdf`;
- `article/figures/fig_mechanism_event_study.pdf`.

Текущий результат:

- фильтрованные агрегаты предсказывают локально оптимальную ставку с RMSE `0.000945` и OOS R2 `0.067`;
- полный распределительный набор снижает RMSE до `0.000924` и повышает OOS R2 до `0.109`;
- после удаления агрегатной части оптимальной ставки остаточные распределительные признаки объясняют около `2.9%` остаточной вариации;
- proxy будущей силы трансмиссии объясняется слабо, но распределительный набор немного улучшает OOS R2: `0.0019` против `0.0015` у агрегатов.

Интерпретация: механизм поддержан в слабой, но содержательной форме. Распределительные признаки
помогают приблизить правило к локально оптимальной ставке; канал через будущую трансмиссию имеет
правильный знак, но небольшой размер.
