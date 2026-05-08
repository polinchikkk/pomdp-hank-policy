# План HANK/SSJ-версии

Центральная формула работы:

```text
HANK-модель
+ равновесные отклики метода последовательностей
+ неполная наблюдаемость
+ ценность распределительной информации
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
q_t=(\pi_t,Y_t,C_t,\overline{\kappa}_t,\ell_t,\xi_t^r)
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

Информационные наборы:

- агрегатные наблюдения;
- история агрегатных наблюдений;
- оценённые агрегаты;
- наблюдаемые распределительные сигналы;
- оценённые агрегаты и распределительные статистики;
- полная информация.

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

- \(VOI_{dist}=J(s^{agg})-J(s^{dist})\);
- разрыв до полной информации;
- доля закрытия разрыва;
- доверительные интервалы;
- доля выигрышных траекторий.

Текущий первый прогон:

- `outputs/ssj/stochastic/main_voi/main_voi_summary.csv`;
- `outputs/ssj/stochastic/main_voi/pairwise_value_of_information.csv`;
- `outputs/ssj/stochastic/main_voi/full_information_gap.csv`;
- `outputs/ssj/stochastic/main_voi/report_main_voi.md`.

Команда:

```bash
python3 experiments/exp08_main_voi.py --information-inputs outputs/ssj/stochastic/information_state_inputs_long.csv --hank-observables outputs/ssj/stochastic/hank_observables.csv --jacobians outputs/ssj/jacobians.npz --output-dir outputs/ssj/stochastic/main_voi --validation-seeds 900:905 --test-seeds 906:911
```

Текущий результат:

- фильтрованные агрегаты: средние потери `0.000804`;
- фильтрованные распределительные показатели: средние потери `0.000686`;
- предельная ценность распределительной информации: `0.000117`;
- доля выигрышных траекторий: `0.603`;
- число тестовых траекторий: `300`.

## 6. Когда распределительная информация важна

Варьировать:

- шум агрегатных наблюдений;
- шум распределительных наблюдений;
- силу шока доходного риска;
- ликвидность домохозяйств;
- силу распределительного канала.

Текущий прогон по шуму наблюдений:

- `outputs/ssj/stochastic/noise_sensitivity/noise_sensitivity_summary.csv`;
- `outputs/ssj/stochastic/noise_sensitivity/report_noise_sensitivity.md`.

Команда:

```bash
python3 experiments/exp11_noise_sensitivity.py --scales 0.5,2.0 --num-candidates 90
```

Текущий результат:

- при снижении шума агрегатов до `0.5` предельная ценность распределительной информации равна `0.000045`;
- в базовом случае она равна `0.000117`;
- при росте шума агрегатов до `2.0` она возрастает до `0.000258`;
- при снижении шума распределительных сигналов до `0.5` она равна `0.000154`;
- при росте шума распределительных сигналов до `2.0` она падает до `0.000063`.

Интерпретация: распределительная информация особенно полезна, когда агрегатные наблюдения
становятся менее точными, и теряет часть ценности, когда сами распределительные показатели
измеряются с большим шумом.

## 7. Разложение распределительной информации

Сравнить добавление:

- средней предельной склонности к потреблению;
- доли низколиквидных домохозяйств;
- процентной экспозиции;
- всех статистик вместе.

Текущий результат:

- средняя предельная склонность к потреблению: снижение потерь `0.000062`;
- доля низколиквидных домохозяйств: снижение потерь `0.000031`;
- процентная экспозиция: снижение потерь `0.000042`;
- все распределительные статистики вместе: снижение потерь `0.000117`.

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

Текущий результат:

- при 50 HANK/SSJ-путях предельная ценность распределительной информации равна `0.000117`;
- при 100 HANK/SSJ-путях она равна `0.000063`;
- доверительный интервал для разности потерь в расширенном прогоне остаётся ниже нуля;
- эффект устойчив по знаку, но его размер следует интерпретировать как умеренный.

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

Текущий результат:

- при фактической распределительной динамике \(MVOI_{dist}=0.000117\);
- при выключенном распределительном сигнале \(MVOI_{dist}=0\);
- доля совпадений в нулевом сравнении равна `1.0`.

Интерпретация: выигрыш требует содержательного распределительного сигнала и не возникает просто
от добавления дополнительных входов правила.

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

Текущий результат:

- фактор `0`: \(MVOI_{dist}\approx 0\);
- фактор `0.5`: \(MVOI_{dist}=0.000063\);
- фактор `1`: \(MVOI_{dist}=0.000117\);
- фактор `2`: \(MVOI_{dist}=0.000154\).

Интерпретация: ценность распределительной информации возрастает вместе с информативностью
распределительного сигнала.

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

Текущий результат:

- `sigma_z = 0.84`: \(MVOI_{dist}=0.000055\);
- `sigma_z = 0.88`: \(MVOI_{dist}=0.000064\);
- `sigma_z = 0.92`: \(MVOI_{dist}=0.000045\).

Интерпретация: ценность распределительной информации остаётся положительной около базовой
калибровки, но не растёт механически с одним только доходным риском.

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

Текущий результат:

- без отдельного шока доходного риска \(MVOI_{dist}=0.000117\);
- с шоком доходного риска \(MVOI_{dist}=0.000115\);
- интервал для разности потерь остаётся ниже нуля.

Интерпретация: добавление шока доходного риска не ломает основной результат, но в текущей
калибровке почти не усиливает предельную ценность распределительной информации.

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

Текущий результат:

- при `omega = 0.005` средняя MPC равна `0.056`, а \(MVOI_{dist}=-0.000031\);
- при `omega = 0.010` средняя MPC равна `0.082`, а \(MVOI_{dist}=-0.000016\);
- при `omega = 0.015` средняя MPC равна `0.106`, а \(MVOI_{dist}\) близка к нулю;
- при `omega = 0.020` средняя MPC равна `0.133`, а \(MVOI_{dist}=0.000072\).

Интерпретация: распределительная информация не имеет автоматической ценности. Она становится
полезнее в калибровках, где ликвидный клин и связанная с ним MPC-трансмиссия сильнее.
