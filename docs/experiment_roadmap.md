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

Финальный режим оптимизации правил задаётся единым конфигом:

- `config/final_policy_optimization.yaml`;
- случайные кандидаты используются только как стартовые точки;
- supervised initialization строится по локально SSJ-оптимальной ставке;
- затем для каждого информационного состояния применяется одинаковый multi-start continuous
  budget;
- ridge-варианты оцениваются как регуляризованные спецификации;
- sign/bounds constraints фиксируются в конфиге и одинаковы для всех состояний.

Основная команда `exp21_main_voi_joint_filter.py` по умолчанию передаёт этот конфиг в
`exp08_main_voi.py`. Для старого быстрого режима можно явно передать пустой конфиг через прямой
запуск `exp08_main_voi.py`, но такой режим не считается финальной спецификацией статьи.

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

## 12. Аудит численного шума оптимизации

Цель: проверить, что основной эффект \(MVOI_{dist}\) больше шума, возникающего из-за seed и
настроек оптимизатора.

Артефакты:

- `outputs/ssj/stochastic/optimizer_noise_audit_key_pair/optimizer_noise_runs.csv`;
- `outputs/ssj/stochastic/optimizer_noise_audit_key_pair/optimizer_noise_coefficients.csv`;
- `outputs/ssj/stochastic/optimizer_noise_audit_key_pair/optimizer_noise_mvoi_distribution.csv`;
- `outputs/ssj/stochastic/optimizer_noise_audit_key_pair/optimizer_noise_state_summary.csv`;
- `outputs/ssj/stochastic/optimizer_noise_audit_key_pair/optimizer_noise_mvoi_summary.csv`;
- `outputs/ssj/stochastic/optimizer_noise_audit_key_pair/report_optimizer_noise_audit.md`;
- `article/figures/fig_optimizer_noise_mvoi.pdf`.

Команда для ключевой пары:

```bash
python3 experiments/exp28_optimizer_noise_audit.py \
  --optimizer-seeds 1000:1019 \
  --num-starts-list 1 \
  --maxiter-list 12,50 \
  --continuous-methods L-BFGS-B \
  --information-states filtered_aggregates,filtered_distribution \
  --output-dir outputs/ssj/stochastic/optimizer_noise_audit_key_pair
```

Команда для полной ночной сетки:

```bash
python3 experiments/exp28_optimizer_noise_audit.py \
  --optimizer-seeds 1000:1049 \
  --num-starts-list 1,5,20,50 \
  --maxiter-list 12,50,200 \
  --continuous-methods L-BFGS-B,Powell,Nelder-Mead
```

Текущий короткий результат: для ключевой пары MVOI остаётся положительным во всех запусках; при
исходном режиме \(1\) старт и \(12\) итераций средний MVOI равен `0.000055`, а разброс по seed
оптимизатора ниже численной точности. При \(50\) итерациях MVOI возрастает до `0.000073`.

## 13. Large-sample split и кластерная статистика

Цель: развести три источника случайности:

- `shock_seed` -- независимая HANK/SSJ-траектория;
- `observation_seed` -- шум наблюдений;
- `optimizer_seed` -- случайность подбора правила.

Артефакты:

- `outputs/ssj/stochastic/large_sample/main_voi_summary.csv`;
- `outputs/ssj/stochastic/large_sample/main_voi_by_shock_cluster.csv`;
- `outputs/ssj/stochastic/large_sample/pairwise_value_of_information.csv`;
- `outputs/ssj/stochastic/large_sample/pairwise_value_of_information_by_shock_cluster.csv`;
- `outputs/ssj/stochastic/large_sample/clustered_inference.csv`;
- `outputs/ssj/stochastic/large_sample/trajectory_losses.csv`;
- `outputs/ssj/stochastic/large_sample/large_sample_spec.json`;
- `outputs/ssj/stochastic/large_sample/report_large_sample.md`.

Команда текущей увеличенной проверки:

```bash
python3 experiments/exp29_large_sample_joint_filter.py \
  --train-shock-seeds 0:49 \
  --validation-shock-seeds 200:219 \
  --test-shock-seeds 400:499 \
  --observation-seeds-validation 930:934 \
  --observation-seeds-test 960:969 \
  --num-candidates 120 \
  --maxiter 12 \
  --cluster-bootstrap-reps 1000 \
  --output-dir outputs/ssj/stochastic/large_sample
```

Текущий результат: фильтрация агрегатов устойчиво снижает потери относительно текущих агрегатов.
Полный распределительный блок в large-sample проверке почти не улучшает фильтрованные агрегаты:
кластерное снижение потерь равно примерно `0.000001`, а кластерный интервал включает ноль.
Отдельные распределительные статистики сохраняют положительный, но статистически слабый знак.
Это ослабляет старый сильный claim и делает финальный вывод осторожнее.

Статистическая проверка теперь использует несколько способов построения вывода:

- обычный парный bootstrap по тестовым наблюдениям;
- кластерный bootstrap по `shock_seed`;
- wild-bootstrap по `shock_seed`;
- перестановочную проверку по парным наблюдениям;
- sign-flip проверку по средним разностям внутри HANK/SSJ-кластеров;
- поправку Бенджамини--Хохберга для набора попарных p-value.

Итог: выигрыш фильтрованных агрегатов проходит все проверки, а предельная ценность полного
распределительного блока не проходит кластерную и sign-flip проверку.

## 14. LQG / Riccati-ориентир

Цель: построить верхний ориентир для совместной линейной state-space задачи и проверить, насколько
простые правила далеки от оптимального линейного регулятора.

Сравниваются:

- простое правило на фильтрованных агрегатах;
- простое правило на фильтрованных распределительных показателях;
- LQG с агрегатными наблюдениями;
- LQG с агрегатными и распределительными наблюдениями;
- LQR с полной информацией.

Артефакты:

- `outputs/ssj/stochastic/lqg_oracle/lqg_oracle_summary.csv`;
- `outputs/ssj/stochastic/lqg_oracle/lqg_oracle_pairwise.csv`;
- `outputs/ssj/stochastic/lqg_oracle/lqg_oracle_gains.csv`;
- `outputs/ssj/stochastic/lqg_oracle/report_lqg_oracle.md`;
- `article/figures/fig_lqg_oracle_comparison.pdf`.

Команда:

```bash
python3 experiments/exp36_lqg_information_oracle.py
```

Текущий результат:

- простое правило на фильтрованных агрегатах: средние потери `0.000159`;
- простое правило на распределительной информации: `0.000160`;
- LQG на агрегатных наблюдениях: `0.000111`;
- LQG на агрегатных и распределительных наблюдениях: `0.000107`;
- LQR с полной информацией: `0.000083`;
- предельная ценность распределительных наблюдений внутри LQG: `0.0000036`;
- кластерный интервал для разности `LQG dist - LQG agg`: `[-0.000004, -0.000003]`.

Интерпретация: распределительные наблюдения имеют положительную, но небольшую ценность даже для
оптимального линейного регулятора. При этом LQG заметно улучшает ограниченные простые правила, а
полная информация остаётся лучшим ориентиром. Значит, Riccati-блок усиливает вывод, но одновременно
показывает его границы: распределительная информация не закрывает весь информационный разрыв.

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
python3 experiments/exp30_closed_loop_distributional_ssj.py
```

Текущие артефакты:

- `outputs/ssj/stochastic/closed_loop_distributional_ssj/jacobians_distributional_augmented.npz`;
- `outputs/ssj/stochastic/closed_loop_distributional_ssj/distributional_policy_jacobians_long.csv`;
- `outputs/ssj/stochastic/closed_loop_distributional_ssj/direct_distributional_jacobian_diagnostics.csv`;
- `outputs/ssj/stochastic/closed_loop_distributional_ssj/main_voi_closed_loop_summary.csv`;
- `outputs/ssj/stochastic/closed_loop_distributional_ssj/pairwise_closed_loop_value_of_information.csv`;
- `outputs/ssj/stochastic/closed_loop_distributional_ssj/convergence_diagnostics.csv`;
- `outputs/ssj/stochastic/closed_loop_distributional_ssj/report_closed_loop.md`.

Текущий вывод: после добавления прямых распределительных HANK/SSJ-откликов на траекторию ставки
closed-loop проверка становится существенно сильнее. Фильтрованные распределительные показатели
дают средние потери `0.000846` против `0.002351` у фильтрованных агрегатов; снижение потерь равно
`0.001504`, кластерный интервал для разности отрицателен, sign-flip p-value равен `0.00025`.
Итерации сходятся на всех 300 парных траекториях.

Ограничение: для одного дальнего периода распределительного якобиана HANK transition solver не
сошёлся, поэтому этот столбец заменён Toeplitz-сдвигом нулевого HANK-отклика. Это нужно явно
оставлять в тексте как ограничение локальной closed-loop проверки.

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

### 18.1. Cross-fit residual mechanism

Цель: проверить, не сводится ли механизм к тому, что распределительные признаки просто дублируют
фильтрованные агрегаты.

Команда:

```bash
python3 experiments/exp35_mechanism_residualized_crossfit.py
```

Схема:

- out-of-fold прогноз локально оптимальной ставки по фильтрованным агрегатам;
- out-of-fold прогноз каждого распределительного признака по фильтрованным агрегатам;
- регрессия остатка оптимальной ставки на остаточные распределительные признаки;
- проверка устойчивости коэффициентов по фолдам.

Текущие артефакты:

- `outputs/ssj/stochastic/mechanism_residualized_crossfit/residualized_crossfit_summary.csv`;
- `outputs/ssj/stochastic/mechanism_residualized_crossfit/coefficient_stability_by_fold.csv`;
- `outputs/ssj/stochastic/mechanism_residualized_crossfit/residualized_feature_tests.csv`;
- `outputs/ssj/stochastic/mechanism_residualized_crossfit/report_mechanism_residualized_crossfit.md`;
- `article/figures/fig_mechanism_residualized_crossfit.pdf`.

Текущий результат: residual/partial R2 равен `0.0462`, cross-fit p-value равен `0.0002`, а
коэффициенты по всем шести фолдам сохраняют один знак. Это сильнее обычной регрессии механизма:
распределительные остатки объясняют остаток локально оптимальной ставки после удаления агрегатной
части out-of-fold.

## 19. Проверка локальной линейной аппроксимации SSJ

Цель: показать область применимости локальной линейной аппроксимации, на которой держатся
контрфактические расчёты.

Команда:

```bash
python3 experiments/exp31_validate_ssj_jacobians.py
```

Проверяются пять шоков:

- денежный шок;
- шок доходного риска;
- шок клина ликвидной доходности;
- шок агрегатного спроса;
- шок агрегатного предложения.

Для каждого шока сравниваются нелинейный переходный отклик HANK и локальный линейный отклик при
амплитудах `0.25x`, `0.5x`, `1x`, `2x`.

Текущие артефакты:

- `outputs/ssj/jacobian_validation/jacobian_validation_summary.csv`;
- `outputs/ssj/jacobian_validation/jacobian_validation_by_shock.csv`;
- `outputs/ssj/jacobian_validation/jacobian_validation_responses_long.csv`;
- `outputs/ssj/jacobian_validation/report_jacobian_validation.md`;
- `article/figures/fig_jacobian_validation.pdf`.

Текущий результат: экспортированный SSJ-якобиан денежного шока проходит проверку с низкой
средней относительной ошибкой: `0.008` при базовой амплитуде и `0.022` при амплитуде `2x`.
Для неполитических шоков конечная разностная локальная аппроксимация также даёт небольшие ошибки.
Единственное превышение порога `20%` возникает для доли низколиквидных домохозяйств при малом
шоке предложения `0.25x`, но абсолютная ошибка там мала: около `2.9e-05`. Поэтому текущая
интерпретация остаётся локальной: результаты читаются как выводы в окрестности стационарного
состояния, а не как глобальная нелинейная HANK-оценка.

## 20. Negative и positive controls для идентификации

Цель: проверить, что идентификационная батарея не переоценивает распределительные признаки и
одновременно способна восстановить известный распределительный канал, когда он действительно
задан в данных.

Команды:

```bash
python3 experiments/exp32_null_distribution_channel.py
python3 experiments/exp33_known_distribution_channel.py
```

Текущие артефакты:

- `outputs/ssj/stochastic/null_distribution_channel/null_distribution_channel_summary.csv`;
- `outputs/ssj/stochastic/null_distribution_channel/null_distribution_channel_replications.csv`;
- `outputs/ssj/stochastic/null_distribution_channel/report_null_distribution_channel.md`;
- `outputs/ssj/stochastic/known_distribution_channel/known_distribution_channel_summary.csv`;
- `outputs/ssj/stochastic/known_distribution_channel/known_distribution_channel_monotonicity.csv`;
- `outputs/ssj/stochastic/known_distribution_channel/report_known_distribution_channel.md`;
- `article/figures/fig_known_distribution_channel.pdf`.

Negative control: распределительные признаки заменяются шумными рядами с похожей дисперсией,
авторегрессией и корреляцией с агрегатами, но без связи с HANK/SSJ-трансмиссией. По 20 null-повторам
средний MVOI равен `-6.19e-06`, доля положительных MVOI равна `0.10`, а false positive rate по
sign-flip, bootstrap и permutation равен `0`.

Positive control: в истинный разрыв выпуска добавляется известный распределительный канал. MVOI
возрастает с `2.58e-05` при `gamma=0` до `5.81e-05` при `gamma=0.002`; с допуском `1e-06` рост
монотонный. Начиная с `gamma=0.00025`, sign-flip p-value становится ниже `5%`. Это показывает, что
пайплайн умеет отличать заданный распределительный сигнал от шума.

## 21. Разложение по распределительным признакам

Цель: показать, какой именно распределительный признак несёт ценность для правила ставки и через
какой компонент функции потерь проходит выигрыш.

Команда:

```bash
python3 experiments/exp34_distributional_feature_decomposition.py
```

Методы:

- one-feature-at-a-time;
- leave-one-feature-out;
- Shapley по всем коалициям распределительных признаков;
- residualized Shapley после очистки распределительных признаков от фильтрованных агрегатов.

Текущие артефакты:

- `outputs/ssj/stochastic/feature_decomposition/feature_mvoi.csv`;
- `outputs/ssj/stochastic/feature_decomposition/coalition_mvoi.csv`;
- `outputs/ssj/stochastic/feature_decomposition/feature_shapley_values.csv`;
- `outputs/ssj/stochastic/feature_decomposition/feature_loss_component_decomposition.csv`;
- `outputs/ssj/stochastic/feature_decomposition/report_feature_decomposition.md`;
- `article/figures/fig_distributional_feature_decomposition.pdf`.

Текущий результат:

- one-feature-at-a-time: самый сильный одиночный сигнал -- доля низколиквидных домохозяйств (`0.000034`), затем процентная экспозиция (`0.000031`) и MPC (`0.000022`);
- Shapley: основной маржинальный вклад получает MPC (`0.000030`, около `79%`), потому что признаки перекрываются и дают неаддитивные взаимодействия;
- residualized Shapley: MPC остаётся главным, но вклад более сбалансирован: MPC `61%`, доля низколиквидных `21%`, процентная экспозиция `18%`;
- по компонентам функции потерь полный распределительный блок работает прежде всего через разрыв выпуска: `0.000033` из общего снижения `0.000038`, то есть около `87%`.
