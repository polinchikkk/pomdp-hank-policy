# Ценность распределительной информации для денежно-кредитной политики при неполной наблюдаемости в локальной HANK/SSJ-среде

Репозиторий посвящён локальной задаче выбора правила процентной ставки в HANK/SSJ-среде при
неполной наблюдаемости.

Центральный вопрос:

> в каких условиях распределительная информация о домохозяйствах имеет самостоятельную ценность для выбора правила процентной ставки?

Основная линия работы:

```text
HANK-модель
→ равновесные отклики метода последовательностей
→ шумные наблюдения
→ информационные состояния центрального банка
→ ценность информации для политики
```

Главная метрика:

\[
MVOI_{dist}=J(s^{filt-agg})-J(s^{filt-dist}).
\]

Она измеряет предельную ценность распределительной информации сверх оценённого агрегатного
состояния.

Работа не решает глобальную нелинейную задачу оптимальной политики в HANK-модели. Она оценивает
информационную ценность распределительных статистик для простого правила ставки вокруг
стационарного состояния, отделяя эту ценность от фильтрации агрегатов, механического расширения
числа входов правила и ориентира полной информации.

## Структура

- `hank/` — HANK-ядро, стационарное состояние, переходные траектории и распределительные статистики.
- `hank_ssj/` — слой для HANK/SSJ-артефактов, наблюдаемых переменных, совместной фильтрации и информационных состояний.
- `policy/` — интерпретируемые правила ставки и парные сравнения потерь.
- `experiments/` — эксперименты новой HANK/SSJ-линии.
- `docs/` — постановка, дорожная карта и план проверки модели.
- `article/` — черновик статьи.

## Уже полученные HANK/SSJ-артефакты

Команда

```bash
python3 scripts/run_hank.py --output-dir outputs/hank_core
python3 experiments/exp00_hank_core_audit.py --hank-core-dir outputs/hank_core
```

создаёт HANK-выходы: стационарные агрегаты, распределительные статистики, переходные траектории и
`jacobian_summary.csv`. Аудит HANK-ядра проверяет массу и неотрицательность распределения,
borrowing-constraint shares, policy/Euler proxy residuals, market clearing, steady-state aggregates
и transition solver residuals по shock types. Результаты сохраняются в
`outputs/hank_core/audit/steady_state_audit.json`, `outputs/hank_core/audit/transition_audit.csv`
и `outputs/hank_core/audit/report_hank_core_audit.md`.

Команда

```bash
python3 experiments/exp01_ssj_irfs.py --hank-core-dir outputs/hank_core --output-dir outputs/ssj
```

экспортирует текущий якобиан в `outputs/ssj/jacobians.npz`.

Проверка локальной линейной аппроксимации HANK/SSJ:

```bash
python3 experiments/exp31_validate_ssj_jacobians.py
```

Она сравнивает нелинейные переходные отклики HANK и локальные линейные отклики для денежного
шока, шока доходного риска, шока клина ликвидной доходности, шока спроса и шока предложения.
Результаты сохраняются в `outputs/ssj/jacobian_validation/`, рисунок -- в
`article/figures/fig_jacobian_validation.pdf`.

Команда

```bash
python3 experiments/exp02_build_hank_observables.py --hank-core-dir outputs/hank_core --output-dir outputs/ssj
```

собирает `outputs/ssj/hank_observables.csv`.

Команда

```bash
python3 experiments/exp03_build_observations.py --observables-csv outputs/ssj/hank_observables.csv --output-dir outputs/ssj
```

строит шумные наблюдения `outputs/ssj/hank_observations.csv`.

Команда

```bash
python3 experiments/exp04_filter_states.py --observables-csv outputs/ssj/hank_observables.csv --observations-csv outputs/ssj/hank_observations.csv --observations-spec outputs/ssj/hank_observations_spec.json --output-dir outputs/ssj
```

строит старый скалярный фильтр. Он оставлен как техническая проверка, но не является основной
спецификацией новой версии.

Основная спецификация строится совместным фильтром Калмана:

```bash
python3 experiments/exp20_joint_kalman_filter.py --observables-csv outputs/ssj/hank_observables.csv --observations-csv outputs/ssj/hank_observations.csv --observations-spec outputs/ssj/hank_observations_spec.json --scalar-filtered-states outputs/ssj/filtered_states.csv --output-dir outputs/ssj/state_space
```

Она сохраняет `state_space_spec.json`, `kalman_filtered_states.csv`,
`filter_quality_joint.csv` и `posterior_covariances.npz`.

Команда

```bash
python3 experiments/exp05_build_information_inputs.py --observables-csv outputs/ssj/hank_observables.csv --observations-csv outputs/ssj/hank_observations.csv --filtered-states-csv outputs/ssj/filtered_states.csv --output-dir outputs/ssj
```

собирает входы информационных состояний `outputs/ssj/information_state_inputs_long.csv`.

Для основного эксперимента используются стохастические HANK/SSJ-траектории с неполитическими шоками:

```bash
python3 experiments/exp06_build_shock_library.py --output-dir outputs/ssj/stochastic
python3 experiments/exp07_generate_stochastic_hank_paths.py --output-dir outputs/ssj/stochastic --num-trajectories 50
python3 experiments/exp03_build_observations.py --observables-csv outputs/ssj/stochastic/hank_observables.csv --output-dir outputs/ssj/stochastic --num-seeds 12
python3 experiments/exp20_joint_kalman_filter.py
python3 experiments/exp21_main_voi_joint_filter.py
```

Главные таблицы основного прогона с совместным фильтром сохраняются в
`outputs/ssj/stochastic/main_voi_joint_filter/`. Старый каталог
`outputs/ssj/stochastic/main_voi/` относится к скалярному фильтру и используется только для
сравнения устойчивости.

Финальный режим подбора правил задаётся единым конфигом
`config/final_policy_optimization.yaml`. В нём случайные кандидаты и supervised initialization
используются только как стартовые точки; итоговый результат выбирается после multi-start
continuous optimization с одинаковым бюджетом для каждого информационного состояния. Таблица
`policy_optimization_budget.csv` в выходной папке фиксирует этот одинаковый бюджет.

Closed-loop проверка с прямыми распределительными SSJ-откликами:

```bash
python3 experiments/exp29_large_sample_joint_filter.py
python3 experiments/exp37_audit_distributional_jacobians.py
python3 experiments/audit_rate_to_shock_inversion.py
python3 experiments/exp30_closed_loop_distributional_ssj.py
python3 experiments/exp36_lqg_information_oracle.py
python3 experiments/exp38_final_voi_protocol.py
```

Она использует уже замороженные правила из финального `large_sample`, итеративно пересчитывает
контрфактическое состояние, наблюдения и фильтрованные признаки. В отличие от старой проверки,
контрфактическая ставка напрямую меняет распределительные статистики через HANK/SSJ-матрицы.
Результаты сохраняются в `outputs/ssj/stochastic/closed_loop_distributional_ssj/`.

В финальном протоколе large-sample open-loop/fixed-path оценка является первичной
оценкой, а closed-loop local projection -- главным credibility check. LQG/Riccati oracle является
обязательным методологическим ориентиром для той же линейной state-space задачи.
`exp38_final_voi_protocol.py` требует closed-loop и LQG-артефакты как обязательные этапы, ставит рядом таблицу A
`open-loop / fixed-path evaluation` и таблицу B `closed-loop local projection`, а для главной пары
`filtered_distribution` против `filtered_aggregates` фиксирует mean delta, cluster CI,
sign-flip p-value, win rate, сходимость, spectral radius локальной петли и stability penalty.
Если closed-loop знак сохраняется, но интервал включает ноль, отчёт пишет
`effect direction survives, precision weaker`; если знак меняется, финальный gate падает и требует
диагностики closed-loop среды.

Проверка механизма через локально SSJ-оптимальную ставку:

```bash
python3 experiments/exp22_mechanism_optimal_rate_projection.py
python3 experiments/exp35_mechanism_residualized_crossfit.py
python3 experiments/exp39_mechanism_dashboard.py
```

Она проверяет, помогают ли распределительные признаки предсказывать локально оптимальную ставку и
будущую силу трансмиссии ставки. Таблица сохраняется в
`outputs/ssj/stochastic/mechanism_optimal_rate_projection/`, а рисунки -- в `article/figures/`.
Cross-fit версия дополнительно удаляет из оптимальной ставки и распределительных признаков ту
часть, которая предсказывается фильтрованными агрегатами out-of-fold. Финальный dashboard
`exp39_mechanism_dashboard.py` делает этот блок отдельным результатом: две цели
`local_optimal_rate_t` и `future_marginal_transmission_strength_t`, cross-fitting по `shock_seed`,
модели A-D, OOF R2 gain, MAE gain, знаки коэффициентов, устойчивость по seed и bins, где
распределительная информация реально помогает. Выходы пишутся в `outputs/final_protocol/`.

Отрицательный и положительный контроль идентификации:

```bash
python3 experiments/exp32_null_distribution_channel.py
python3 experiments/exp33_known_distribution_channel.py
python3 experiments/exp38_identification_dashboard.py
```

Первый строит null-мир, где распределительные признаки имеют похожую статистику, но не связаны с
трансмиссией ставки. Второй добавляет в разрыв выпуска известный распределительный канал и
проверяет, растёт ли оцененный MVOI с силой этого канала. Результаты сохраняются в
`outputs/ssj/stochastic/null_distribution_channel/` и
`outputs/ssj/stochastic/known_distribution_channel/`. Сводная строгая таблица
`гипотеза → тест → результат` сохраняется в `outputs/final_protocol/identification_dashboard.csv`.

Разложение ценности распределительной информации по отдельным признакам:

```bash
python3 experiments/exp34_distributional_feature_decomposition.py
```

Скрипт считает one-feature-at-a-time, leave-one-feature-out, Shapley и residualized Shapley для
MPC, доли низколиквидных домохозяйств и процентной экспозиции. Результаты сохраняются в
`outputs/ssj/stochastic/feature_decomposition/`, рисунок -- в
`article/figures/fig_distributional_feature_decomposition.pdf`.

Сводная сравнительная статика робастности:

```bash
python3 experiments/exp24_robustness_comparative_statics.py
```

Она группирует проверки вокруг трёх вопросов: когда распределительная информация полезна, когда
она не полезна и через какой компонент функции потерь проходит выигрыш. Итоговый рисунок
сохраняется в `article/figures/fig_robustness_comparative_statics.pdf`.

Идентификационная батарея, отделяющая содержательную распределительную информацию от простого
увеличения числа признаков:

```bash
python3 experiments/exp23_distributional_identification_battery.py
python3 experiments/exp38_identification_dashboard.py
```

В этой проверке агрегатный блок в расширенном информационном состоянии фиксируется равным
фильтрованным агрегатам, а меняются только распределительные признаки: фактические, искусственные
с похожей статистикой, перемешанные, сдвинутые во времени и остаточные. Результаты сохраняются в
`outputs/ssj/stochastic/identification_battery/`. Для финального протокола dashboard сжимает
эти строки вместе с negative/positive controls в один файл
`outputs/final_protocol/identification_dashboard.csv`.

Проверка устойчивости к классу правила:

```bash
python3 experiments/exp25_policy_class_robustness.py
```

Она сравнивает правило типа Тейлора, Тейлор с потреблением, Тейлор с распределительными
показателями, правило с ограничениями на знаки, регуляризованные линейные правила и малое
нелинейное правило как дополнительную проверку. Главный вопрос: сохраняется ли предельная ценность
распределительной информации за пределами одного базового линейного правила. Результаты сохраняются
в `outputs/ssj/stochastic/policy_class_robustness/`, рисунок -- в
`article/figures/fig_policy_class_robustness.pdf`.

LQG/Riccati-ориентир для совместной линейной state-space задачи:

```bash
python3 experiments/exp36_lqg_information_oracle.py
```

Этот блок сравнивает простые правила на фильтрованных агрегатах и распределительной информации с
оптимальным LQG-регулятором при агрегатных наблюдениях, LQG-регулятором при агрегатных и
распределительных наблюдениях и LQR-ориентиром полной информации. Результаты сохраняются в
`outputs/ssj/stochastic/lqg_oracle/`, рисунок -- в
`article/figures/fig_lqg_oracle_comparison.pdf`.
В финальном протоколе этот блок не является appendix: `exp38_final_voi_protocol.py` добавляет
таблицу `table_lqg_oracle_benchmark.csv` с пятью контроллерами и диагностику, отвечающую на три
вопроса: есть ли ценность распределительных наблюдений внутри LQG, насколько простое правило далеко
от LQG и не является ли MVOI простого правила артефактом плохой оптимизации.

Информационная граница:

```bash
python3 experiments/exp26_information_frontier.py
```

Этот блок строит шесть уровней доступной информации: текущие агрегаты, история агрегатов,
фильтрованные агрегаты, лучший одиночный распределительный сигнал, все распределительные сигналы
и полная информация. Он также считает предельные выигрыши между соседними уровнями и проверяет
устойчивость рангов при разных seed, числе траекторий, калибровках и способах фильтрации.
Артефакты сохраняются в `outputs/ssj/stochastic/information_frontier/`, рисунки -- в
`article/figures/fig_information_frontier.pdf` и
`article/figures/fig_information_marginal_values.pdf`.

Аудит численного шума оптимизации:

```bash
python3 experiments/exp28_optimizer_noise_audit.py
```

Быстрый ключевой прогон для главной пары:

```bash
python3 experiments/exp28_optimizer_noise_audit.py \
  --optimizer-seeds 1000:1019 \
  --num-starts-list 1 \
  --maxiter-list 12,50 \
  --continuous-methods L-BFGS-B \
  --information-states filtered_aggregates,filtered_distribution \
  --output-dir outputs/ssj/stochastic/optimizer_noise_audit_key_pair
```

Тяжёлая финальная сетка для ночного запуска:

```bash
python3 experiments/exp28_optimizer_noise_audit.py \
  --optimizer-seeds 1000:1049 \
  --num-starts-list 1,5,20,50 \
  --maxiter-list 12,50,200 \
  --continuous-methods L-BFGS-B,Powell,Nelder-Mead
```

Скрипт сохраняет распределение MVOI по seed оптимизатора, разброс тестовых потерь по каждому
информационному состоянию, разброс коэффициентов и частоту выбора лучшего состояния.

Основной large-sample прогон с отдельными shock seed и observation seed:

```bash
python3 experiments/exp29_large_sample_joint_filter.py
```

Эта команда является основным статистическим протоколом: train shock seeds `0:199`,
validation shock seeds `200:399`, test shock seeds `400:899`, validation observation seeds
`930:959`, test observation seeds `960:999`. Она сохраняет результаты в
`outputs/ssj/stochastic/large_sample/`.

Маленький прогон оставлен только как smoke-тест инженерного контура:

```bash
python3 experiments/exp29_large_sample_joint_filter.py --smoke-test
```

Smoke-тест пишет в `outputs/ssj/stochastic/large_sample_smoke/` и не должен использоваться как
таблица финальной статьи.

В этом блоке `shock_seed` задаёт независимую HANK/SSJ-траекторию, `observation_seed` задаёт шум
наблюдений. Основной вывод строится иерархически: `shock_seed` является внешним кластером, а
`observation_seed` -- вложенным измерительным шумом. Итоговая таблица лежит в
`outputs/ssj/stochastic/large_sample/main_voi_summary.csv`, основной вывод -- в
`hierarchical_inference.csv` и `main_inference.csv`; подробные строки по отдельным HANK/SSJ-шокам
вынесены в `main_voi_by_shock_cluster.csv`.

Файл `pairwise_value_of_information.csv` дополнительно содержит несколько проверок для одной и той
же парной разности: обычный bootstrap, кластерный bootstrap, wild-bootstrap, перестановочный
p-value, sign-flip p-value по HANK/SSJ-кластерам, долю выигрышей и долю проигрышей. Эти проверки
остаются диагностикой; основная статистика статьи берётся из `hierarchical_inference.csv`.

Старые проверки с искусственными распределительными статистиками оставлены как дополнительная
техническая проверка:

```bash
python3 experiments/exp09_make_placebo_inputs.py --information-inputs outputs/ssj/stochastic/information_state_inputs_long.csv --output-dir outputs/ssj/stochastic/placebo
python3 experiments/exp08_main_voi.py --information-inputs outputs/ssj/stochastic/placebo/information_state_inputs_permuted_distribution_long.csv --hank-observables outputs/ssj/stochastic/hank_observables.csv --jacobians outputs/ssj/jacobians.npz --output-dir outputs/ssj/stochastic/placebo/main_voi_permuted --validation-seeds 900:905 --test-seeds 906:911
python3 experiments/exp08_main_voi.py --information-inputs outputs/ssj/stochastic/placebo/information_state_inputs_fake_distribution_long.csv --hank-observables outputs/ssj/stochastic/hank_observables.csv --jacobians outputs/ssj/jacobians.npz --output-dir outputs/ssj/stochastic/placebo/main_voi_fake --validation-seeds 900:905 --test-seeds 906:911
python3 experiments/exp10_summarize_placebo_tests.py
```

Чувствительность к шуму наблюдений:

```bash
python3 experiments/exp11_noise_sensitivity.py --scales 0.5,2.0 --num-candidates 90
```

Итоговая таблица сохраняется в `outputs/ssj/stochastic/noise_sensitivity/`.

Проверка без распределительного сигнала:

```bash
python3 experiments/exp13_no_distributional_signal.py --num-candidates 90
```

Она сохраняет агрегатные HANK/SSJ-траектории, но выключает динамику распределительных статистик.

Чувствительность к силе распределительного сигнала:

```bash
python3 experiments/exp14_distributional_signal_strength.py --factors 0,0.5,1,2 --num-candidates 90
```

В этом прогоне масштаб шума измерения фиксируется по исходным HANK/SSJ-траекториям, а амплитуда
распределительных отклонений меняется.

Разложение потерь по компонентам:

```bash
python3 experiments/exp15_loss_decomposition.py
```

Чувствительность к калибровке доходного риска HANK:

```bash
python3 experiments/exp16_income_risk_calibration.py --sigma-z-values 0.84,0.88,0.92 --num-trajectories 30 --num-candidates 70
```

Для каждого значения `sigma_z` заново решается HANK steady state, пересчитываются SSJ-отклики и
строятся HANK/SSJ-траектории.

Проверка устойчивости к числу стохастических HANK/SSJ-траекторий:

```bash
python3 experiments/exp17_trajectory_count_robustness.py --path-counts 100 --num-candidates 120
```

Она повторяет основной расчёт на 100 HANK/SSJ-путях и сохраняет сводку в
`outputs/ssj/stochastic/trajectory_count_robustness/`.

Шок доходного риска как отдельный источник HANK/SSJ-динамики:

```bash
python3 experiments/exp18_income_risk_shock_source.py --num-trajectories 50 --num-candidates 90
```

Этот прогон добавляет временный шок `sigma_z` в HANK transition solver и сохраняет результат в
`outputs/ssj/stochastic/income_risk_shock_source/`.

Чувствительность к клину ликвидной доходности:

```bash
python3 experiments/exp19_liquid_wedge_channel_calibration.py --omega-values 0.005,0.01,0.015,0.02 --num-trajectories 30 --num-candidates 70
```

Для каждого значения `omega` заново решается HANK steady state, пересчитываются SSJ-отклики и
оценивается предельная ценность распределительной информации.

Фигуры для текущего черновика статьи:

```bash
python3 experiments/exp12_build_article_figures.py
```

Скрипт собирает основные рисунки, включая `fig_distributional_effect_evidence.pdf` с парной
статистической проверкой эффекта и `fig_additional_robustness.pdf` с проверками по числу
траекторий, шоку доходного риска и клину ликвидной доходности.

## Ближайшие задачи

1. Уточнить финальные таблицы и подписи рисунков для статьи.
2. Сжать раздел результатов, если статья становится слишком табличной.
