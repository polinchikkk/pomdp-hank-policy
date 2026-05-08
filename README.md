# Ценность распределительной информации в HANK

Репозиторий посвящён денежно-кредитной политике при неполной наблюдаемости в HANK-экономике.

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

Она измеряет предельную ценность распределительной информации сверх оценённого агрегатного состояния.

## Структура

- `hank/` — HANK-ядро, стационарное состояние, переходные траектории и распределительные статистики.
- `hank_ssj/` — слой для HANK/SSJ-артефактов, наблюдаемых переменных и информационных состояний.
- `policy/` — интерпретируемые правила ставки и парные сравнения потерь.
- `experiments/` — эксперименты новой HANK/SSJ-линии.
- `docs/` — постановка, дорожная карта и план проверки модели.
- `article/` — черновик статьи.

## Уже полученные HANK/SSJ-артефакты

Команда

```bash
python3 scripts/run_hank.py --output-dir outputs/hank_core
```

создаёт HANK-выходы: стационарные агрегаты, распределительные статистики, переходные траектории и `jacobian_summary.csv`.

Команда

```bash
python3 experiments/exp01_ssj_irfs.py --hank-core-dir outputs/hank_core --output-dir outputs/ssj
```

экспортирует текущий якобиан в `outputs/ssj/jacobians.npz`.

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

строит фильтрованные состояния `outputs/ssj/filtered_states.csv` и таблицу качества фильтрации `outputs/ssj/filter_quality.csv`.

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
python3 experiments/exp04_filter_states.py --observables-csv outputs/ssj/stochastic/hank_observables.csv --observations-csv outputs/ssj/stochastic/hank_observations.csv --observations-spec outputs/ssj/stochastic/hank_observations_spec.json --output-dir outputs/ssj/stochastic
python3 experiments/exp05_build_information_inputs.py --observables-csv outputs/ssj/stochastic/hank_observables.csv --observations-csv outputs/ssj/stochastic/hank_observations.csv --filtered-states-csv outputs/ssj/stochastic/filtered_states.csv --output-dir outputs/ssj/stochastic
python3 experiments/exp08_main_voi.py --information-inputs outputs/ssj/stochastic/information_state_inputs_long.csv --hank-observables outputs/ssj/stochastic/hank_observables.csv --jacobians outputs/ssj/jacobians.npz --output-dir outputs/ssj/stochastic/main_voi --validation-seeds 900:905 --test-seeds 906:911
```

Главные таблицы первого прогона сохраняются в `outputs/ssj/stochastic/main_voi/`.

Проверки с искусственными распределительными статистиками:

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
