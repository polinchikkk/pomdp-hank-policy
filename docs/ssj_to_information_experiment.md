# Переход к HANK/SSJ-постановке

## Новая центральная формула

Основная линия работы должна быть:

```text
HANK-модель
+ равновесное отображение метода последовательностей
+ неполная наблюдаемость
+ ценность распределительной информации
```

## Что уже есть

- `hank/` содержит двухактивное HANK-ядро на библиотеке `sequence-jacobian`.
- `hank/model.py` задаёт блоки домохозяйств, фирм, номинальных жёсткостей, финансового блока и правила ставки.
- `hank/pipeline.py` строит стационарное состояние, переходные траектории, распределительные статистики и сводные таблицы.
- `hank/sjacobian.py` уже считает якобиан равновесных откликов на шок денежно-кредитной политики.
- `hank_ssj/` добавлен как новый интерфейсный слой для HANK/SSJ-артефактов и будущих информационных экспериментов.
- В текущем рабочем прогоне команда `python3 scripts/run_hank.py --output-dir outputs/hank_core` успешно создаёт HANK-артефакты, включая `jacobian_summary.csv`, стационарные агрегаты, распределительные статистики, таблицы и графики.
- Команда `python3 experiments/exp01_ssj_irfs.py --hank-core-dir outputs/hank_core --output-dir outputs/ssj` экспортирует текущий длинный якобиан в `outputs/ssj/jacobians.npz`.
- Команда `python3 experiments/exp02_build_hank_observables.py --hank-core-dir outputs/hank_core --output-dir outputs/ssj` собирает `outputs/ssj/hank_observables.csv`.
- Команда `python3 experiments/exp03_build_observations.py --observables-csv outputs/ssj/hank_observables.csv --output-dir outputs/ssj` строит шумные наблюдения `outputs/ssj/hank_observations.csv`.
- Команда `python3 experiments/exp04_filter_states.py --observables-csv outputs/ssj/hank_observables.csv --observations-csv outputs/ssj/hank_observations.csv --observations-spec outputs/ssj/hank_observations_spec.json --output-dir outputs/ssj` строит `outputs/ssj/filtered_states.csv`.
- Команда `python3 experiments/exp05_build_information_inputs.py --observables-csv outputs/ssj/hank_observables.csv --observations-csv outputs/ssj/hank_observations.csv --filtered-states-csv outputs/ssj/filtered_states.csv --output-dir outputs/ssj` собирает входы правил в `outputs/ssj/information_state_inputs_long.csv`.
- Команда `python3 experiments/exp06_build_shock_library.py --output-dir outputs/ssj/stochastic` строит HANK-библиотеку откликов на неполитические шоки.
- Команда `python3 experiments/exp07_generate_stochastic_hank_paths.py --output-dir outputs/ssj/stochastic --num-trajectories 30` генерирует стохастические HANK/SSJ-траектории.
- Команда `python3 experiments/exp08_main_voi.py --information-inputs outputs/ssj/stochastic/information_state_inputs_long.csv --hank-observables outputs/ssj/stochastic/hank_observables.csv --jacobians outputs/ssj/jacobians.npz --output-dir outputs/ssj/stochastic/main_voi --validation-seeds 900:905 --test-seeds 906:911` считает первый основной эксперимент по ценности информации.

## Что ещё не закрыто

Текущий `jacobian_summary.csv` из HANK-ядра содержит отклики на `monetary_policy_shock`. Это полезно для локальной проекции альтернативной траектории ставки, но ещё не полностью закрывает новую постановку.

Текущий экспорт в `outputs/ssj/jacobians.npz` поэтому следует трактовать как первый SSJ-артефакт, а не как полный набор якобианов для оптимизации правил. Он содержит матрицы откликов на шок денежно-кредитной политики для агрегированных переменных, но ещё не содержит шока доходного риска.

Для главного эксперимента нужны дополнительные объекты:

- шок доходного риска;
- прямой HANK-расчёт процентной экспозиции;
- проверки с искусственными и перемешанными распределительными статистиками;
- увеличение числа стохастических траекторий.

## Минимальный порядок реализации

1. Запустить HANK-ядро и получить `outputs/hank_core/jacobian_summary.csv`.
2. Экспортировать текущие SSJ-артефакты в `outputs/ssj/jacobians.npz`.
3. Построить библиотеку HANK-откликов на неполитические шоки.
4. Сгенерировать стохастические HANK/SSJ-траектории.
5. Повторить эксперимент по ценности информации для информационных состояний:
   - агрегаты;
   - история агрегатов;
   - оценённые агрегаты;
   - наблюдаемые распределительные сигналы;
   - оценённые агрегаты и распределительные статистики;
   - полная информация.
6. Расширить HANK-ядро, чтобы напрямую считать процентную экспозицию.
7. Добавить шок доходного риска.
8. Добавить проверки с искусственными статистиками:
   - случайная статистика;
   - перемешанные распределительные статистики;
   - калибровка без распределительного канала.
