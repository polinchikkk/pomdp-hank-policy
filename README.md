# pomdp-hank-policy

Проект сейчас собран вокруг одной темы: **дизайн информационного состояния для денежно-кредитной политики при неполной наблюдаемости**.

Главный вопрос:

> какое низкоразмерное представление информации центрального банка достаточно для хорошего правила процентной ставки в HANK-мотивированной среде со скрытым марковским режимом?

Работа больше не строится вокруг тезиса «обучаемое правило лучше правила Тейлора». Старые сравнения stage 6 вынесены в архивную ветку:

```text
archive/stage6-before-information-state-redesign-20260430
```

## Новая основная линия

Сравниваются четыре способа сжать информацию регулятора:

- короткая история наблюдений;
- апостериорное среднее скрытого состояния;
- апостериорное среднее с режимным расхождением;
- распределительно расширенное информационное состояние.

Правило политики использует только доступные наблюдения или оценки скрытого состояния. Функция потерь считается по истинным значениям инфляционного разрыва, разрыва выпуска и изменения ставки.

## Основные файлы

- [docs/information_state_design_spec.md](/Users/polinazosimova/pomdp-hank-policy/docs/information_state_design_spec.md)
- [docs/repository_cleanup_map.md](/Users/polinazosimova/pomdp-hank-policy/docs/repository_cleanup_map.md)
- [hank_regime_learning_baseline/information_state_design.py](/Users/polinazosimova/pomdp-hank-policy/hank_regime_learning_baseline/information_state_design.py)
- [hank_regime_learning_baseline/environment.py](/Users/polinazosimova/pomdp-hank-policy/hank_regime_learning_baseline/environment.py)
- [hank_regime_learning_baseline/evaluation.py](/Users/polinazosimova/pomdp-hank-policy/hank_regime_learning_baseline/evaluation.py)
- [scripts/run_information_state_design.py](/Users/polinazosimova/pomdp-hank-policy/scripts/run_information_state_design.py)
- [scripts/run_information_state_noise_comparison.py](/Users/polinazosimova/pomdp-hank-policy/scripts/run_information_state_noise_comparison.py)

## Быстрый запуск

Установка зависимостей:

```bash
python3 -m pip install -r requirements.txt
```

Проверочный запуск на одном сценарии:

```bash
python3 scripts/run_information_state_design.py \
  --output-dir outputs/information_state_design_smoke \
  --scenario full_macro_moderate_gap \
  --validation-count 1 \
  --test-count 2 \
  --horizon 10 \
  --max-rounds 1
```

Полный запуск основной матрицы:

```bash
python3 scripts/run_information_state_design.py \
  --output-dir outputs/information_state_design_main
```

Сравнение по уровню шума наблюдений:

```bash
python3 scripts/run_information_state_noise_comparison.py \
  --output-dir outputs/information_state_design_noise_comparison
```

## Структура

- `hank_full_baseline/` — полная HANK-мотивация: домохозяйства, стационарное состояние, переходная динамика.
- `hank_partial_info_baseline/` — базовая постановка с неполной наблюдаемостью.
- `regime_switching_baseline/` — скрытый марковский режим и переключающийся фильтр.
- `hank_regime_learning_baseline/` — новая основная среда для сравнения информационных состояний.
- `scripts/` — воспроизводимые запуски.
- `outputs/` — результаты текущих расчетов.

## Что удалено из main

Из основной ветки убраны старые stage 6-модули, скрипты и таблицы: архитектурные абляции, проверки переноса, старые PPO-сравнения, validation suite и старые итоговые таблицы.

Они не потеряны: вся прежняя рабочая поверхность сохранена в ветке `archive/stage6-before-information-state-redesign-20260430`.
