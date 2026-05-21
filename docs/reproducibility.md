# Воспроизводимость расчётов

Этот документ описывает основной путь воспроизведения таблиц и рисунков статьи. Старые и
архивные эксперименты сохранены в репозитории, но не являются основной расчётной спецификацией.

## 1. HANK-ядро

```bash
python3 scripts/run_hank.py --output-dir outputs/hank_core
python3 experiments/exp00_hank_core_audit.py --hank-core-dir outputs/hank_core
```

Выходы:

- `outputs/hank_core/`;
- `outputs/hank_core/audit/steady_state_audit.json`;
- `outputs/hank_core/audit/transition_audit.csv`.

## 2. Проверка SSJ-аппроксимации

```bash
python3 experiments/final/01_validate_ssj.py
```

Скрипт сравнивает локальные SSJ-отклики с переходными траекториями HANK-модели в рассматриваемом
диапазоне шоков.

## 3. Локальная модель в пространстве состояний

Основная тестовая выборка строится скриптом:

```bash
python3 experiments/final/03_estimate_information_value.py
```

Внутри этого этапа:

1. по HANK/SSJ-траекториям оценивается матрица перехода \(A\);
2. по остаткам оценивается ковариация инноваций \(\Sigma_\varepsilon\);
3. для каждого информационного режима строится матрица наблюдений \(M_\varphi\);
4. фильтр Калмана формирует агрегатную и распределительную оценки состояния;
5. для каждого режима заново настраивается линейное правило процентной ставки.

Технические идентификаторы `shock_seed`, `observation_seed` и `optimizer_seed` фиксируют,
соответственно, экономическую траекторию, шум наблюдений и случайность оптимизации.

## 4. Основная оценка ценности информации

Главное сравнение строится как

\[
\Delta J_{\mathrm{dist}}
=
J^\star_{\mathrm{agg}}
-
J^\star_{\mathrm{agg+dist}}.
\]

Ключевые выходы:

- `outputs/ssj/stochastic/large_sample/`;
- `outputs/ssj/stochastic/large_sample/test/state_space/state_space_spec.json`;
- `outputs/ssj/stochastic/large_sample/test/information_inputs/`;
- `article/figures/fig_information_regimes.pdf`;
- `article/figures/fig_information_value.pdf`.

## 5. Механизм

```bash
python3 experiments/final/04_mechanism_checks.py
```

Эти расчёты проверяют, помогают ли распределительные статистики прогнозировать локально
желательную ставку и показатель локального отклика экономики на изменение ставки.

## 6. Линейно-квадратичный ориентир

```bash
python3 experiments/final/05_lqg_benchmark.py
```

В тексте статьи этот блок называется линейно-квадратичным ориентиром. Часть старых имён файлов в
выходных папках сохранена для совместимости с уже построенными таблицами.

## 7. Проверка с обратной связью ставки

```bash
python3 experiments/final/06_feedback_rate_check.py
```

Этот расчёт возвращает выбранную ставку в локальную динамику состояния и проверяет, сохраняется
ли знак эффекта при обратной связи инструмента политики.

## 8. Таблицы и рисунки статьи

Основные артефакты статьи лежат в:

- `article/tables/`;
- `article/figures/`;
- `article/main.tex`.

PDF пересобирается из папки `article/`:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```
