# pomdp-hank-policy

Исследовательский проект о построении экономической политики при неполной наблюдаемости и режимной неопределенности в макроэкономических моделях.

Цель проекта: пройти путь от простого воспроизводимого DSGE baseline к policy-среде с hidden states, а затем перейти к belief-state policy и learning-based подходам.

## Логика проекта

1. **Этап 1. RBC baseline**
   Проверка вычислительной инфраструктуры на простой модели: решение, симуляция, IRF и базовые sanity checks.

2. **Этап 2. Новокейнсианская policy-среда**
   Переход к малой линейной NK-модели с правилом Тейлора, IRF по ключевым шокам и determinacy map.

3. **Этап 3. Hidden states**
   Переписывание модели в форме пространства состояний и восстановление скрытого состояния через фильтр Калмана.

4. **Следующий шаг**
   Переход от hidden-state inference к belief-state policy design и более гибким learning-based схемам.

## Что уже реализовано

### Этап 1

- стандартная RBC-модель с технологическим шоком;
- решение через generalized Schur / QZ decomposition;
- stochastic simulation и IRF;
- sanity checks и residual diagnostics;
- внешняя `gensys`-сверка IRF.

### Этап 2

- малая линейная NK-модель;
- policy block с Taylor rule;
- BK-check и determinacy map по `(\phi_pi, \phi_x)`;
- IRF по demand, cost-push и monetary shocks;
- simulation baseline для policy-relevant variables.

### Этап 3

- linear-Gaussian state-space baseline поверх stage 2;
- скрытое состояние: вектор шоков `r_n`, `u`, `nu`;
- базовые наблюдения: `(x, pi, i)`;
- дополнительный stress test по урезанному набору наблюдений `(pi, i)`;
- явная реализация фильтра Калмана;
- сравнение true vs filtered state по всем латентным компонентам;
- sensitivity к уровню measurement noise и observation design;
- innovation diagnostics и mild parameter misspecification;
- Monte Carlo-сводка по точности и покрытию фильтра.

## Быстрый запуск

Установка зависимостей:

```bash
python3 -m pip install -r requirements.txt
```

Запуск этапов:

```bash
python3 scripts/run_stage1.py
python3 scripts/run_stage2.py
python3 scripts/run_stage3.py
```

По умолчанию результаты сохраняются в:

- `outputs/stage1`
- `outputs/stage2`
- `outputs/stage3`

При первом запуске внешней benchmark-сверки stage 1 проект автоматически подтягивает `dsge==0.1.3` в пользовательский cache, чтобы загрузить внешнюю реализацию `gensys`.

## Ключевые артефакты

### Этап 1

- `outputs/stage1/steady_state.json`
- `outputs/stage1/solution.json`
- `outputs/stage1/irf.csv`
- `outputs/stage1/benchmark_summary.json`
- `outputs/stage1/stage1_report.md`

### Этап 2

- `outputs/stage2/model_spec.json`
- `outputs/stage2/solution.json`
- `outputs/stage2/irf_demand.csv`
- `outputs/stage2/irf_costpush.csv`
- `outputs/stage2/irf_monetary.csv`
- `outputs/stage2/determinacy_map.csv`
- `outputs/stage2/diagnostics_summary.json`
- `outputs/stage2/stage2_report.md`
- `outputs/stage2/figures/irf_demand.png`
- `outputs/stage2/figures/irf_demand_shocks.png`
- `outputs/stage2/figures/irf_costpush.png`
- `outputs/stage2/figures/irf_costpush_shocks.png`
- `outputs/stage2/figures/irf_monetary.png`
- `outputs/stage2/figures/irf_monetary_shocks.png`
- `outputs/stage2/figures/simulated_paths.png`
- `outputs/stage2/figures/simulated_shocks.png`
- `outputs/stage2/figures/determinacy_map.png`

The concise human-written note for stage 1 is in `docs/stage1_note.md`. The generated stage-2 report is `outputs/stage2/stage2_report.md`.
>>>>>>> 2d04508 (nk baseline)

### Импульсные отклики

<<<<<<< HEAD
![IRF для RBC baseline](outputs/stage1/figures/irf.png)

После положительного технологического шока выпуск, потребление, инвестиции и труд увеличиваются на ударе, а капитал накапливается постепенно. Такая форма откликов соответствует стандартной экономической логике RBC-модели.

### Стохастическая симуляция

![Стохастическая симуляция RBC baseline](outputs/stage1/figures/simulated_paths.png)

Симулированные траектории подтверждают устойчивость baseline-решения и согласуются с локальной аппроксимацией в окрестности стационарного состояния.

---

Результаты этапа сохраняются в `outputs/stage1/`:

- `steady_state.json` — параметры модели и стационарное состояние;
- `solution.json` — матрицы линейной политики и перехода;
- `simulated_paths.csv` — стохастическая симуляция;
- `irf.csv` — импульсные отклики;
- `diagnostics.csv` — диагностические остатки;
- `diagnostics_summary.json` — краткая сводка по проверкам;
- `stage1_report.md` — текстовый отчёт по этапу 1;
- `figures/` — графики.
=======
- `rbc_baseline/model.py`: model equations, steady state, observable reconstruction.
- `rbc_baseline/solver.py`: QZ/generalized-Schur solver for the linear policy system with Blanchard-Kahn checks.
- `rbc_baseline/benchmark.py`: external `gensys` benchmark loader and IRF comparison utilities.
- `rbc_baseline/pipeline.py`: simulation, IRF, diagnostics, plots, and artifact export.
- `nk_baseline/model.py`: small linear NK policy model specification.
- `nk_baseline/solver.py`: QZ/generalized-Schur NK solver and determinacy diagnostics.
- `nk_baseline/pipeline.py`: stage-2 IRF, simulation, determinacy-map, and report pipeline.
- `scripts/run_stage1.py`: one-command entry point for the full baseline run.
- `scripts/run_stage2.py`: one-command entry point for the stage-2 NK baseline.
- `docs/stage1_note.md`: short technical note for the baseline stage.

## Transition To Stage 3

The next stage should add hidden states and partial observability on top of the stage-2 NK policy environment, and only after that move to rule-based versus learning-based policy comparison.
>>>>>>> 2d04508 (nk baseline)
