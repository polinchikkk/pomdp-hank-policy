# pomdp-hank-policy

Исследовательский проект о построении экономической политики при неполной наблюдаемости и режимной неопределенности в макроэкономических моделях.  
Цель проекта: перейти от классического пайплайна к более гибкой постановке, в которой политика строится на основе belief state и в дальнейшем может быть реализована с использованием learning-based подходов.

## Логика проекта

<<<<<<< HEAD
1. **Baseline на простой модели**  
   Проверка вычислений на стандартной RBC-модели.

2. **Переход к новокейнсианской среде**  
   Добавление policy block и постановки задачи экономической политики.

3. **Скрытые состояния и неполная наблюдаемость**  
   Представление модели в форме пространства состояний и восстановление скрытых компонент.

4. **Режимная неопределенность**  
   Переход к средам со скрытыми режимами и структурными сдвигами.
=======
The repository now contains two completed baseline stages.

Stage 1:

- RBC sanity-check environment;
- QZ/generalized-Schur solution;
- simulation and IRF;
- external `gensys` IRF cross-check.

Stage 2:

- small linear New Keynesian policy model;
- Taylor-rule monetary policy environment;
- QZ/generalized-Schur solution with BK checks;
- demand, cost-push, and monetary-shock IRF;
- simulated paths and determinacy map over `(\phi_pi, \phi_x)`.

Still not implemented:
>>>>>>> 2d04508 (nk baseline)

5. **Learning-based policy design**  
   Сравнение классической схемы с более гибкими подходами к построению политики.

<<<<<<< HEAD
---

## Baseline на RBC-модели
=======
## Stage 1 Model

The stage-1 baseline is a standard RBC economy with technology shock `z_t`:
>>>>>>> 2d04508 (nk baseline)

На первом этапе реализована стандартная модель реального делового цикла (RBC) с технологическим шоком, описываемым процессом AR(1).

<<<<<<< HEAD
Цель этого этапа — создание воспроизводимого вычислительного baseline, на котором можно проверить:

- корректность задания модели
- вычисление стационарного состояния
- получение локального решения
- стохастическую симуляцию траекторий
- построение импульсных откликов
- базовые диагностические проверки

В текущей реализации:

- стационарное состояние вычисляется в замкнутой форме
- равновесные условия линеаризуются численно
- линейная система рациональных ожиданий решается методом обобщённого разложения Шура (generalized Schur / QZ decomposition)
- дополнительно проверяется выполнение условия Бланшара–Кана
=======
## Stage 2 Model

The stage-2 baseline is a small linear New Keynesian policy model:

- IS curve:
  `x_t = E_t[x_(t+1)] - (1 / sigma) * (i_t - E_t[pi_(t+1)] - r_t^n)`
- New Keynesian Phillips curve:
  `pi_t = beta * E_t[pi_(t+1)] + kappa * x_t + u_t`
- Taylor rule:
  `i_t = phi_pi * pi_t + phi_x * x_t + nu_t`
- Shock laws:
  `r_(t+1)^n = rho_r * r_t^n + sigma_r * eps_(t+1)^r`
  `u_(t+1) = rho_u * u_t + sigma_u * eps_(t+1)^u`
  `nu_(t+1) = rho_nu * nu_t + sigma_nu * eps_(t+1)^nu`
>>>>>>> 2d04508 (nk baseline)

### Что получено на этапе 1

<<<<<<< HEAD
В результате реализации baseline получены:

- стационарное состояние модели;
- матрицы линейной политики и перехода;
- стохастическая симуляция траекторий;
- импульсные отклики на положительный технологический шок;
- диагностические остатки и базовые sanity checks.

### Основные проверки
=======
Install dependencies and run either baseline:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_stage1.py
python3 scripts/run_stage2.py
```

By default these write outputs into `outputs/stage1` and `outputs/stage2`.
The first benchmark run also downloads `dsge==0.1.3` into the user cache to load an external `gensys` implementation for IRF comparison.
>>>>>>> 2d04508 (nk baseline)

Для baseline были проведены следующие проверки:

<<<<<<< HEAD
- устойчивость линейного решения;
- выполнение условия Бланшара–Кана;
- корректность знаков импульсных откликов;
- малость остатков равновесных уравнений на стохастической симуляции;
- устойчивость результатов к смене `seed`.
=======
Stage 1 outputs:
>>>>>>> 2d04508 (nk baseline)

---

<<<<<<< HEAD
## Визуализация baseline
=======
Stage 2 outputs:

- `outputs/stage2/model_spec.json`
- `outputs/stage2/solution.json`
- `outputs/stage2/irf_demand.csv`
- `outputs/stage2/irf_costpush.csv`
- `outputs/stage2/irf_monetary.csv`
- `outputs/stage2/simulated_paths.csv`
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
