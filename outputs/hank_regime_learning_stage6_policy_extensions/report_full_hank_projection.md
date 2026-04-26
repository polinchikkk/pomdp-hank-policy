# Full-HANK projection для правил этапа 6

Средние тестовые траектории ставки из reduced-state экспериментов передаются в полную HANK как траектории monetary-policy shock. Это не является полной оптимизацией правила в full HANK; это проверка согласованности направления результатов при пропуске выбранных траекторий через full-HANK transition solver.

Если full-scale траектория не сходится в nonlinear solver, используется последовательное уменьшение амплитуды. Поэтому поле `scale_used` важно для интерпретации: значение ниже единицы означает, что full-HANK solver принимает только локальную версию соответствующей policy path.

## Результаты

- Базовый макроэкономический набор × умеренная разделимость режимов, `classical_filtered_rule`: scale `0.25`, full-HANK cumulative loss `1.2480e-04`, peak inflation `2.0308e-03`, peak output gap `1.3647e-03`.
- Базовый макроэкономический набор × умеренная разделимость режимов, `optimized_linear_estimated_state`: scale `1.00`, full-HANK cumulative loss `1.0967e-05`, peak inflation `6.8790e-04`, peak output gap `3.6284e-04`.
- Базовый макроэкономический набор × умеренная разделимость режимов, `history_observables_rule`: scale `1.00`, full-HANK cumulative loss `1.2668e-06`, peak inflation `2.3074e-04`, peak output gap `1.0844e-04`.
- Базовый макроэкономический набор × высокая разделимость режимов, `classical_filtered_rule`: scale `0.25`, full-HANK cumulative loss `1.8733e-04`, peak inflation `2.2534e-03`, peak output gap `1.6593e-03`.
- Базовый макроэкономический набор × высокая разделимость режимов, `optimized_linear_estimated_state`: scale `1.00`, full-HANK cumulative loss `1.4531e-05`, peak inflation `6.3158e-04`, peak output gap `4.1220e-04`.
- Базовый макроэкономический набор × высокая разделимость режимов, `history_observables_rule`: scale `1.00`, full-HANK cumulative loss `1.9242e-06`, peak inflation `2.5236e-04`, peak output gap `1.5080e-04`.
- Ограниченный информационный набор × умеренная разделимость режимов, `classical_filtered_rule`: scale `0.25`, full-HANK cumulative loss `1.3458e-04`, peak inflation `2.1251e-03`, peak output gap `1.5265e-03`.
- Ограниченный информационный набор × умеренная разделимость режимов, `optimized_linear_estimated_state`: scale `1.00`, full-HANK cumulative loss `9.9350e-06`, peak inflation `6.5770e-04`, peak output gap `3.5427e-04`.
- Ограниченный информационный набор × умеренная разделимость режимов, `history_observables_rule`: scale `1.00`, full-HANK cumulative loss `1.2668e-06`, peak inflation `2.3074e-04`, peak output gap `1.0844e-04`.
- Ограниченный информационный набор × высокая разделимость режимов, `classical_filtered_rule`: scale `0.25`, full-HANK cumulative loss `1.7498e-04`, peak inflation `2.1725e-03`, peak output gap `1.6190e-03`.
- Ограниченный информационный набор × высокая разделимость режимов, `optimized_linear_estimated_state`: scale `1.00`, full-HANK cumulative loss `1.2399e-05`, peak inflation `5.7839e-04`, peak output gap `3.7862e-04`.
- Ограниченный информационный набор × высокая разделимость режимов, `history_observables_rule`: scale `1.00`, full-HANK cumulative loss `1.9242e-06`, peak inflation `2.5236e-04`, peak output gap `1.5080e-04`.