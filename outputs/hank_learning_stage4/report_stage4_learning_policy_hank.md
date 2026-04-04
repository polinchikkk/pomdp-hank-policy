# Этап 4. Learning-based policy layer в полной HANK при неполной информации

## Постановка

- Структурная two-asset HANK-среда, reduced-state representation и Kalman filtering block сохранены без изменений относительно этапа 3.
- Меняется только последнее звено: вместо fixed Taylor-type rule используется learning-based policy layer.
- Практически baseline реализован как residual PPO: агент получает filtered policy-relevant state и учит непрерывную поправку к classical filtered rule.
- Это сохраняет честное сравнение `filtering + fixed rule` против `filtering + learned policy` внутри одной и той же информационной среды.

## Обучение

- PPO training seeds: 11, 22.
- Evaluation seeds: 700.
- Горизонт эпизода: 60 периодов.
- Reward совпадает с classical loss: `-(pi^2 + 0.50 * y_gap^2 + 0.05 * Delta i^2)`.

## Основные результаты

- Лучший основной сценарий для RL относительно classical: `Фильтрация: инфляция и ставка`; разница накопленной потери RL минус classical `-8.7891e-06`.
- Наименее благоприятный основной сценарий: `Фильтрация: инфляция, выпуск и ставка`; разница накопленной потери `-3.4913e-06`.
- RL сравнивается главным образом с classical filter-plus-rule; full-information rule используется только как reference layer.

## Seeds и стабильность

- `distribution_augmented_filtered_state`: лучший training seed `22`, лучший validation return `-0.0011`.
- `distribution_augmented_no_distribution_state`: лучший training seed `11`, лучший validation return `-0.0008`.
- `full_macro_filtered_state`: лучший training seed `22`, лучший validation return `-0.0010`.
- `high_noise_filtered_state`: лучший training seed `22`, лучший validation return `-0.0010`.
- `macro_core_filtered_state`: лучший training seed `22`, лучший validation return `-0.0010`.
- `macro_core_filtered_state_uncertainty`: лучший training seed `22`, лучший validation return `-0.0005`.
- `macro_core_raw_observations`: лучший training seed `22`, лучший validation return `-0.0010`.
- `thin_information_filtered_state`: лучший training seed `22`, лучший validation return `-0.0010`.

## Интерпретация

- Этот этап не заменяет structural model и не учит политику на полном распределении агентов напрямую.
- Learning-based layer работает только поверх filtered reduced HANK state и поэтому изолирует именно policy-mapping problem.
- Основной содержательный вопрос здесь: когда гибкая learned reaction function улучшает или не улучшает classical filtered Taylor-type rule при ограниченной информации.