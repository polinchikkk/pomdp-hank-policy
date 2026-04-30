# Карта очистки репозитория

## Архивная ветка

Перед очисткой создана ветка:

```text
archive/stage6-before-information-state-redesign-20260430
```

В ней сохранено состояние проекта до удаления старых материалов stage 6: старые таблицы, графики, проверки PPO, проверки переноса, HANK-проекции и дополнительные сценарии.

## Что осталось в main

В основной ветке оставлена новая линия работы:

- `docs/information_state_design_spec.md`
- `hank_regime_learning_baseline/information_state_design.py`
- `hank_regime_learning_baseline/environment.py`
- `hank_regime_learning_baseline/evaluation.py`
- `hank_regime_learning_baseline/regime_config.py`
- `hank_regime_learning_baseline/scenario_catalog.py`
- `scripts/run_information_state_design.py`
- `scripts/run_information_state_noise_comparison.py`

Также сохранены базовые блоки, на которые опирается новая постановка:

- `hank_full_baseline/`
- `hank_partial_info_baseline/`
- `regime_switching_baseline/`
- `hank_learning_policy_baseline/`

Последний блок оставлен как техническая зависимость и как возможное приложение с нелинейными правилами, но он больше не является центром работы.

## Что удалено из main

Из основной ветки удалены:

- старые выходы `outputs/hank_regime_learning_stage6_*`;
- старые stage 6-скрипты `scripts/run_hank_regime_*`, кроме базового `scripts/run_hank_regime.py`;
- старые модули `hank_regime_learning_baseline` для архитектурных абляций, misspecification map, policy extensions, PPO-проверок, stage 6-summary и reduced-state validation.

Эти материалы не уничтожены: они доступны в архивной ветке.
