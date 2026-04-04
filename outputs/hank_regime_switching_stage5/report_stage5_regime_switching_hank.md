# Этап 5. Regime-switching HANK при неполной информации

## Постановка

- Используется reduced-state HANK layer из этапа 3, но поверх него задаётся скрытое переключение между режимами `normal` и `stress`.
- В `stress` режиме усиливается policy transmission в инфляции, выпуске и распределительных состояниях (`low_liquidity_gap`, `mean_mpc_gap`).
- Регулятор не наблюдает режим напрямую и использует switching Kalman / IMM filter.
- Classical benchmark строится как `switching filter + fixed Taylor-type rule`.

## Сценарии

- `Фильтрация: инфляция, выпуск и ставка × умеренный режимный разрыв`: noisy observables `pi, output_gap`, regime gap `Умеренный режимный разрыв`.
- `Фильтрация: инфляция, выпуск и ставка × сильный режимный разрыв`: noisy observables `pi, output_gap`, regime gap `Сильный режимный разрыв`.
- `Фильтрация: инфляция и ставка × умеренный режимный разрыв`: noisy observables `pi`, regime gap `Умеренный режимный разрыв`.
- `Фильтрация: инфляция и ставка × сильный режимный разрыв`: noisy observables `pi`, regime gap `Сильный режимный разрыв`.

## Качество фильтрации режима

- Наилучшая regime classification accuracy: `Фильтрация: инфляция, выпуск и ставка × сильный режимный разрыв` с accuracy `0.800` и Brier score `1.3169e-01`.
- Наиболее сложный режимный сценарий для фильтра: `Фильтрация: инфляция и ставка × сильный режимный разрыв` с accuracy `0.367`.

## Качество classical policy under regime uncertainty

- Лучший сценарий по разнице накопленной потери относительно полной информации: `Фильтрация: инфляция и ставка × умеренный режимный разрыв` с delta cumulative loss `-4.0451e-03`.
- Наиболее затратный сценарий: `Фильтрация: инфляция, выпуск и ставка × сильный режимный разрыв` с delta cumulative loss `3.8417e-04`.

## Ограничение текущего шага

- Это regime-switching reduced-state HANK overlay, откалиброванный на full-HANK baseline, а не новая структурная full HANK solution с эндогенным режимным блоком.
- Но именно такая среда нужна как следующий кандидат для проверки, где flexible RL policy может превосходить classical filter-plus-rule benchmark.