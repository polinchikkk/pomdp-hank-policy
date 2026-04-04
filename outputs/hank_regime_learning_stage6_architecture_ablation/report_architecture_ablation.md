# Stage 6 Architecture Ablation

На одной и той же 2x2 regime-switching карте сравниваются три policy architectures:
- filter + fixed rule
- filter + learned rule
- raw observations + learned rule

## Инфляция, выпуск, ставка × умеренный режимный разрыв
- `filter + fixed rule`: 7.605286e-03
- `filter + learned rule`: 2.199401e-03
- `raw observations + learned rule`: 2.009669e-03
- Rawobs minus classical: -5.595617e-03
- Rawobs minus belief-state RL: -1.897320e-04

## Инфляция, выпуск, ставка × сильный режимный разрыв
- `filter + fixed rule`: 8.486024e-03
- `filter + learned rule`: 2.604255e-03
- `raw observations + learned rule`: 2.398612e-03
- Rawobs minus classical: -6.087412e-03
- Rawobs minus belief-state RL: -2.056424e-04

## Инфляция, ставка × умеренный режимный разрыв
- `filter + fixed rule`: 8.502107e-03
- `filter + learned rule`: 3.683723e-03
- `raw observations + learned rule`: 4.182711e-03
- Rawobs minus classical: -4.319396e-03
- Rawobs minus belief-state RL: 4.989883e-04

## Инфляция, ставка × сильный режимный разрыв
- `filter + fixed rule`: 1.015686e-02
- `filter + learned rule`: 3.627200e-03
- `raw observations + learned rule`: 7.650551e-03
- Rawobs minus classical: -2.506308e-03
- Rawobs minus belief-state RL: 4.023351e-03
