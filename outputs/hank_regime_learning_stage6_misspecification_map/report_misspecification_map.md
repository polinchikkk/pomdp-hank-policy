# Stage 6 Misspecification Map

В этой серии лучший learned controller из architecture ablation сравнивается с несколькими misspecified classical architectures на той же 2x2 regime-switching карте.

## Best Learned Selection

- `macro_core_moderate_gap`: лучшая learned architecture = `raw_observations`, mean cumulative loss = `2.009669e-03`.
- `macro_core_strong_gap`: лучшая learned architecture = `raw_observations`, mean cumulative loss = `2.398612e-03`.
- `thin_information_moderate_gap`: лучшая learned architecture = `belief_state`, mean cumulative loss = `3.683723e-03`.
- `thin_information_strong_gap`: лучшая learned architecture = `belief_state`, mean cumulative loss = `3.627200e-03`.

## Misspecification Summary

### Single-regime normal-only filter
- Mean relative improvement of learned policy vs misspecified classical: `66.57%`.
- Mean excess loss of misspecified classical vs correctly specified switching rule: `4.245727e-04`.
- Scenario win share: `1.00`.
- Mean seed win rate: `1.00`.

### Normal-biased regime persistence
- Mean relative improvement of learned policy vs misspecified classical: `66.44%`.
- Mean excess loss of misspecified classical vs correctly specified switching rule: `4.093701e-04`.
- Scenario win share: `1.00`.
- Mean seed win rate: `1.00`.

### Overstated measurement noise
- Mean relative improvement of learned policy vs misspecified classical: `62.92%`.
- Mean excess loss of misspecified classical vs correctly specified switching rule: `-2.406331e-04`.
- Scenario win share: `1.00`.
- Mean seed win rate: `1.00`.

### Inflation-only simple rule
- Mean relative improvement of learned policy vs misspecified classical: `-136.88%`.
- Mean excess loss of misspecified classical vs correctly specified switching rule: `-7.015303e-03`.
- Scenario win share: `0.00`.
- Mean seed win rate: `0.03`.
