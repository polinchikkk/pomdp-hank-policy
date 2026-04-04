# Stage 6 Environment Shift

В этой серии learned policy и baseline-tuned simple rule настраиваются на baseline regime environment, а затем без перенастройки оцениваются на новых структурных средах.

## Selected Learned Architectures

- `macro_core_moderate_gap`: input mode `raw_observations`, checkpoint `32`, selection objective `9.590364e-04`.
- `macro_core_strong_gap`: input mode `raw_observations`, checkpoint `32`, selection objective `8.697673e-04`.
- `thin_information_moderate_gap`: input mode `belief_state`, checkpoint `16`, selection objective `1.485334e-03`.
- `thin_information_strong_gap`: input mode `belief_state`, checkpoint `32`, selection objective `9.185121e-04`.

## Transfer Summary

### Baseline regime environment
- Learned vs fixed: `61.58%` в среднем.
- Learned vs retuned simple rule: `-333.79%` в среднем.
- Scenario win share vs fixed: `1.00`.
- Scenario win share vs retuned: `0.00`.

### More persistent hidden regimes
- Learned vs fixed: `60.88%` в среднем.
- Learned vs retuned simple rule: `-352.87%` в среднем.
- Scenario win share vs fixed: `1.00`.
- Scenario win share vs retuned: `0.00`.

### Shifted macro transmission
- Learned vs fixed: `56.05%` в среднем.
- Learned vs retuned simple rule: `-476.98%` в среднем.
- Scenario win share vs fixed: `1.00`.
- Scenario win share vs retuned: `0.00`.

### Stronger distributional channel
- Learned vs fixed: `51.85%` в среднем.
- Learned vs retuned simple rule: `-326.36%` в среднем.
- Scenario win share vs fixed: `1.00`.
- Scenario win share vs retuned: `0.00`.
