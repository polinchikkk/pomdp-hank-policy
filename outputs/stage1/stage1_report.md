# Stage 1 Note: RBC Baseline

## Model

- Baseline model: real business cycle model with a persistent technology shock.
- State variables: log capital deviation and technology level `z_t`.
- Control variables: log consumption deviation and log labor deviation.
- Shock law: `z_(t+1) = rho z_t + sigma epsilon_(t+1)`.

## Solution Method

- Solution class: first-order local perturbation around the deterministic steady state.
- Numerical approach: steady state in closed form, Jacobians by finite differences, and a generalized Schur (QZ) decomposition of the linearized equilibrium system.
- Canonical interpretation: this is a Klein/Sims-style linear rational expectations solution with explicit stable-versus-unstable root separation.
- This is the stage-1 benchmark and contains no hidden states, filtering, policy block, RL, regime switching, or external data.

## Core Outputs

- Steady state saved in `steady_state.json`.
- Linear policy and transition matrices saved in `solution.json`.
- Stochastic simulation saved in `simulated_paths.csv`.
- Impulse responses saved in `irf.csv`.
- External gensys benchmark IRFs saved in `gensys_irf.csv`.
- IRF comparison table saved in `irf_comparison.csv`.
- Diagnostic residuals saved in `diagnostics.csv`.
- Figures saved in `figures/steady_state_summary.png`, `figures/simulated_paths.png`, `figures/irf.png`, and `figures/irf_qz_vs_gensys.png`.

## Calibration Snapshot

- beta = 0.990
- alpha = 0.360
- delta = 0.025
- rho = 0.950
- sigma = 0.010
- implied steady-state labor disutility weight psi = 7.903
- steady-state levels: y = 1.2223, c = 0.9089, i = 0.3134, k = 12.5365, n = 0.3300

## Sanity Checks

- Linearized system residual max: 2.554e-15
- Spectral radius of the transition matrix: 0.9575
- Solver backend: generalized_schur_qz
- Stable roots: 2
- Finite roots outside the unit circle: 1
- Infinite generalized eigenvalues: 1
- Blanchard-Kahn condition satisfied: True
- Stable invariant block condition number: 1.1057e+00
- Max conditional residual on the stochastic simulation: 0.0030
- RMS conditional residual: 0.0004
- Runtime: 0.452 seconds
- Impact IRF signs after a positive technology shock: {'output': True, 'consumption': True, 'investment': True, 'labor': True}

## External Benchmark

- External benchmark source: dsge==0.1.3 (dsge/gensys.py)
- gensys existence/uniqueness code: [1, 1]
- External benchmark reports existence: True | uniqueness: True
- Max absolute IRF difference between internal QZ solver and external gensys benchmark: 3.509e-15
- RMS IRF difference: 1.073e-15

Seed stability summary:
- seed 0: finite=True, max |state dev|=0.0921, max |residual|=0.0034
- seed 1: finite=True, max |state dev|=0.1097, max |residual|=0.0023
- seed 2: finite=True, max |state dev|=0.1040, max |residual|=0.0027
- seed 3: finite=True, max |state dev|=0.0903, max |residual|=0.0040
- seed 4: finite=True, max |state dev|=0.1166, max |residual|=0.0037

## Interpretation

- The impact IRF is economically sensible for a positive technology shock: output, consumption, investment, and labor all rise on impact.
- The QZ split delivers exactly two stable roots for two predetermined state variables, so the Blanchard-Kahn count condition is met.
- The transition matrix is stable because its spectral radius is below one.
- The external gensys benchmark reproduces the same IRFs up to numerical precision, which provides an independent cross-check beyond the internal solver diagnostics.
- Conditional residuals remain small in the simulated neighborhood of the steady state, which is consistent with a first-order local approximation.

## Not Implemented Yet

- New Keynesian policy block.
- Hidden states or state-space filtering.
- RL or any adaptive policy design.
- HANK structure, regime switching, and real data.
