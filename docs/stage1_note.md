# Stage 1 Note: Minimal RBC Baseline

## Goal

Stage 1 was used to build a clean sanity-check baseline for the future DSGE and RL pipeline. The target was not novelty, but a reproducible computational foundation that can later be compared against richer NK, hidden-state, and policy-learning setups.

## Model And Method

The chosen benchmark is a standard RBC model with a persistent technology shock. The endogenous state is capital, the exogenous state is technology, and the control block contains consumption and labor. The model is solved with a first-order local approximation around the deterministic steady state. In code, the steady state is computed analytically, the equilibrium system is linearized numerically by finite-difference Jacobians, and the policy matrices are recovered with a generalized Schur (QZ) decomposition of the linear rational expectations system.

This choice matches the stage-1 requirement well: it is simple, canonical, easy to debug, and gives interpretable IRFs without introducing hidden states, filtering, RL, or empirical calibration.

## Outputs

The baseline package now includes:

- model code in [model.py](/Users/polinazosimova/pomdp-hank-policy/rbc_baseline/model.py);
- solver code in [solver.py](/Users/polinazosimova/pomdp-hank-policy/rbc_baseline/solver.py);
- external benchmark code in [benchmark.py](/Users/polinazosimova/pomdp-hank-policy/rbc_baseline/benchmark.py);
- simulation, IRF, diagnostics, and export logic in [pipeline.py](/Users/polinazosimova/pomdp-hank-policy/rbc_baseline/pipeline.py);
- one-command execution in [run_stage1.py](/Users/polinazosimova/pomdp-hank-policy/scripts/run_stage1.py);
- generated artifacts under [outputs/stage1](/Users/polinazosimova/pomdp-hank-policy/outputs/stage1).

The generated figures are:

- steady-state and parameter summary in [steady_state_summary.png](/Users/polinazosimova/pomdp-hank-policy/outputs/stage1/figures/steady_state_summary.png);
- stochastic simulated paths in [simulated_paths.png](/Users/polinazosimova/pomdp-hank-policy/outputs/stage1/figures/simulated_paths.png);
- IRFs to a one-standard-deviation technology shock in [irf.png](/Users/polinazosimova/pomdp-hank-policy/outputs/stage1/figures/irf.png).
- direct overlay of the internal QZ IRFs and external `gensys` IRFs in [irf_qz_vs_gensys.png](/Users/polinazosimova/pomdp-hank-policy/outputs/stage1/figures/irf_qz_vs_gensys.png).

## Sanity Checks

The current run passes the intended stage-1 checks:

- the linear solution residual is `2.55e-15`, so the local system is solved to numerical precision;
- the QZ split produces exactly `2` stable roots for `2` predetermined state variables, so the Blanchard-Kahn count condition is satisfied;
- the transition matrix spectral radius is `0.9575`, so the linear dynamics are stable;
- impact IRFs are economically sensible for a positive technology shock: output, consumption, investment, and labor all increase on impact;
- the external `gensys` benchmark from `dsge==0.1.3` returns `RC = [1, 1]`, so the benchmark solver also reports existence and uniqueness;
- the maximum IRF gap between the internal QZ solver and the external `gensys` benchmark is about `3.5e-15`, which is numerical zero for practical purposes;
- the maximum conditional residual on the simulation sample is `0.0030`, with RMS residual `0.00036`;
- seed-based reruns over seeds `0` to `4` remain finite and do not generate explosive paths;
- end-to-end runtime, including the external benchmark check, stays around one second on the current machine after caching.

The automatically generated machine report with the same run statistics is in [stage1_report.md](/Users/polinazosimova/pomdp-hank-policy/outputs/stage1/stage1_report.md).

The repository now includes an actual external `gensys`-based IRF cross-check. A natural next paper-ready extension would be to add a Dynare `.mod` replication as a third benchmark layer, but the stage-1 package is no longer relying only on internal diagnostics.

## What Is Still Missing

This stage intentionally does not include a policy block, hidden states, Kalman filtering, regime switching, RL, HANK heterogeneity, or real data. Those belong to later stages once the baseline DSGE infrastructure is already stable and reproducible.
