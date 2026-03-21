# pomdp-hank-policy

Belief-state policy design under regime uncertainty in a HANK model: comparison with the classical filter-plus-rule pipeline.

## Current Scope

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

- hidden states;
- Kalman filtering;
- regime switching;
- RL;
- HANK blocks;
- real data.

## Stage 1 Model

The stage-1 baseline is a standard RBC economy with technology shock `z_t`:

- Euler equation:
  `1 / c_t = beta E_t[(1 / c_(t+1)) (alpha exp(z_(t+1)) k_(t+1)^(alpha-1) n_(t+1)^(1-alpha) + 1 - delta)]`
- Intratemporal labor condition:
  `psi n_t^nu = w_t / c_t`
- Production:
  `y_t = exp(z_t) k_t^alpha n_t^(1-alpha)`
- Resource constraint and capital law:
  `k_(t+1) = (1 - delta) k_t + y_t - c_t`
- Shock process:
  `z_(t+1) = rho z_t + sigma epsilon_(t+1)`

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

## Quick Start

Install dependencies and run either baseline:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_stage1.py
python3 scripts/run_stage2.py
```

By default these write outputs into `outputs/stage1` and `outputs/stage2`.
The first benchmark run also downloads `dsge==0.1.3` into the user cache to load an external `gensys` implementation for IRF comparison.

## Generated Artifacts

Stage 1 outputs:

- `outputs/stage1/steady_state.json`
- `outputs/stage1/solution.json`
- `outputs/stage1/simulated_paths.csv`
- `outputs/stage1/irf.csv`
- `outputs/stage1/diagnostics.csv`
- `outputs/stage1/diagnostics_summary.json`
- `outputs/stage1/benchmark_summary.json`
- `outputs/stage1/stage1_report.md`
- `outputs/stage1/gensys_irf.csv`
- `outputs/stage1/irf_comparison.csv`
- `outputs/stage1/figures/steady_state_summary.png`
- `outputs/stage1/figures/simulated_paths.png`
- `outputs/stage1/figures/irf.png`
- `outputs/stage1/figures/irf_qz_vs_gensys.png`

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

## Repository Layout

- `rbc_baseline/model.py`: model equations, steady state, observable reconstruction.
- `rbc_baseline/solver.py`: QZ/generalized-Schur solver for the linear policy system with Blanchard-Kahn checks.
- `rbc_baseline/benchmark.py`: external `gensys` benchmark loader and IRF comparison utilities.
- `rbc_baseline/pipeline.py`: simulation, IRF, diagnostics, plots, and artifact export.
- `nk_baseline/model.py`: small linear NK policy model specification.
- `nk_baseline/solver.py`: QZ/generalized-Schur NK solver and determinacy diagnostics.
- `nk_baseline/pipeline.py`: stage-2 IRF, simulation, determinacy-map, and report pipeline.
- `scripts/run_stage1.py`: one-command entry point for the full baseline run.
- `scripts/run_stage2.py`: one-command entry point for the stage-2 NK baseline.
- `docs/stage1_note.md`: short technical note for the baseline stage.

## Transition To Stage 3

The next stage should add hidden states and partial observability on top of the stage-2 NK policy environment, and only after that move to rule-based versus learning-based policy comparison.
