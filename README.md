# pomdp-hank-policy

Belief-state policy design under regime uncertainty in a HANK model: comparison with the classical filter-plus-rule pipeline.

## Current Scope

The repository now contains a completed stage-1 baseline: a small, reproducible RBC pipeline that serves as the sanity-check layer before moving to NK, hidden-state, and RL blocks.

Implemented in stage 1:

- formal RBC model with a persistent technology shock;
- deterministic steady state;
- first-order local solution around the steady state via generalized Schur / QZ decomposition;
- stochastic simulation from synthetic shocks;
- IRF construction;
- external `gensys` IRF cross-check;
- residual-based sanity checks;
- generated figures and a short technical note.

Not implemented yet:

- hidden states;
- Kalman filtering;
- regime switching;
- RL;
- HANK blocks;
- real data.

## Model

The baseline model is a standard RBC economy with technology shock `z_t`:

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

State variables:

- log deviation of capital from steady state;
- current technology state `z_t`.

Control variables:

- log deviation of consumption from steady state;
- log deviation of labor from steady state.

## Quick Start

Install dependencies and run the full stage-1 pipeline:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/run_stage1.py
```

By default this writes all outputs into `outputs/stage1`.
The first benchmark run also downloads `dsge==0.1.3` into the user cache to load an external `gensys` implementation for IRF comparison.

## Generated Artifacts

After running the pipeline you will find:

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

The concise human-written note for stage 1 is in `docs/stage1_note.md`.

## Repository Layout

- `rbc_baseline/model.py`: model equations, steady state, observable reconstruction.
- `rbc_baseline/solver.py`: QZ/generalized-Schur solver for the linear policy system with Blanchard-Kahn checks.
- `rbc_baseline/benchmark.py`: external `gensys` benchmark loader and IRF comparison utilities.
- `rbc_baseline/pipeline.py`: simulation, IRF, diagnostics, plots, and artifact export.
- `scripts/run_stage1.py`: one-command entry point for the full baseline run.
- `docs/stage1_note.md`: short technical note for the baseline stage.

## Transition To Stage 2

Stage 2 should build on this infrastructure by replacing the simple RBC block with a small New Keynesian environment and then extending the state representation toward hidden-state and policy-design problems.
