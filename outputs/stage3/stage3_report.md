# Stage 3 Report: Multishock Hidden-State Baseline

## Setup

- Structural base: stage-2 linear New Keynesian policy model.
- Hidden state vector: natural-rate shock `r_n`, cost-push shock `u`, and monetary-policy shock `nu`.
- Baseline observables: output gap `x`, inflation `pi`, and nominal rate `i`.
- Additional observability stress test: policy panel with only `pi` and `i`.
- State-space form: `state_(t+1) = A state_t + B eps_(t+1)`, `obs_t = C state_t + D eta_t`.
- The model remains linearized around the zero steady state, so every latent state and observation is measured as a deviation from that normalized reference point.
- All plotted quantities are reported in percentage points.
- The observation equation is inherited from the stage-2 NK policy solution, so the filter must disentangle competing latent shocks rather than denoise a hand-written one-factor measurement system.

## Baseline Filter

- Baseline measurement-noise scenario: `medium`
- Baseline observation design: `full_panel`
- Aggregate RMSE across latent states: 1.4067e-03
- Aggregate mean absolute error: 1.0275e-03
- Mean 95% confidence-band width: 4.9341e-03
- Single-path mean empirical 95% coverage: 93.75%
- Log-likelihood: 2281.003

- `r_n`: RMSE 1.1249e-03, correlation 0.9976, 95% coverage 93.75%
- `u`: RMSE 7.4377e-04, correlation 0.9980, 95% coverage 93.33%
- `nu`: RMSE 2.0293e-03, correlation 0.7617, 95% coverage 94.17%

- The single-path coverage number is illustrative only; the main calibration check for interval coverage comes from the Monte Carlo averages below.

## Noise Sensitivity

- `small` noise: aggregate RMSE 5.9450e-04, mean 95% coverage 94.03%
- `medium` noise: aggregate RMSE 1.4067e-03, mean 95% coverage 93.75%
- `large` noise: aggregate RMSE 2.3988e-03, mean 95% coverage 93.61%

- Filter accuracy deteriorates monotonically as measurement noise rises, but the ranking now reflects recovery of three latent shocks rather than one isolated AR(1) process.

## Observation-Set Stress Test

- `full_panel` observables: aggregate RMSE 1.4067e-03, mean 95% coverage 93.75%
- `policy_panel` observables: aggregate RMSE 2.8895e-03, mean 95% coverage 93.06%

- Restricting the observation set to the policy panel makes inference materially harder, which is a more meaningful incomplete-information stress test than changing noise alone.

## Monte Carlo Robustness

- Monte Carlo runs per scenario: 25
- For the baseline `medium` scenario, mean aggregate 95% coverage across runs is 95.22%, which is close to the nominal 95% target.

- `r_n`: mean RMSE 1.1136e-03 (std 5.24e-05), mean 95% coverage 95.22% (std 1.89%)
- `u`: mean RMSE 7.0943e-04 (std 3.20e-05), mean 95% coverage 95.43% (std 1.37%)
- `nu`: mean RMSE 1.9262e-03 (std 8.77e-05), mean 95% coverage 95.02% (std 1.49%)

- Single-path coverage can drift across individual trajectories, so Monte Carlo coverage is the more informative benchmark for interval calibration.

## Innovation Diagnostics

- `x`: raw mean 1.308e-03, raw lag-1 autocorr -0.032, standardized mean 6.546e-02, standardized std 1.006, standardized lag-1 autocorr -0.032
- `pi`: raw mean -4.965e-04, raw lag-1 autocorr -0.029, standardized mean 1.883e-02, standardized std 1.059, standardized lag-1 autocorr -0.013
- `i`: raw mean -5.465e-04, raw lag-1 autocorr -0.011, standardized mean -1.112e-01, standardized std 1.050, standardized lag-1 autocorr 0.050

- Innovation moments stay close to their ideal values, which supports the internal consistency of the linear-Gaussian filter specification.

## Mild Parameter Misspecification

- True data-generating process uses `rho_r = 0.80`.
- Misspecified filter uses `rho_r = 0.70`.
- Misspecified aggregate RMSE: 1.4481e-03
- Misspecified mean 95% coverage: 94.03%

- `r_n`: RMSE 1.2848e-03, correlation 0.9968, 95% coverage 94.58%
- `u`: RMSE 7.0949e-04, correlation 0.9982, 95% coverage 93.75%
- `nu`: RMSE 2.0339e-03, correlation 0.7603, 95% coverage 93.75%

- Even a mild persistence misspecification degrades latent-state recovery, which is a useful bridge from the clean baseline toward more realistic partial-information settings.

## Interpretation

- Stage 3 now treats incomplete information as a multishock inference problem: the policymaker observes noisy macro variables but does not directly observe which structural shock is driving them.
- Among the three hidden components, the monetary-policy shock `nu` is recovered least precisely, which points to weaker identifiability of this component from the observed macro panel.
- This is still a linear-Gaussian baseline, but it now includes both innovation diagnostics and a first misspecification stress test rather than relying only on in-model fit.
- The resulting pipeline is a stronger bridge from stage 2 policy analysis to later work on hidden states, filtering, and learning-based policy design.

## Not Implemented Yet

- Correlated structural shocks, correlated measurement noise, Kalman smoothing, or full maximum-likelihood estimation.
- Hidden endogenous state blocks, structural breaks, or regime switching.
- RL and belief-state policy optimization.
