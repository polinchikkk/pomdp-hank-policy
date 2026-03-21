# Stage 2 Report: Small NK Policy Baseline

## Model

- Baseline environment: canonical linear New Keynesian model in gap variables around the zero steady state.
- The model is written directly in linearized deviation variables, so the steady-state reference point is normalized to zero.
- Endogenous policy variables: output gap `x_t`, inflation `pi_t`, nominal rate `i_t`.
- Exogenous shocks: natural-rate shock `r^n_t`, cost-push shock `u_t`, monetary-policy shock `nu_t`.
- Policy rule: `i_t = phi_pi * pi_t + phi_x * x_t + nu_t`.

## Calibration

- beta = 0.990
- sigma = 1.000
- kappa = 0.100
- phi_pi = 1.500
- phi_x = 0.500
- rho_r = 0.800
- rho_u = 0.500
- rho_nu = 0.500

## Solution

- Solver: generalized_schur_qz
- Stable roots: 3
- Roots outside the unit circle: 2
- Infinite generalized eigenvalues: 1
- Blanchard-Kahn condition satisfied: True
- Spectral radius of the transition matrix: 0.8000
- Linear-system residual max: 4.441e-16
- Runtime: 0.225 seconds

## IRF Sanity Checks

- Positive demand shock: `x`, `pi`, and `i` all rise on impact -> {'x_positive': True, 'pi_positive': True, 'i_positive': True}
- Positive cost-push shock: inflation rises and output falls on impact -> {'x_negative': True, 'pi_positive': True, 'i_positive': True}
- Positive monetary tightening shock: output gap and inflation fall while the nominal rate rises on impact -> {'x_negative': True, 'pi_negative': True, 'i_positive': True}

## Baseline Policy Diagnostics

- Determinacy map computed on a grid over `(phi_pi, phi_x)`.
- Determinate parameter combinations in the grid: 76.8%
- For `phi_pi > 1`, the entire scanned grid is determinate: 100.0%
- For `phi_pi <= 1`, determinacy weakens sharply: 13.6%
- This pattern is consistent with the Taylor principle in the scanned parameter region.
- Simulated paths remain finite: True
- Internal linear-equation consistency check: 2.082e-17
- This residual metric confirms that the computed policy functions solve the linearized system; the main economic validation still comes from IRF logic and determinacy.

## Not Implemented Yet

- Hidden states or signal extraction.
- Regime switching or time variation in the policy rule.
- RL or adaptive policy design.
- Heterogeneous agents and real-data estimation.
