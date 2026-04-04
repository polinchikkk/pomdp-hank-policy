from __future__ import annotations

import numpy as np
from sequence_jacobian import combine, create_model, grids, hetblocks, simple, solved

from .calibration import HANKCalibration


hh = hetblocks.hh_twoasset.hh


@simple
def pricing(pi, mc, r, Y, kappap, mup):
    nkpc = (
        kappap * (mc - 1 / mup)
        + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))
        - (1 + pi).apply(np.log)
    )
    return nkpc


@simple
def arbitrage(div, p, r):
    equity = div(+1) + p(+1) - p * (1 + r(+1))
    return equity


@simple
def labor(Y, w, K, Z, alpha):
    N = (Y / Z / K(-1) ** alpha) ** (1 / (1 - alpha))
    mc = w * N / (1 - alpha) / Y
    return N, mc


@simple
def investment(Q, K, r, N, mc, Z, delta, epsI, alpha):
    inv = (K / K(-1) - 1) / (delta * epsI) + 1 - Q
    val = (
        alpha * Z(+1) * (N(+1) / K) ** (1 - alpha) * mc(+1)
        - (K(+1) / K - (1 - delta) + (K(+1) / K - 1) ** 2 / (2 * delta * epsI))
        + K(+1) / K * Q(+1)
        - (1 + r(+1)) * Q
    )
    return inv, val


@simple
def dividend(Y, w, N, K, pi, mup, kappap, delta, epsI):
    psip = mup / (mup - 1) / 2 / kappap * (1 + pi).apply(np.log) ** 2 * Y
    k_adjust = K(-1) * (K / K(-1) - 1) ** 2 / (2 * delta * epsI)
    I = K - (1 - delta) * K(-1) + k_adjust
    div = Y - w * N - I - psip
    return psip, I, div


@solved(unknowns={"i": (-0.1, 0.2)}, targets=["mp_rule"], solver="brentq")
def taylor(i, rstar, monetary_policy_shock, pi, Y, Y_ss, rho_i, phi_pi, phi_y):
    output_gap = Y / Y_ss - 1
    mp_rule = i - rho_i * i(-1) - (1 - rho_i) * (rstar + phi_pi * pi + phi_y * output_gap) - monetary_policy_shock
    return mp_rule, output_gap


@simple
def fiscal(r, w, N, G, Bg):
    tax = (r * Bg + G) / w / N
    return tax


@simple
def finance(i, p, pi, r, div, omega, pshare):
    rb = r - omega
    ra = pshare(-1) * (div + p) / p(-1) + (1 - pshare(-1)) * (1 + r) - 1
    fisher = 1 + i(-1) - (1 + r) * (1 + pi)
    return rb, ra, fisher


@simple
def wage(pi, w):
    piw = (1 + pi) * w / w(-1) - 1
    return piw


@simple
def union(piw, N, tax, w, UCE, kappaw, muw, vphi, frisch, beta):
    wnkpc = (
        kappaw * (vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * N * UCE / muw)
        + beta * (1 + piw(+1)).apply(np.log)
        - (1 + piw).apply(np.log)
    )
    return wnkpc


@simple
def mkt_clearing(p, A, B, Bg, C, I, G, CHI, psip, omega, Y):
    wealth = A + B
    asset_mkt = p + Bg - wealth
    goods_mkt = C + I + G + CHI + psip + omega * B - Y
    return asset_mkt, wealth, goods_mkt


@simple
def share_value(p, tot_wealth, Bh):
    pshare = p / (tot_wealth - Bh)
    return pshare


@solved(unknowns={"pi": (-0.1, 0.1)}, targets=["nkpc"], solver="brentq")
def pricing_solved(pi, mc, r, Y, kappap, mup):
    nkpc = (
        kappap * (mc - 1 / mup)
        + Y(+1) / Y * (1 + pi(+1)).apply(np.log) / (1 + r(+1))
        - (1 + pi).apply(np.log)
    )
    return nkpc


@solved(unknowns={"p": (5.0, 15.0)}, targets=["equity"], solver="brentq")
def arbitrage_solved(div, p, r):
    equity = div(+1) + p(+1) - p * (1 + r(+1))
    return equity


@simple
def partial_ss(Y, N, K, r, tot_wealth, Bg, delta):
    p = tot_wealth - Bg
    mc = 1 - r * (p - K) / Y
    mup = 1 / mc
    alpha = (r + delta) * K / Y / mc
    Z = Y * K ** (-alpha) * N ** (alpha - 1)
    w = mc * (1 - alpha) * Y / N
    return p, mc, mup, alpha, Z, w


@simple
def union_ss(tax, w, UCE, N, muw, frisch):
    vphi = (1 - tax) * w * UCE / muw / N ** (1 + 1 / frisch)
    wnkpc = vphi * N ** (1 + 1 / frisch) - (1 - tax) * w * UCE / muw
    return vphi, wnkpc


def constrained_agrid(amax, n, amin=0.0, dense_region_max=None, dense_region_share=0.6):
    if n <= 3:
        return grids.agrid(amax=amax, n=n, amin=amin)
    if dense_region_max is None or dense_region_max <= amin or dense_region_max >= amax:
        return grids.agrid(amax=amax, n=n, amin=amin)

    dense_points = max(3, min(n - 1, int(round((n - 1) * dense_region_share))))
    upper_points = n - dense_points + 1

    dense_grid = grids.agrid(amax=dense_region_max, n=dense_points, amin=amin)
    upper_grid = grids.agrid(amax=amax, n=upper_points, amin=dense_region_max)
    grid = np.concatenate([dense_grid, upper_grid[1:]])
    grid[0] = amin
    grid[-1] = amax
    return grid


def make_grids(
    bmax,
    amax,
    kmax,
    nB,
    nA,
    nK,
    nZ,
    rho_z,
    sigma_z,
    b_dense_region_max,
    a_dense_region_max,
    k_dense_region_max,
    b_dense_region_share,
    a_dense_region_share,
    k_dense_region_share,
):
    b_grid = constrained_agrid(
        amax=bmax,
        n=nB,
        dense_region_max=b_dense_region_max,
        dense_region_share=b_dense_region_share,
    )
    a_grid = constrained_agrid(
        amax=amax,
        n=nA,
        dense_region_max=a_dense_region_max,
        dense_region_share=a_dense_region_share,
    )
    k_grid = constrained_agrid(
        amax=kmax,
        n=nK,
        dense_region_max=k_dense_region_max,
        dense_region_share=k_dense_region_share,
    )[::-1].copy()
    e_grid, _, Pi = grids.markov_rouwenhorst(rho=rho_z, sigma=sigma_z, N=nZ)
    return b_grid, a_grid, k_grid, e_grid, Pi


def income(e_grid, tax, w, N, transfer):
    z_grid = (1 - tax) * w * N * e_grid + transfer
    return z_grid


def build_models(config: HANKCalibration) -> dict:
    if config.fiscal_mode != "passive":
        raise NotImplementedError("В первой полной HANK-версии реализован только пассивный фискальный блок.")

    household = hh.add_hetinputs([income, make_grids])
    production = combine([labor, investment])
    production_solved = production.solved(
        unknowns={"Q": 1.0, "K": config.K},
        targets=["inv", "val"],
        solver="broyden_custom",
    )
    blocks = [
        household,
        pricing_solved,
        arbitrage_solved,
        production_solved,
        dividend,
        taylor,
        fiscal,
        share_value,
        finance,
        wage,
        union,
        mkt_clearing,
    ]
    model = create_model(blocks, name="Two-Asset HANK")

    blocks_ss = [
        household,
        partial_ss,
        dividend,
        taylor,
        fiscal,
        share_value,
        finance,
        union_ss,
        mkt_clearing,
    ]
    model_ss = create_model(blocks_ss, name="Two-Asset HANK SS")

    return {
        "model": model,
        "model_ss": model_ss,
        "calibration": config.calibration_dict(),
        "ss_unknowns": config.steady_state_unknowns(),
        "ss_targets": config.steady_state_targets(),
        "unknowns": ["r", "w", "Y"],
        "targets": ["asset_mkt", "fisher", "wnkpc"],
        "exogenous": ["monetary_policy_shock", "rstar", "Z", "G"],
    }
