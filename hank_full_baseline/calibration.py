from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict, List


@dataclass
class HANKCalibration:
    """Baseline calibration for the two-asset HANK model."""

    Y: float = 1.0
    N: float = 1.0
    K: float = 10.0
    r: float = 0.0125
    rstar: float = 0.0125
    tot_wealth: float = 14.0
    delta: float = 0.02
    pi: float = 0.0
    kappap: float = 0.1
    muw: float = 1.1
    Bh: float = 1.04
    Bg: float = 2.8
    G: float = 0.2
    transfer: float = 0.0
    eis: float = 0.5
    frisch: float = 1.0
    chi0: float = 0.35
    chi2: float = 2.0
    epsI: float = 4.0
    omega: float = 0.015
    kappaw: float = 0.1
    phi_pi: float = 1.5
    phi_y: float = 0.125
    rho_i: float = 0.7
    nZ: int = 3
    nB: int = 18
    nA: int = 28
    nK: int = 10
    bmax: float = 50.0
    amax: float = 4000.0
    kmax: float = 1.0
    b_dense_region_max: float = 5.0
    a_dense_region_max: float = 25.0
    k_dense_region_max: float = 0.35
    b_dense_region_share: float = 0.65
    a_dense_region_share: float = 0.60
    k_dense_region_share: float = 0.75
    rho_z: float = 0.966
    sigma_z: float = 0.88
    beta_guess: float = 0.9726
    chi1_guess: float = 23.0
    shock_T: int = 60
    shock_period: int = 0
    mp_shock_size: float = 0.001
    mp_shock_persistence: float = 0.0
    mpc_transfer_shock_size: float = 0.001
    mpc_transfer_horizon: int = 4
    low_liquidity_quantile: float = 0.2
    high_liquidity_quantile: float = 0.8
    wealthy_htm_a_quantile: float = 0.5
    low_liquidity_threshold: float = 0.25
    central_mass: float = 0.95
    fiscal_mode: str = "passive"
    output_dir: str = "outputs/hank_policy_stage2"
    random_seed: int = 0

    def calibration_dict(self) -> Dict[str, Any]:
        values = asdict(self)
        keep = [
            "Y",
            "N",
            "K",
            "r",
            "rstar",
            "tot_wealth",
            "delta",
            "pi",
            "kappap",
            "muw",
            "Bh",
            "Bg",
            "G",
            "transfer",
            "eis",
            "frisch",
            "chi0",
            "chi2",
            "epsI",
            "omega",
            "kappaw",
            "phi_pi",
            "phi_y",
            "rho_i",
            "nZ",
            "nB",
            "nA",
            "nK",
            "bmax",
            "amax",
            "kmax",
            "b_dense_region_max",
            "a_dense_region_max",
            "k_dense_region_max",
            "b_dense_region_share",
            "a_dense_region_share",
            "k_dense_region_share",
            "rho_z",
            "sigma_z",
            "Y_ss",
            "monetary_policy_shock",
        ]
        values["Y_ss"] = values["Y"]
        values["monetary_policy_shock"] = 0.0
        return {key: values[key] for key in keep}

    def steady_state_unknowns(self) -> Dict[str, float]:
        return {"beta": self.beta_guess, "chi1": self.chi1_guess}

    def steady_state_targets(self) -> Dict[str, Any]:
        return {"asset_mkt": 0.0, "B": "Bh"}

    def rule_spec(self) -> Dict[str, float]:
        return {
            "rho_i": self.rho_i,
            "phi_pi": self.phi_pi,
            "phi_y": self.phi_y,
            "shock_name": "monetary_policy_shock",
            "shock_size": self.mp_shock_size,
            "shock_persistence": self.mp_shock_persistence,
            "output_measure": "output_gap",
        }


def default_calibration() -> HANKCalibration:
    return HANKCalibration()


def sequence_jacobian_tutorial_calibration(
    base: HANKCalibration | None = None,
    *,
    preserve_grid: bool = True,
) -> HANKCalibration:
    """Economic calibration used in the sequence-jacobian two-asset tutorial.

    By default we keep the project's reduced numerical grid so that the
    comparison isolates economic calibration choices rather than grid size.
    """

    cfg = default_calibration() if base is None else base
    updates = {
        "chi0": 0.25,
        "chi2": 2.0,
        "omega": 0.005,
        "sigma_z": 0.92,
        "rho_z": 0.966,
        "eis": 0.5,
        "frisch": 1.0,
        "epsI": 4.0,
        "kappap": 0.1,
        "kappaw": 0.1,
        "beta_guess": 0.976,
        "chi1_guess": 6.5,
    }
    if not preserve_grid:
        updates.update({
            "nB": 50,
            "nA": 70,
            "nK": 50,
            "b_dense_region_share": 0.7,
            "a_dense_region_share": 0.65,
            "k_dense_region_share": 0.8,
        })
    return replace(cfg, **updates)


def calibration_table_metadata() -> List[Dict[str, str]]:
    return [
        {"parameter": "beta", "label": "beta", "description": "Коэффициент дисконтирования", "source": "Подбирается по asset market"},
        {"parameter": "eis", "label": "eis", "description": "Межвременная эластичность замещения", "source": "Baseline two-asset HANK"},
        {"parameter": "frisch", "label": "frisch", "description": "Фришева эластичность труда", "source": "Baseline two-asset HANK"},
        {"parameter": "chi0", "label": "chi0", "description": "Сдвиг в знаменателе функции издержек ребалансировки", "source": "Baseline two-asset HANK"},
        {"parameter": "chi1_guess", "label": "chi1", "description": "Масштаб издержек ребалансировки", "source": "Подбирается в steady state"},
        {"parameter": "chi2", "label": "chi2", "description": "Кривизна издержек ребалансировки", "source": "Baseline two-asset HANK"},
        {"parameter": "phi_pi", "label": "phi_pi", "description": "Коэффициент реакции правила Тейлора на инфляцию", "source": "Классическая денежно-кредитная политика"},
        {"parameter": "phi_y", "label": "phi_y", "description": "Коэффициент реакции правила Тейлора на выпуск", "source": "Классическая денежно-кредитная политика"},
        {"parameter": "rho_i", "label": "rho_i", "description": "Инерционность процентной ставки", "source": "Классическая денежно-кредитная политика"},
        {"parameter": "kappap", "label": "kappap", "description": "Наклон ценовой кривой Филлипса", "source": "Номинальные жесткости"},
        {"parameter": "kappaw", "label": "kappaw", "description": "Наклон кривой Филлипса по заработной плате", "source": "Номинальные жесткости"},
        {"parameter": "delta", "label": "delta", "description": "Норма амортизации капитала", "source": "Производственный блок"},
        {"parameter": "omega", "label": "omega", "description": "Спрэд между доходностью ликвидного актива и ставкой политики", "source": "Клин ликвидной доходности"},
        {"parameter": "rho_z", "label": "rho_z", "description": "Персистентность идосинкратического дохода", "source": "Процесс идосинкратического дохода"},
        {"parameter": "sigma_z", "label": "sigma_z", "description": "Безусловная дисперсия лог-идосинкратического дохода", "source": "Процесс идосинкратического дохода"},
        {"parameter": "nB", "label": "nB", "description": "Число узлов по liquid asset", "source": "Численная сетка"},
        {"parameter": "nA", "label": "nA", "description": "Число узлов по illiquid asset", "source": "Численная сетка"},
        {"parameter": "nZ", "label": "nZ", "description": "Число узлов доходного процесса", "source": "Численная сетка"},
        {"parameter": "mp_shock_size", "label": "shock", "description": "Размер монетарного шока", "source": "Эксперимент денежно-кредитной политики"},
        {"parameter": "mp_shock_persistence", "label": "rho_eps", "description": "Персистентность монетарного шока", "source": "Эксперимент денежно-кредитной политики"},
    ]
