from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.steady_state import solve_steady_state
from hank_partial_info_baseline.state_space import fit_reduced_state_space

from .regime_evaluation import evaluate_policy_under_regime_uncertainty, evaluate_regime_filter
from .regime_model import RegimeSwitchingConfig, build_regime_switching_model, regime_model_spec_payload
from .regime_simulation import (
    RegimePolicyRun,
    simulate_filtered_policy,
    simulate_full_information_policy,
    simulate_hidden_regimes,
)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported type: {type(value)!r}")


def _save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default))


def _state_frames(run: RegimePolicyRun) -> tuple[pd.DataFrame, pd.DataFrame]:
    periods = np.arange(run.true_states.shape[0], dtype=int)
    true_frame = pd.DataFrame({
        "scenario": run.scenario_name,
        "scenario_label": run.scenario_label,
        "period": periods,
        "hidden_regime": run.hidden_regimes,
        "policy_name": run.policy_name,
        "policy_label": run.policy_label,
    })
    filtered_frame = true_frame.copy()
    state_names = (
        "rstar_gap",
        "productivity_gap",
        "fiscal_gap",
        "inflation_gap",
        "output_gap",
        "low_liquidity_gap",
        "mean_mpc_gap",
    )
    for index, state_name in enumerate(state_names):
        true_frame[f"true_{state_name}"] = run.true_states[:, index]
        if run.filtered_states is not None:
            filtered_frame[f"filtered_{state_name}"] = run.filtered_states[:, index]
            filtered_frame[f"true_{state_name}"] = run.true_states[:, index]
    if run.filtered_mode_probabilities is not None:
        filtered_frame["p_normal"] = run.filtered_mode_probabilities[:, 0]
        filtered_frame["p_stress"] = run.filtered_mode_probabilities[:, 1]
    filtered_frame["policy_rate"] = run.policy_rate
    true_frame["policy_rate"] = run.policy_rate
    return true_frame, filtered_frame


def _observation_frame(run: RegimePolicyRun) -> pd.DataFrame:
    frame = pd.DataFrame({
        "scenario": run.scenario_name,
        "scenario_label": run.scenario_label,
        "period": np.arange(run.observations.shape[0], dtype=int),
        "hidden_regime": run.hidden_regimes,
        "policy_name": run.policy_name,
        "policy_label": run.policy_label,
        "policy_rate": run.policy_rate,
    })
    for index, name in enumerate(run.noisy_observation_names):
        frame[f"observed_{name}"] = run.observations[:, index]
    return frame


def _write_report(
    output_dir: Path,
    config: RegimeSwitchingConfig,
    filter_metrics: pd.DataFrame,
    policy_metrics: pd.DataFrame,
) -> None:
    easiest_filter = filter_metrics.loc[filter_metrics["regime_accuracy"].idxmax()]
    hardest_filter = filter_metrics.loc[filter_metrics["regime_accuracy"].idxmin()]
    best_policy = policy_metrics.loc[policy_metrics["delta_cumulative_policy_loss_filtered_minus_full_information"].idxmin()]
    worst_policy = policy_metrics.loc[policy_metrics["delta_cumulative_policy_loss_filtered_minus_full_information"].idxmax()]

    lines = [
        "# Этап 5. Regime-switching HANK при неполной информации",
        "",
        "## Постановка",
        "",
        "- Используется reduced-state HANK layer из этапа 3, но поверх него задаётся скрытое переключение между режимами `normal` и `stress`.",
        "- В `stress` режиме усиливается policy transmission в инфляции, выпуске и распределительных состояниях (`low_liquidity_gap`, `mean_mpc_gap`).",
        "- Регулятор не наблюдает режим напрямую и использует switching Kalman / IMM filter.",
        "- Classical benchmark строится как `switching filter + fixed Taylor-type rule`.",
        "",
        "## Сценарии",
        "",
    ]
    for scenario in config.scenario_specs():
        lines.append(
            f"- `{scenario['label']}`: noisy observables `{', '.join(scenario['noisy_observations'])}`, regime gap `{scenario['gap_label']}`."
        )

    lines.extend([
        "",
        "## Качество фильтрации режима",
        "",
        f"- Наилучшая regime classification accuracy: `{easiest_filter['scenario_label']}` с accuracy `{easiest_filter['regime_accuracy']:.3f}` и Brier score `{easiest_filter['stress_brier_score']:.4e}`.",
        f"- Наиболее сложный режимный сценарий для фильтра: `{hardest_filter['scenario_label']}` с accuracy `{hardest_filter['regime_accuracy']:.3f}`.",
        "",
        "## Качество classical policy under regime uncertainty",
        "",
        f"- Лучший сценарий по разнице накопленной потери относительно полной информации: `{best_policy['scenario_label']}` с delta cumulative loss `{best_policy['delta_cumulative_policy_loss_filtered_minus_full_information']:.4e}`.",
        f"- Наиболее затратный сценарий: `{worst_policy['scenario_label']}` с delta cumulative loss `{worst_policy['delta_cumulative_policy_loss_filtered_minus_full_information']:.4e}`.",
        "",
        "## Ограничение текущего шага",
        "",
        "- Это regime-switching reduced-state HANK overlay, откалиброванный на full-HANK baseline, а не новая структурная full HANK solution с эндогенным режимным блоком.",
        "- Но именно такая среда нужна как следующий кандидат для проверки, где flexible RL policy может превосходить classical filter-plus-rule benchmark.",
    ])
    (output_dir / "report_stage5_regime_switching_hank.md").write_text("\n".join(lines))


def run_pipeline(
    config: RegimeSwitchingConfig | None = None,
    output_dir: str | None = None,
    scenario_names: list[str] | None = None,
):
    config = RegimeSwitchingConfig() if config is None else config
    if output_dir is not None:
        config = replace(config, output_dir=output_dir)

    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)

    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    reduced_model = fit_reduced_state_space(bundle, hank_config, config.partial_config)

    selected_scenarios = config.scenario_specs()
    if scenario_names is not None:
        selected = set(scenario_names)
        selected_scenarios = [scenario for scenario in selected_scenarios if scenario["name"] in selected]

    _save_json(root / "stage5_config.json", config.to_dict())
    _save_json(root / "filter_spec.json", config.filter_spec_payload())
    _save_json(root / "reduced_state_space.json", {
        "state_names": list(reduced_model.state_names),
        "observation_names": list(reduced_model.observation_names),
        "training_summary": reduced_model.training_summary,
        "steady_state_statistics": reduced_model.steady_state_statistics,
    })
    _save_json(root / "scenario_spec.json", selected_scenarios)

    filter_metric_rows = []
    state_metric_frames = []
    policy_metric_rows = []
    policy_path_frames = []
    true_state_frames = []
    filtered_state_frames = []
    observation_frames = []
    model_specs = {}

    for scenario_index, scenario in enumerate(selected_scenarios):
        regime_model = build_regime_switching_model(reduced_model, config, scenario["gap_scale"])
        model_specs[scenario["name"]] = regime_model_spec_payload(regime_model, scenario)

        rng = np.random.default_rng(config.random_seed + 100 * scenario_index)
        hidden_regimes = simulate_hidden_regimes(
            regime_model,
            horizon=config.partial_config.horizon,
            seed=config.random_seed + 1_000 * scenario_index,
        )
        innovations = np.zeros((config.partial_config.horizon, len(regime_model.state_names)), dtype=float)
        for period, regime in enumerate(hidden_regimes):
            innovations[period] = rng.multivariate_normal(
                mean=np.zeros(len(regime_model.state_names), dtype=float),
                cov=regime_model.process_noise_covariances[regime],
            )
        noise_std = {
            name: config.partial_config.base_measurement_noise()[name] * scenario["noise_scale"]
            for name in scenario["noisy_observations"]
        }
        measurement_noise = np.column_stack([
            rng.normal(scale=noise_std[name], size=config.partial_config.horizon)
            for name in scenario["noisy_observations"]
        ])

        full_run = simulate_full_information_policy(
            model=regime_model,
            config=config,
            scenario=scenario,
            hidden_regimes=hidden_regimes,
            innovations=innovations,
            measurement_noise=measurement_noise,
            phi_pi=hank_config.phi_pi,
            phi_y=hank_config.phi_y,
            rho_i=hank_config.rho_i,
        )
        filtered_run = simulate_filtered_policy(
            model=regime_model,
            config=config,
            scenario=scenario,
            hidden_regimes=hidden_regimes,
            innovations=innovations,
            measurement_noise=measurement_noise,
            phi_pi=hank_config.phi_pi,
            phi_y=hank_config.phi_y,
            rho_i=hank_config.rho_i,
        )

        filter_metrics, state_metric_frame = evaluate_regime_filter(filtered_run)
        policy_metrics, policy_paths = evaluate_policy_under_regime_uncertainty(
            filtered_run=filtered_run,
            full_information_run=full_run,
            lambda_y=config.lambda_y,
            lambda_i=config.lambda_i,
        )
        filter_metric_rows.append(filter_metrics)
        state_metric_frames.append(state_metric_frame)
        policy_metric_rows.append(policy_metrics)
        policy_path_frames.append(policy_paths)

        full_true_frame, _ = _state_frames(full_run)
        _, filtered_frame = _state_frames(filtered_run)
        true_state_frames.append(full_true_frame)
        filtered_state_frames.append(filtered_frame)
        observation_frames.append(_observation_frame(filtered_run))

    _save_json(root / "regime_model_spec.json", model_specs)

    filter_metrics = pd.DataFrame(filter_metric_rows)
    state_metrics = pd.concat(state_metric_frames, ignore_index=True) if state_metric_frames else pd.DataFrame()
    policy_metrics = pd.DataFrame(policy_metric_rows)
    policy_paths = pd.concat(policy_path_frames, ignore_index=True) if policy_path_frames else pd.DataFrame()
    true_state_paths = pd.concat(true_state_frames, ignore_index=True) if true_state_frames else pd.DataFrame()
    filtered_state_paths = pd.concat(filtered_state_frames, ignore_index=True) if filtered_state_frames else pd.DataFrame()
    observations = pd.concat(observation_frames, ignore_index=True) if observation_frames else pd.DataFrame()

    filter_metrics.to_csv(root / "filter_metrics.csv", index=False)
    state_metrics.to_csv(root / "filter_state_metrics.csv", index=False)
    policy_metrics.to_csv(root / "policy_metrics.csv", index=False)
    policy_paths.to_csv(root / "policy_paths.csv", index=False)
    true_state_paths.to_csv(root / "true_state_paths.csv", index=False)
    filtered_state_paths.to_csv(root / "filtered_state_paths.csv", index=False)
    observations.to_csv(root / "observations.csv", index=False)

    _write_report(root, config, filter_metrics, policy_metrics)
    return {
        "filter_metrics": filter_metrics,
        "policy_metrics": policy_metrics,
        "policy_paths": policy_paths,
    }
