from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hank_full_baseline.calibration import default_calibration
from hank_full_baseline.steady_state import solve_steady_state
from hank_partial_info_baseline.state_space import fit_reduced_state_space

from .config import RegimeLearningConfig, RegimeLearningVariant
from .evaluation import _is_unstable
from .tuning import (
    _run_single_variant,
    default_universal_candidate_lookup,
    extreme_sticky_regime_config,
)


def _save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _scenario_label_map() -> dict[str, str]:
    return {
        "macro_core_moderate_gap": "Инфляция, выпуск, ставка × умеренный режимный разрыв",
        "macro_core_strong_gap": "Инфляция, выпуск, ставка × сильный режимный разрыв",
        "thin_information_moderate_gap": "Инфляция, ставка × умеренный режимный разрыв",
        "thin_information_strong_gap": "Инфляция, ставка × сильный режимный разрыв",
    }


def _build_variants() -> list[RegimeLearningVariant]:
    labels = _scenario_label_map()
    variants = []
    for scenario_name, scenario_label in labels.items():
        variants.append(
            RegimeLearningVariant(
                name=f"{scenario_name}_belief_state",
                scenario_name=scenario_name,
                scenario_label=scenario_label,
                input_mode="belief_state",
                include_distributional_state=True,
                description="Filtered state plus regime belief as RL input.",
            )
        )
        variants.append(
            RegimeLearningVariant(
                name=f"{scenario_name}_raw_observations",
                scenario_name=scenario_name,
                scenario_label=scenario_label,
                input_mode="raw_observations",
                include_distributional_state=True,
                description="Raw observables plus lagged rate as RL input.",
            )
        )
    return variants


def _compute_seed_win_rates(seed_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario_name, frame in seed_frame.groupby("scenario_name"):
        rows.append(
            {
                "scenario_name": scenario_name,
                "rawobs_vs_classical_win_rate": float(
                    np.mean(frame["rawobs_cumulative_policy_loss"].to_numpy(dtype=float) < frame["classical_cumulative_policy_loss"].to_numpy(dtype=float))
                ),
                "belief_vs_classical_win_rate": float(
                    np.mean(frame["belief_cumulative_policy_loss"].to_numpy(dtype=float) < frame["classical_cumulative_policy_loss"].to_numpy(dtype=float))
                ),
                "rawobs_vs_belief_win_rate": float(
                    np.mean(frame["rawobs_cumulative_policy_loss"].to_numpy(dtype=float) < frame["belief_cumulative_policy_loss"].to_numpy(dtype=float))
                ),
                "num_evaluation_seeds": int(frame["evaluation_seed"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values("scenario_name").reset_index(drop=True)


def _load_existing_variant_results(variant_root: Path) -> dict[str, pd.DataFrame]:
    return {
        "policy_metrics": pd.read_csv(variant_root / "policy_metrics.csv"),
        "selected_policy_summary": pd.read_csv(variant_root / "selected_policy_summary.csv"),
        "training_seed_summary": pd.read_csv(variant_root / "training_seed_summary.csv"),
    }


def run_architecture_ablation(
    *,
    output_dir: str = "outputs/hank_regime_learning_stage6_architecture_ablation",
    variant_names: Iterable[str] | None = None,
    skip_completed: bool = True,
) -> dict[str, pd.DataFrame]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    candidate = default_universal_candidate_lookup()["larger_network"]
    regime_config = extreme_sticky_regime_config()
    config = RegimeLearningConfig(
        output_dir=str(root),
        horizon=60,
        gamma=0.99,
        lambda_y=0.5,
        lambda_i=0.05,
        action_bound=candidate.action_bound,
        classical_policy_mode="switching",
        training_seeds=(11,),
        selection_seeds=(500, 501),
        evaluation_seeds=(900, 901, 902, 903, 904, 905, 906, 907, 908, 909),
        regime_config=regime_config,
        ppo=candidate.ppo,
    )

    hank_config = default_calibration()
    bundle = solve_steady_state(hank_config)
    reduced_model = fit_reduced_state_space(bundle, hank_config, regime_config.partial_config)
    variants = _build_variants()
    requested_variants = set(variant_names) if variant_names is not None else None
    if requested_variants is not None:
        variants = [variant for variant in variants if variant.name in requested_variants]
        if not variants:
            raise ValueError("No architecture-ablation variants matched the requested names.")

    _save_json(root / "architecture_ablation_spec.json", {
        "candidate_name": candidate.name,
        "candidate_description": candidate.description,
        "action_bound": candidate.action_bound,
        "ppo": candidate.ppo.to_dict(),
        "training_seeds": list(config.training_seeds),
        "selection_seeds": list(config.selection_seeds),
        "evaluation_seeds": list(config.evaluation_seeds),
        "classical_policy_mode": config.classical_policy_mode,
        "variants": [variant.to_dict() for variant in variants],
    })

    comparison_rows = []
    seed_rows = []
    policy_metric_frames = []
    policy_path_frames = []
    selected_frames = []
    training_frames = []

    per_variant_results: dict[str, dict[str, pd.DataFrame]] = {}
    for variant in variants:
        variant_root = root / variant.name
        completed = (
            (variant_root / "policy_metrics.csv").exists()
            and (variant_root / "selected_policy_summary.csv").exists()
            and (variant_root / "training_seed_summary.csv").exists()
            and (variant_root / "policy_paths.csv").exists()
            and (variant_root / "policy_comparison.csv").exists()
        )
        if skip_completed and completed:
            results = _load_existing_variant_results(variant_root)
        else:
            results = _run_single_variant(
                root=variant_root,
                config=config,
                variant=variant,
                reduced_model=reduced_model,
                hank_config=hank_config,
            )
        per_variant_results[variant.name] = results
        policy_metrics = results["policy_metrics"].copy()
        policy_metrics.insert(0, "variant_architecture", variant.input_mode)
        policy_metric_frames.append(policy_metrics)
        policy_paths = pd.read_csv(variant_root / "policy_paths.csv")
        policy_paths.insert(0, "variant_architecture", variant.input_mode)
        policy_path_frames.append(policy_paths)
        selected = results["selected_policy_summary"].copy()
        selected.insert(0, "variant_architecture", variant.input_mode)
        selected_frames.append(selected)
        training = results["training_seed_summary"].copy()
        training.insert(0, "variant_architecture", variant.input_mode)
        training_frames.append(training)

    scenario_labels = _scenario_label_map()
    for scenario_name, scenario_label in scenario_labels.items():
        belief_key = f"{scenario_name}_belief_state"
        rawobs_key = f"{scenario_name}_raw_observations"
        if belief_key not in per_variant_results or rawobs_key not in per_variant_results:
            continue
        belief_metrics = per_variant_results[belief_key]["policy_metrics"].copy()
        rawobs_metrics = per_variant_results[rawobs_key]["policy_metrics"].copy()

        belief_pivot = belief_metrics.pivot_table(
            index="evaluation_seed",
            columns="policy_name",
            values="cumulative_policy_loss",
            aggfunc="first",
        )
        rawobs_pivot = rawobs_metrics.pivot_table(
            index="evaluation_seed",
            columns="policy_name",
            values="cumulative_policy_loss",
            aggfunc="first",
        )
        seed_frame = pd.DataFrame({
            "scenario_name": scenario_name,
            "scenario_label": scenario_label,
            "evaluation_seed": belief_pivot.index.to_numpy(dtype=int),
            "classical_cumulative_policy_loss": belief_pivot["classical_filtered_rule"].to_numpy(dtype=float),
            "belief_cumulative_policy_loss": belief_pivot["learning_policy"].to_numpy(dtype=float),
            "rawobs_cumulative_policy_loss": rawobs_pivot["learning_policy"].to_numpy(dtype=float),
            "full_information_cumulative_policy_loss": belief_pivot["full_information_rule"].to_numpy(dtype=float),
        })
        seed_frame["rawobs_minus_classical"] = (
            seed_frame["rawobs_cumulative_policy_loss"] - seed_frame["classical_cumulative_policy_loss"]
        )
        seed_frame["belief_minus_classical"] = (
            seed_frame["belief_cumulative_policy_loss"] - seed_frame["classical_cumulative_policy_loss"]
        )
        seed_frame["rawobs_minus_belief"] = (
            seed_frame["rawobs_cumulative_policy_loss"] - seed_frame["belief_cumulative_policy_loss"]
        )
        seed_rows.append(seed_frame)

        belief_policy = belief_metrics[belief_metrics["policy_name"] == "learning_policy"].copy()
        rawobs_policy = rawobs_metrics[rawobs_metrics["policy_name"] == "learning_policy"].copy()
        classical_policy = belief_metrics[belief_metrics["policy_name"] == "classical_filtered_rule"].copy()
        comparison_rows.append(
            {
                "scenario_name": scenario_name,
                "scenario_label": scenario_label,
                "classical_mean_cumulative_loss": float(classical_policy["cumulative_policy_loss"].mean()),
                "belief_mean_cumulative_loss": float(belief_policy["cumulative_policy_loss"].mean()),
                "rawobs_mean_cumulative_loss": float(rawobs_policy["cumulative_policy_loss"].mean()),
                "belief_minus_classical": float(
                    belief_policy["cumulative_policy_loss"].mean() - classical_policy["cumulative_policy_loss"].mean()
                ),
                "rawobs_minus_classical": float(
                    rawobs_policy["cumulative_policy_loss"].mean() - classical_policy["cumulative_policy_loss"].mean()
                ),
                "rawobs_minus_belief": float(
                    rawobs_policy["cumulative_policy_loss"].mean() - belief_policy["cumulative_policy_loss"].mean()
                ),
                "classical_policy_rate_rmse": float(classical_policy["policy_rate_rmse"].mean()),
                "belief_policy_rate_rmse": float(belief_policy["policy_rate_rmse"].mean()),
                "rawobs_policy_rate_rmse": float(rawobs_policy["policy_rate_rmse"].mean()),
                "classical_policy_volatility": float(classical_policy["policy_instrument_volatility"].mean()),
                "belief_policy_volatility": float(belief_policy["policy_instrument_volatility"].mean()),
                "rawobs_policy_volatility": float(rawobs_policy["policy_instrument_volatility"].mean()),
                "belief_unstable": int(belief_policy["unstable"].max()),
                "rawobs_unstable": int(rawobs_policy["unstable"].max()),
            }
        )

    architecture_comparison = pd.DataFrame(comparison_rows).sort_values("scenario_name").reset_index(drop=True)
    seed_level = pd.concat(seed_rows, ignore_index=True)
    seed_win_rates = _compute_seed_win_rates(seed_level)
    policy_metrics = pd.concat(policy_metric_frames, ignore_index=True)
    policy_paths = pd.concat(policy_path_frames, ignore_index=True)
    selected_summary = pd.concat(selected_frames, ignore_index=True)
    training_summary = pd.concat(training_frames, ignore_index=True)

    architecture_comparison.to_csv(root / "architecture_comparison.csv", index=False)
    seed_level.to_csv(root / "architecture_seed_level.csv", index=False)
    seed_win_rates.to_csv(root / "architecture_seed_win_rates.csv", index=False)
    policy_metrics.to_csv(root / "policy_metrics_all.csv", index=False)
    policy_paths.to_csv(root / "policy_paths_all.csv", index=False)
    selected_summary.to_csv(root / "selected_policy_summary_all.csv", index=False)
    training_summary.to_csv(root / "training_seed_summary_all.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = np.arange(len(architecture_comparison), dtype=float)
    width = 0.25
    ax.bar(positions - width, architecture_comparison["classical_mean_cumulative_loss"], width=width, label="Filter + fixed rule", color="#4e79a7")
    ax.bar(positions, architecture_comparison["belief_mean_cumulative_loss"], width=width, label="Filter + learned rule", color="#59a14f")
    ax.bar(positions + width, architecture_comparison["rawobs_mean_cumulative_loss"], width=width, label="Raw observations + learned rule", color="#f28e2b")
    ax.set_xticks(positions)
    ax.set_xticklabels(architecture_comparison["scenario_label"], rotation=25, ha="right")
    ax.set_ylabel("Средняя накопленная loss")
    ax.set_title("Architecture Ablation в regime-switching HANK")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "fig_architecture_ablation_losses.png", dpi=220)
    fig.savefig(root / "fig_architecture_ablation_losses.pdf")
    plt.close(fig)

    report_lines = [
        "# Stage 6 Architecture Ablation",
        "",
        "На одной и той же 2x2 regime-switching карте сравниваются три policy architectures:",
        "- filter + fixed rule",
        "- filter + learned rule",
        "- raw observations + learned rule",
        "",
    ]
    for row in architecture_comparison.to_dict(orient="records"):
        report_lines.extend(
            [
                f"## {row['scenario_label']}",
                f"- `filter + fixed rule`: {row['classical_mean_cumulative_loss']:.6e}",
                f"- `filter + learned rule`: {row['belief_mean_cumulative_loss']:.6e}",
                f"- `raw observations + learned rule`: {row['rawobs_mean_cumulative_loss']:.6e}",
                f"- Rawobs minus classical: {row['rawobs_minus_classical']:.6e}",
                f"- Rawobs minus belief-state RL: {row['rawobs_minus_belief']:.6e}",
                "",
            ]
        )
    (root / "report_architecture_ablation.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "architecture_comparison": architecture_comparison,
        "seed_level": seed_level,
        "seed_win_rates": seed_win_rates,
        "policy_metrics": policy_metrics,
    }
