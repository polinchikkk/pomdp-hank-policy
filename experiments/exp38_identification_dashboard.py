from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from policy.inference import paired_bootstrap_ci, sign_flip_test, summarize_paired_inference  # noqa: E402


DASHBOARD_COLUMNS = (
    "test_name",
    "scientific_question",
    "expected_if_real_channel",
    "expected_if_feature_count_artifact",
    "estimated_mvoi",
    "ci_low",
    "ci_high",
    "sign_flip_p",
    "passed",
)


@dataclass(frozen=True)
class DashboardMetric:
    estimated_mvoi: float
    ci_low: float
    ci_high: float
    sign_flip_p: float
    monotone: bool | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collapse distributional-identification checks into one hypothesis-test dashboard."
    )
    parser.add_argument("--identification-dir", default="outputs/ssj/stochastic/identification_battery")
    parser.add_argument("--null-dir", default="outputs/ssj/stochastic/null_distribution_channel")
    parser.add_argument("--known-dir", default="outputs/ssj/stochastic/known_distribution_channel")
    parser.add_argument("--output-dir", default="outputs/final_protocol")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--mvoi-tol", type=float, default=1e-12)
    parser.add_argument("--bootstrap-reps", type=int, default=4_000)
    parser.add_argument("--sign-flip-reps", type=int, default=4_000)
    args = parser.parse_args()

    identification_dir = Path(args.identification_dir)
    null_dir = Path(args.null_dir)
    known_dir = Path(args.known_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    identification_metrics = _load_identification_metrics(
        identification_dir,
        bootstrap_reps=int(args.bootstrap_reps),
        sign_flip_reps=int(args.sign_flip_reps),
    )
    null_metric, null_pass_extra = _load_null_metric(
        null_dir,
        alpha=float(args.alpha),
        bootstrap_reps=int(args.bootstrap_reps),
        sign_flip_reps=int(args.sign_flip_reps),
    )
    known_metrics = _load_known_metrics(known_dir)

    rows = [
        _row(
            test_name="actual_distribution",
            metric=identification_metrics["actual_distribution"],
            scientific_question=(
                "Do actual MPC, low-liquidity and interest-rate-exposure features add information "
                "about future monetary-transmission losses beyond filtered aggregates?"
            ),
            expected_if_real_channel="Positive MVOI with confidence interval above zero.",
            expected_if_feature_count_artifact="Positive effects should also appear for fake or shuffled features.",
            passed=_positive_effect(identification_metrics["actual_distribution"], alpha=float(args.alpha), tol=float(args.mvoi_tol)),
        ),
        _row(
            test_name="shuffled_distribution",
            metric=identification_metrics["shuffled_distribution"],
            scientific_question="Does the effect survive after distributional features are shuffled away from their economic path?",
            expected_if_real_channel="The MVOI should collapse once the link to the HANK path is broken.",
            expected_if_feature_count_artifact="The MVOI remains positive because only the number of features matters.",
            passed=not _positive_effect(identification_metrics["shuffled_distribution"], alpha=float(args.alpha), tol=float(args.mvoi_tol)),
        ),
        _row(
            test_name="fake_same_autocorr_distribution",
            metric=identification_metrics["fake_same_autocorr_distribution"],
            scientific_question="Can synthetic features with similar persistence and aggregate correlation reproduce the effect?",
            expected_if_real_channel="Synthetic same-autocorrelation features should not deliver positive MVOI.",
            expected_if_feature_count_artifact="Synthetic features deliver a positive MVOI similar to actual distributional features.",
            passed=not _positive_effect(
                identification_metrics["fake_same_autocorr_distribution"],
                alpha=float(args.alpha),
                tol=float(args.mvoi_tol),
            ),
        ),
        _row(
            test_name="lagged_distribution",
            metric=identification_metrics["lagged_distribution"],
            scientific_question="Do stale distributional features retain the same policy value?",
            expected_if_real_channel="Lagging should weaken or remove the positive MVOI.",
            expected_if_feature_count_artifact="The MVOI remains positive because lagged features merely expand the rule.",
            passed=not _positive_effect(identification_metrics["lagged_distribution"], alpha=float(args.alpha), tol=float(args.mvoi_tol)),
        ),
        _row(
            test_name="residualized_distribution",
            metric=identification_metrics["residualized_distribution"],
            scientific_question="Is there distributional information left after projecting out filtered aggregates?",
            expected_if_real_channel="Residualized distributional features still have positive MVOI.",
            expected_if_feature_count_artifact="Residualization removes the apparent effect or leaves only fake-feature effects.",
            passed=_positive_effect(
                identification_metrics["residualized_distribution"],
                alpha=float(args.alpha),
                tol=float(args.mvoi_tol),
            ),
        ),
        _row(
            test_name="no_distribution_channel",
            metric=null_metric,
            scientific_question="Does the pipeline find a distributional value in a null world with no transmission signal?",
            expected_if_real_channel="No positive MVOI and a false-positive rate near the target alpha.",
            expected_if_feature_count_artifact="Positive MVOI appears even when the distributional channel is switched off.",
            passed=(
                not _positive_effect(null_metric, alpha=float(args.alpha), tol=float(args.mvoi_tol))
                and bool(null_pass_extra)
            ),
        ),
        _row(
            test_name="known_distribution_channel_low",
            metric=known_metrics["known_distribution_channel_low"],
            scientific_question="Does the battery recover a deliberately injected weak distributional channel?",
            expected_if_real_channel="MVOI is positive at low injected channel strength.",
            expected_if_feature_count_artifact="MVOI does not track the injected channel strength.",
            passed=_known_pass(known_metrics["known_distribution_channel_low"], alpha=float(args.alpha), tol=float(args.mvoi_tol)),
        ),
        _row(
            test_name="known_distribution_channel_medium",
            metric=known_metrics["known_distribution_channel_medium"],
            scientific_question="Does MVOI keep rising for a medium injected distributional channel?",
            expected_if_real_channel="MVOI is positive and remains monotone along the gamma grid.",
            expected_if_feature_count_artifact="MVOI is flat, erratic, or unrelated to gamma.",
            passed=_known_pass(known_metrics["known_distribution_channel_medium"], alpha=float(args.alpha), tol=float(args.mvoi_tol)),
        ),
        _row(
            test_name="known_distribution_channel_high",
            metric=known_metrics["known_distribution_channel_high"],
            scientific_question="Does the strongest injected distributional channel produce the clearest MVOI?",
            expected_if_real_channel="MVOI is positive and the gamma-grid monotonicity check passes.",
            expected_if_feature_count_artifact="MVOI fails to strengthen with the known channel.",
            passed=_known_pass(known_metrics["known_distribution_channel_high"], alpha=float(args.alpha), tol=float(args.mvoi_tol)),
        ),
    ]
    dashboard = pd.DataFrame(rows, columns=DASHBOARD_COLUMNS)
    dashboard.to_csv(output_dir / "identification_dashboard.csv", index=False)
    print(f"Wrote {output_dir / 'identification_dashboard.csv'}")


def _load_identification_metrics(
    identification_dir: Path,
    *,
    bootstrap_reps: int,
    sign_flip_reps: int,
) -> dict[str, DashboardMetric]:
    summary_path = identification_dir / "identification_battery_summary.csv"
    losses_path = identification_dir / "identification_battery_trajectory_losses.csv"
    _require_file(summary_path)
    _require_file(losses_path)
    summary = pd.read_csv(summary_path)
    losses = pd.read_csv(losses_path)
    metrics: dict[str, DashboardMetric] = {}
    for variant, group in losses.groupby("variant", sort=False):
        delta = group["delta_distribution_minus_aggregates"].to_numpy(dtype=float)
        inference = summarize_paired_inference(
            delta,
            n_boot=bootstrap_reps,
            n_perm=sign_flip_reps,
            seed=3881,
            tie_eps=1e-10,
        )
        metrics[str(variant)] = DashboardMetric(
            estimated_mvoi=float(-inference.mean_delta),
            ci_low=float(-inference.bootstrap_ci_high),
            ci_high=float(-inference.bootstrap_ci_low),
            sign_flip_p=float(inference.sign_flip_p_value),
        )

    for _, row_data in summary.iterrows():
        variant = str(row_data["variant"])
        metrics.setdefault(variant, _metric_from_delta_summary(row_data))

    return {
        "actual_distribution": _require_metric(metrics, "actual_distribution"),
        "fake_same_autocorr_distribution": _require_metric(metrics, "fake_matched_distribution"),
        "lagged_distribution": _require_metric(metrics, "lagged_distribution"),
        "residualized_distribution": _require_metric(metrics, "residualized_distribution"),
        "shuffled_distribution": _shuffled_metric(metrics),
    }


def _load_null_metric(
    null_dir: Path,
    *,
    alpha: float,
    bootstrap_reps: int,
    sign_flip_reps: int,
) -> tuple[DashboardMetric, bool]:
    summary_path = null_dir / "null_distribution_channel_summary.csv"
    replications_path = null_dir / "null_distribution_channel_replications.csv"
    _require_file(summary_path)
    _require_file(replications_path)
    summary = pd.read_csv(summary_path).iloc[0]
    replications = pd.read_csv(replications_path)
    mvoi = replications["loss_reduction"].to_numpy(dtype=float)
    ci_low, ci_high = paired_bootstrap_ci(mvoi, n_boot=bootstrap_reps, seed=3882)
    metric = DashboardMetric(
        estimated_mvoi=float(np.mean(mvoi)),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        sign_flip_p=float(sign_flip_test(mvoi, n_perm=sign_flip_reps, seed=3883)),
    )
    false_positive_rate = float(summary.get("sign_flip_false_positive_rate", math.nan))
    null_pass_extra = math.isfinite(false_positive_rate) and false_positive_rate <= float(alpha)
    return metric, null_pass_extra


def _load_known_metrics(known_dir: Path) -> dict[str, DashboardMetric]:
    summary_path = known_dir / "known_distribution_channel_summary.csv"
    _require_file(summary_path)
    summary = pd.read_csv(summary_path).sort_values("gamma").reset_index(drop=True)
    positive = summary[summary["gamma"].to_numpy(dtype=float) > 0.0].reset_index(drop=True)
    if len(positive) < 3:
        raise ValueError("Known distribution channel summary must contain at least three positive gamma values.")
    selected = {
        "known_distribution_channel_low": positive.iloc[0],
        "known_distribution_channel_medium": positive.iloc[len(positive) // 2],
        "known_distribution_channel_high": positive.iloc[-1],
    }
    return {
        name: _metric_from_delta_summary(row_data, monotone=bool(row_data.get("mvoi_monotone_non_decreasing_so_far", True)))
        for name, row_data in selected.items()
    }


def _metric_from_delta_summary(row_data: pd.Series, *, monotone: bool | None = None) -> DashboardMetric:
    return DashboardMetric(
        estimated_mvoi=float(row_data["loss_reduction"]),
        ci_low=float(-row_data["ci_high"]),
        ci_high=float(-row_data["ci_low"]),
        sign_flip_p=float(row_data.get("sign_flip_p_value", math.nan)),
        monotone=monotone,
    )


def _shuffled_metric(metrics: dict[str, DashboardMetric]) -> DashboardMetric:
    candidates = [
        metric
        for key, metric in metrics.items()
        if key in {"permuted_by_scenario", "permuted_by_time"}
    ]
    if not candidates:
        raise ValueError("Identification battery must contain a shuffled/permuted distribution variant.")
    return max(candidates, key=lambda metric: metric.estimated_mvoi)


def _require_metric(metrics: dict[str, DashboardMetric], key: str) -> DashboardMetric:
    if key not in metrics:
        raise ValueError(f"Identification dashboard is missing required metric: {key}")
    return metrics[key]


def _row(
    *,
    test_name: str,
    scientific_question: str,
    expected_if_real_channel: str,
    expected_if_feature_count_artifact: str,
    metric: DashboardMetric,
    passed: bool,
) -> dict[str, object]:
    return {
        "test_name": test_name,
        "scientific_question": scientific_question,
        "expected_if_real_channel": expected_if_real_channel,
        "expected_if_feature_count_artifact": expected_if_feature_count_artifact,
        "estimated_mvoi": metric.estimated_mvoi,
        "ci_low": metric.ci_low,
        "ci_high": metric.ci_high,
        "sign_flip_p": metric.sign_flip_p,
        "passed": bool(passed),
    }


def _positive_effect(metric: DashboardMetric, *, alpha: float, tol: float) -> bool:
    p_ok = math.isfinite(metric.sign_flip_p) and metric.sign_flip_p < float(alpha)
    return bool(metric.estimated_mvoi > float(tol) and metric.ci_low > 0.0 and p_ok)


def _known_pass(metric: DashboardMetric, *, alpha: float, tol: float) -> bool:
    return bool(_positive_effect(metric, alpha=alpha, tol=tol) and metric.monotone is not False)


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required identification dashboard artifact is missing: {path}")


if __name__ == "__main__":
    main()
