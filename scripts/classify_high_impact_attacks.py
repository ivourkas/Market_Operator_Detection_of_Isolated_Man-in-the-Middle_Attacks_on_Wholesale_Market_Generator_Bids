#!/usr/bin/env python3
"""
Classify paired attack outcomes by economic tail risk and physical response.

The input is the paired metric table produced by compute_paired_impact_metrics.py.
Each row already compares one attacked OPF solution against the matching normal
solution for the same system, season, OPF iteration, and generated load profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_PATH = Path("results/revision/paired_impact_metrics.csv")
OUTPUT_DIR = Path("results/revision")

COST_METRIC = "delta_cost_abs"
DISPATCH_METRIC = "dispatch_l1"
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_RANDOM_SEED = 20260813
ZERO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Thresholds:
    """Season-specific reference values used to interpret one attack outcome."""

    cost_mean: float
    cost_std: float
    cost_q90: float
    cost_q95: float
    cost_mean_ci_low: float
    cost_mean_ci_high: float
    cost_q90_ci_low: float
    cost_q90_ci_high: float
    dispatch_mean: float
    dispatch_std: float
    dispatch_q90: float
    dispatch_q95: float
    dispatch_mean_ci_low: float
    dispatch_mean_ci_high: float
    dispatch_q90_ci_low: float
    dispatch_q90_ci_high: float


def bootstrap_interval(
    values: np.ndarray,
    statistic: str,
    rng: np.random.Generator,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float, float]:
    """
    Estimate a 95% bootstrap interval for a mean or 90th percentile.

    The interval is descriptive: it tells us how stable the seasonal reference
    value is under resampling of the available paired OPF outcomes.
    """

    clean_values = values[np.isfinite(values)]
    if len(clean_values) == 0:
        return np.nan, np.nan

    if len(clean_values) == 1:
        value = float(clean_values[0])
        return value, value

    samples = rng.choice(clean_values, size=(iterations, len(clean_values)), replace=True)
    if statistic == "mean":
        boot_stats = samples.mean(axis=1)
    elif statistic == "q90":
        boot_stats = np.quantile(samples, 0.90, axis=1)
    else:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")

    low, high = np.quantile(boot_stats, [0.025, 0.975])
    return float(low), float(high)


def compute_thresholds(group: pd.DataFrame, rng: np.random.Generator) -> Thresholds:
    """
    Build one system-season reference distribution.

    Cost and dispatch receive separate thresholds because an attack can be costly
    without changing dispatch much, or physically disruptive without being one of
    the most expensive cases.
    """

    cost = group[COST_METRIC].to_numpy(dtype=float)
    dispatch = group[DISPATCH_METRIC].to_numpy(dtype=float)

    cost_mean_ci = bootstrap_interval(cost, "mean", rng)
    cost_q90_ci = bootstrap_interval(cost, "q90", rng)
    dispatch_mean_ci = bootstrap_interval(dispatch, "mean", rng)
    dispatch_q90_ci = bootstrap_interval(dispatch, "q90", rng)

    return Thresholds(
        cost_mean=float(np.mean(cost)),
        cost_std=float(np.std(cost, ddof=1)) if len(cost) > 1 else 0.0,
        cost_q90=float(np.quantile(cost, 0.90)),
        cost_q95=float(np.quantile(cost, 0.95)),
        cost_mean_ci_low=cost_mean_ci[0],
        cost_mean_ci_high=cost_mean_ci[1],
        cost_q90_ci_low=cost_q90_ci[0],
        cost_q90_ci_high=cost_q90_ci[1],
        dispatch_mean=float(np.mean(dispatch)),
        dispatch_std=float(np.std(dispatch, ddof=1)) if len(dispatch) > 1 else 0.0,
        dispatch_q90=float(np.quantile(dispatch, 0.90)),
        dispatch_q95=float(np.quantile(dispatch, 0.95)),
        dispatch_mean_ci_low=dispatch_mean_ci[0],
        dispatch_mean_ci_high=dispatch_mean_ci[1],
        dispatch_q90_ci_low=dispatch_q90_ci[0],
        dispatch_q90_ci_high=dispatch_q90_ci[1],
    )


def build_threshold_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compute seasonal cost and dispatch reference values for every system."""

    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    rows: list[dict[str, float | int | str]] = []

    for (system, season), group in metrics.groupby(["system", "season"], sort=True):
        thresholds = compute_thresholds(group, rng)
        rows.append(
            {
                "system": system,
                "season": int(season),
                "n_pairs": int(len(group)),
                "cost_mean": thresholds.cost_mean,
                "cost_std": thresholds.cost_std,
                "cost_q90": thresholds.cost_q90,
                "cost_q95": thresholds.cost_q95,
                "cost_manual_review_threshold": thresholds.cost_q90 + thresholds.cost_std,
                "cost_mean_ci_low": thresholds.cost_mean_ci_low,
                "cost_mean_ci_high": thresholds.cost_mean_ci_high,
                "cost_q90_ci_low": thresholds.cost_q90_ci_low,
                "cost_q90_ci_high": thresholds.cost_q90_ci_high,
                "dispatch_mean": thresholds.dispatch_mean,
                "dispatch_std": thresholds.dispatch_std,
                "dispatch_q90": thresholds.dispatch_q90,
                "dispatch_q95": thresholds.dispatch_q95,
                "dispatch_manual_review_threshold": thresholds.dispatch_q90 + thresholds.dispatch_std,
                "dispatch_mean_ci_low": thresholds.dispatch_mean_ci_low,
                "dispatch_mean_ci_high": thresholds.dispatch_mean_ci_high,
                "dispatch_q90_ci_low": thresholds.dispatch_q90_ci_low,
                "dispatch_q90_ci_high": thresholds.dispatch_q90_ci_high,
            }
        )

    return pd.DataFrame(rows)


def classify_cost_severity(row: pd.Series) -> str:
    """Convert a cost delta into a simple triage label."""

    if row[COST_METRIC] >= row["cost_manual_review_threshold"]:
        return "manual_review"
    if row[COST_METRIC] >= row["cost_q90"]:
        return "high_cost"
    return "ordinary"


def classify_dispatch_severity(row: pd.Series) -> str:
    """Convert a dispatch delta into a simple physical-response label."""

    if row[DISPATCH_METRIC] >= row["dispatch_manual_review_threshold"]:
        return "manual_review"
    if row[DISPATCH_METRIC] >= row["dispatch_q90"]:
        return "high_dispatch"
    return "ordinary"


def classify_impact_pattern(row: pd.Series) -> str:
    """
    Label the shape of the attack, not just its size.

    The four labels separate economic and operational consequences. This matters
    because two attacks can have similar costs while moving the fleet very
    differently.
    """

    high_cost = row[COST_METRIC] >= row["cost_q90"]
    high_dispatch = row[DISPATCH_METRIC] >= row["dispatch_q90"]

    if high_cost and high_dispatch:
        return "critical_attack"
    if high_cost and not high_dispatch:
        return "economic_impact_attack"
    if not high_cost and high_dispatch:
        return "hidden_redispatch_attack"
    return "weak_or_nonbinding_attack"


def classify_zero_response(row: pd.Series) -> bool:
    """
    Flag attacks that changed the input scenario but left the OPF solution unchanged.

    This is not a detector failure by itself. It means the attacked parameter did
    not become binding for that load profile and season.
    """

    return bool(
        row["delta_cost_abs"] <= ZERO_TOLERANCE
        and row["dispatch_l1"] <= ZERO_TOLERANCE
        and row["flow_l1"] <= ZERO_TOLERANCE
    )


def classify_rows(metrics: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Attach seasonal thresholds and row-level labels to every paired attack."""

    merged = metrics.merge(thresholds, on=["system", "season"], how="left", validate="many_to_one")

    if merged["cost_q90"].isna().any() or merged["dispatch_q90"].isna().any():
        raise ValueError("Some paired rows did not receive threshold values.")

    merged["cost_severity"] = merged.apply(classify_cost_severity, axis=1)
    merged["dispatch_severity"] = merged.apply(classify_dispatch_severity, axis=1)
    merged["impact_pattern"] = merged.apply(classify_impact_pattern, axis=1)
    merged["zero_response"] = merged.apply(classify_zero_response, axis=1)

    return merged


def summarize_classifications(classified: pd.DataFrame) -> pd.DataFrame:
    """Summarize how often each attack type falls into each impact pattern."""

    rows: list[dict[str, float | int | str]] = []

    group_cols = ["system", "attack_type", "attack_type_name"]
    for (system, attack_type, attack_type_name), group in classified.groupby(group_cols, sort=True):
        pattern_counts = group["impact_pattern"].value_counts(normalize=True)
        cost_counts = group["cost_severity"].value_counts(normalize=True)
        dispatch_counts = group["dispatch_severity"].value_counts(normalize=True)

        rows.append(
            {
                "system": system,
                "attack_type": int(attack_type),
                "attack_type_name": attack_type_name,
                "n_pairs": int(len(group)),
                "zero_response_pct": 100.0 * float(group["zero_response"].mean()),
                "ordinary_cost_pct": 100.0 * float(cost_counts.get("ordinary", 0.0)),
                "high_cost_pct": 100.0 * float(cost_counts.get("high_cost", 0.0)),
                "manual_review_cost_pct": 100.0 * float(cost_counts.get("manual_review", 0.0)),
                "ordinary_dispatch_pct": 100.0 * float(dispatch_counts.get("ordinary", 0.0)),
                "high_dispatch_pct": 100.0 * float(dispatch_counts.get("high_dispatch", 0.0)),
                "manual_review_dispatch_pct": 100.0 * float(dispatch_counts.get("manual_review", 0.0)),
                "weak_or_nonbinding_attack_pct": 100.0
                * float(pattern_counts.get("weak_or_nonbinding_attack", 0.0)),
                "economic_impact_attack_pct": 100.0
                * float(pattern_counts.get("economic_impact_attack", 0.0)),
                "hidden_redispatch_attack_pct": 100.0
                * float(pattern_counts.get("hidden_redispatch_attack", 0.0)),
                "critical_attack_pct": 100.0 * float(pattern_counts.get("critical_attack", 0.0)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(INPUT_PATH)
    thresholds = build_threshold_table(metrics)
    classified = classify_rows(metrics, thresholds)
    summary = summarize_classifications(classified)

    thresholds.to_csv(OUTPUT_DIR / "impact_tail_thresholds_by_system_season.csv", index=False)
    classified.to_csv(OUTPUT_DIR / "paired_impact_classification.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "impact_classification_by_attack_type.csv", index=False)

    print("Wrote:")
    print(f"  {OUTPUT_DIR / 'impact_tail_thresholds_by_system_season.csv'}")
    print(f"  {OUTPUT_DIR / 'paired_impact_classification.csv'}")
    print(f"  {OUTPUT_DIR / 'impact_classification_by_attack_type.csv'}")


if __name__ == "__main__":
    main()
