#!/usr/bin/env python3
"""
Build compact tables for the manuscript revision.

The detailed analysis CSVs are intentionally rich. This script distills them
into smaller tables that can be read directly during manuscript revision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path("results/revision")
OUTPUT_DIR = INPUT_DIR / "paper_tables"

PAIRED_METRICS_PATH = INPUT_DIR / "paired_impact_metrics.csv"
CLASSIFICATION_PATH = INPUT_DIR / "paired_impact_classification.csv"
THRESHOLDS_PATH = INPUT_DIR / "impact_tail_thresholds_by_system_season.csv"

ZERO_TOLERANCE = 1e-9
ROUND_DECIMALS = 3


def require_columns(df: pd.DataFrame, required: list[str], source_name: str) -> None:
    """Fail early if an upstream table does not contain the expected schema."""

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def percentage(mask: pd.Series | np.ndarray) -> float:
    """Return a percentage on the familiar 0-to-100 scale."""

    return 100.0 * float(np.mean(mask))


def rounded(df: pd.DataFrame) -> pd.DataFrame:
    """Round floating columns for paper-facing CSVs."""

    result = df.copy()
    float_cols = result.select_dtypes(include=["float64", "float32"]).columns
    result[float_cols] = result[float_cols].round(ROUND_DECIMALS)
    return result


def summarize_attack_impacts(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the actual OPF consequences of each attack type.

    This table is the main evidence table: it shows typical impact, tail impact,
    and whether the rest of the fleet moved after one generator was attacked.
    """

    required = [
        "system",
        "attack_type",
        "attack_type_name",
        "delta_cost_abs",
        "dispatch_l1",
        "target_dispatch_change",
        "non_target_dispatch_change",
        "flow_l1",
        "ramp_l1",
    ]
    require_columns(metrics, required, "paired_impact_metrics.csv")

    rows: list[dict[str, float | int | str]] = []
    group_cols = ["system", "attack_type", "attack_type_name"]

    for (system, attack_type, attack_name), group in metrics.groupby(group_cols, sort=True):
        dispatch_positive = group["dispatch_l1"] > ZERO_TOLERANCE
        if dispatch_positive.any():
            non_target_share = (
                group.loc[dispatch_positive, "non_target_dispatch_change"]
                / group.loc[dispatch_positive, "dispatch_l1"]
            )
            median_non_target_share_pct = 100.0 * float(non_target_share.median())
        else:
            median_non_target_share_pct = np.nan

        zero_response = (
            (group["delta_cost_abs"].abs() <= ZERO_TOLERANCE)
            & (group["dispatch_l1"].abs() <= ZERO_TOLERANCE)
            & (group["flow_l1"].abs() <= ZERO_TOLERANCE)
        )

        rows.append(
            {
                "system": system,
                "attack_type": int(attack_type),
                "attack_type_name": attack_name,
                "n_pairs": int(len(group)),
                "zero_response_pct": percentage(zero_response),
                "cost_nonzero_pct": percentage(group["delta_cost_abs"].abs() > ZERO_TOLERANCE),
                "dispatch_nonzero_pct": percentage(group["dispatch_l1"].abs() > ZERO_TOLERANCE),
                "flow_nonzero_pct": percentage(group["flow_l1"].abs() > ZERO_TOLERANCE),
                "median_cost_delta": float(group["delta_cost_abs"].median()),
                "q95_cost_delta": float(group["delta_cost_abs"].quantile(0.95)),
                "median_dispatch_l1": float(group["dispatch_l1"].median()),
                "q95_dispatch_l1": float(group["dispatch_l1"].quantile(0.95)),
                "median_target_dispatch_change": float(group["target_dispatch_change"].median()),
                "median_non_target_dispatch_change": float(group["non_target_dispatch_change"].median()),
                "median_non_target_share_when_dispatch_changes_pct": median_non_target_share_pct,
                "median_flow_l1": float(group["flow_l1"].median()),
                "q95_flow_l1": float(group["flow_l1"].quantile(0.95)),
                "median_ramp_l1": float(group["ramp_l1"].median()),
                "q95_ramp_l1": float(group["ramp_l1"].quantile(0.95)),
            }
        )

    return rounded(pd.DataFrame(rows))


def build_main_paper_impact_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Build the lean impact table intended for the manuscript body.

    Pair counts are intentionally omitted here. They belong in the experimental
    setup text, not in the main result table, because readers may confuse them
    with train/test sample counts.
    """

    detailed = summarize_attack_impacts(metrics)
    columns = [
        "system",
        "attack_type_name",
        "zero_response_pct",
        "median_cost_delta",
        "q95_cost_delta",
        "median_dispatch_l1",
        "median_non_target_dispatch_change",
        "median_flow_l1",
    ]
    table = detailed[columns].copy()
    table = table.rename(
        columns={
            "system": "System",
            "attack_type_name": "Attack Type",
            "zero_response_pct": "Zero-response (%)",
            "median_cost_delta": "Median Cost Delta (model obj.)",
            "q95_cost_delta": "Q95 Cost Delta (model obj.)",
            "median_dispatch_l1": "Median Dispatch Change (p.u.-h)",
            "median_non_target_dispatch_change": "Median Non-target Redispatch (p.u.-h)",
            "median_flow_l1": "Median Flow Change (p.u.-h)",
        }
    )
    table["Attack Type"] = table["Attack Type"].str.replace("_", " ").str.title()
    return table


def summarize_effect_patterns(classified: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize the four interpretability classes for each attack type.

    The classes are based on system-season Q90 thresholds for cost and dispatch:
    low/low, high-cost only, high-dispatch only, and high/high.
    """

    required = ["system", "attack_type", "attack_type_name", "impact_pattern", "zero_response"]
    require_columns(classified, required, "paired_impact_classification.csv")

    pattern_order = [
        "weak_or_nonbinding_attack",
        "economic_impact_attack",
        "hidden_redispatch_attack",
        "critical_attack",
    ]

    rows: list[dict[str, float | int | str]] = []
    group_cols = ["system", "attack_type", "attack_type_name"]

    for (system, attack_type, attack_name), group in classified.groupby(group_cols, sort=True):
        counts = group["impact_pattern"].value_counts(normalize=True)
        row: dict[str, float | int | str] = {
            "system": system,
            "attack_type": int(attack_type),
            "attack_type_name": attack_name,
            "n_pairs": int(len(group)),
            "zero_response_pct": percentage(group["zero_response"]),
        }
        for pattern in pattern_order:
            row[f"{pattern}_pct"] = 100.0 * float(counts.get(pattern, 0.0))
        rows.append(row)

    return rounded(pd.DataFrame(rows))


def summarize_tail_thresholds(thresholds: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the thresholds that will likely be described in the paper.

    Q90 marks high-impact cases. Q95 is a stricter tail marker and is easier to
    defend than adding another composite score.
    """

    required = [
        "system",
        "season",
        "n_pairs",
        "cost_q90",
        "cost_q95",
        "cost_q90_ci_low",
        "cost_q90_ci_high",
        "dispatch_q90",
        "dispatch_q95",
        "dispatch_q90_ci_low",
        "dispatch_q90_ci_high",
    ]
    require_columns(thresholds, required, "impact_tail_thresholds_by_system_season.csv")

    cols = [
        "system",
        "season",
        "n_pairs",
        "cost_q90",
        "cost_q95",
        "cost_q90_ci_low",
        "cost_q90_ci_high",
        "dispatch_q90",
        "dispatch_q95",
        "dispatch_q90_ci_low",
        "dispatch_q90_ci_high",
    ]
    return rounded(thresholds[cols].copy())


def select_case_examples(classified: pd.DataFrame) -> pd.DataFrame:
    """
    Select a few concrete rows for case-study discussion.

    These rows are not a replacement for aggregate statistics. They are useful
    when the manuscript needs one clear example of each impact pattern.
    """

    required = [
        "pair_id",
        "system",
        "season",
        "attack_type_name",
        "target_generator",
        "target_bus_external",
        "delta_cost_abs",
        "dispatch_l1",
        "target_dispatch_change",
        "non_target_dispatch_change",
        "flow_l1",
        "impact_pattern",
    ]
    require_columns(classified, required, "paired_impact_classification.csv")

    examples: list[pd.DataFrame] = []
    selectors = (
        ("critical_attack", "delta_cost_abs"),
        ("economic_impact_attack", "delta_cost_abs"),
        ("hidden_redispatch_attack", "dispatch_l1"),
    )

    for pattern, sort_col in selectors:
        subset = classified[classified["impact_pattern"] == pattern].copy()
        if subset.empty:
            continue
        examples.append(subset.sort_values(sort_col, ascending=False).head(8))

    if not examples:
        return pd.DataFrame(columns=required)

    selected = pd.concat(examples, ignore_index=True)
    return rounded(selected[required])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(PAIRED_METRICS_PATH)
    classified = pd.read_csv(CLASSIFICATION_PATH)
    thresholds = pd.read_csv(THRESHOLDS_PATH)

    attack_impacts = summarize_attack_impacts(metrics)
    effect_patterns = summarize_effect_patterns(classified)
    tail_thresholds = summarize_tail_thresholds(thresholds)
    case_examples = select_case_examples(classified)
    main_impact_table = build_main_paper_impact_table(metrics)

    attack_impacts.to_csv(OUTPUT_DIR / "table_attack_impact_summary.csv", index=False)
    effect_patterns.to_csv(OUTPUT_DIR / "table_attack_effect_patterns.csv", index=False)
    tail_thresholds.to_csv(OUTPUT_DIR / "table_tail_thresholds.csv", index=False)
    case_examples.to_csv(OUTPUT_DIR / "table_case_examples.csv", index=False)
    main_impact_table.to_csv(OUTPUT_DIR / "main_table_paired_impact_summary.csv", index=False)

    print("Wrote:")
    print(f"  {OUTPUT_DIR / 'table_attack_impact_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'table_attack_effect_patterns.csv'}")
    print(f"  {OUTPUT_DIR / 'table_tail_thresholds.csv'}")
    print(f"  {OUTPUT_DIR / 'table_case_examples.csv'}")
    print(f"  {OUTPUT_DIR / 'main_table_paired_impact_summary.csv'}")


if __name__ == "__main__":
    main()
