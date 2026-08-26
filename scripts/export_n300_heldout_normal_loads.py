#!/usr/bin/env python3
"""
Export the original N300 held-out normal load profiles for matched OOD tests.

The coordinated-attack stress test should differ from the original experiment
only in attack structure. This script therefore extracts the normal rows already
used as the N300 balanced 80/20 held-out controls and writes their 24-hour total
load profiles for MATLAB.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_severity_conditioned_ml import (  # noqa: E402
    build_audited_train_test,
    load_final_ml_inputs,
)


N_BUSES = 300
N_HOURS = 24
FEATURES_PER_HOUR = 1418
EXPECTED_HELDOUT_NORMALS = 192
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "results"
    / "revision"
    / "ood_stress_test"
    / "n300_heldout_normal_load_profiles.csv"
)


def resolve_project_path(path: Path) -> Path:
    """Interpret relative paths from the project root."""

    return path if path.is_absolute() else PROJECT_ROOT / path


def bus_power_columns_for_hour(hour: int) -> list[str]:
    """
    Return the N300 active-power columns for one OPF hour.

    In the OPF export, every hour starts with the 300 bus active-power decision
    variables. Summing them gives the served load for that solved hour.
    """

    first_column_number = hour * FEATURES_PER_HOUR + 1
    return [f"ft{first_column_number + bus_offset}" for bus_offset in range(N_BUSES)]


def extract_heldout_normal_rows() -> pd.DataFrame:
    """Recreate the original N300 balanced 80/20 split and keep held-out normals."""

    config = load_final_ml_inputs("N300")
    _, test_raw = build_audited_train_test(config)
    heldout_normals = test_raw[test_raw["attack"] == 0].copy()

    if len(heldout_normals) != EXPECTED_HELDOUT_NORMALS:
        raise ValueError(
            f"Expected {EXPECTED_HELDOUT_NORMALS} N300 held-out normal rows, "
            f"found {len(heldout_normals)}."
        )
    if heldout_normals["raw_row_index"].nunique() != len(heldout_normals):
        raise ValueError("Held-out normal rows contain duplicated raw_row_index values.")
    if (heldout_normals["fval"].astype(float) <= -99999).any():
        raise ValueError("Held-out normal rows contain infeasible OPF sentinel values.")

    return heldout_normals.sort_values(["temporal_index", "raw_row_index"]).reset_index(drop=True)


def build_load_profile_table(heldout_normals: pd.DataFrame) -> pd.DataFrame:
    """Convert held-out normal OPF rows into 24-hour total-load profiles."""

    metadata_columns = [
        "raw_row_index",
        "original_ssn",
        "ssn",
        "year",
        "repetition",
        "day_in_season",
        "session_day",
        "temporal_index",
        "fval",
    ]
    missing_metadata = [column for column in metadata_columns if column not in heldout_normals.columns]
    if missing_metadata:
        raise ValueError(f"Held-out normal rows are missing metadata columns: {missing_metadata}")

    profile = heldout_normals[metadata_columns].copy().reset_index(drop=True)
    profile = profile.rename(columns={"fval": "fval_normal"})
    profile.insert(0, "matched_profile_id", range(len(profile)))

    for hour in range(N_HOURS):
        hour_columns = bus_power_columns_for_hour(hour)
        missing_hour_columns = [column for column in hour_columns if column not in heldout_normals.columns]
        if missing_hour_columns:
            preview = ", ".join(missing_hour_columns[:5])
            raise ValueError(f"Missing N300 hour-{hour + 1} bus-power columns: {preview}")

        hour_power = heldout_normals[hour_columns].apply(pd.to_numeric, errors="raise")
        profile[f"load_h{hour + 1:02d}"] = hour_power.sum(axis=1).to_numpy()

    validate_load_profile_table(profile)
    return profile


def validate_load_profile_table(profile: pd.DataFrame) -> None:
    """Check that the exported profiles are complete and season-balanced."""

    load_columns = [f"load_h{hour + 1:02d}" for hour in range(N_HOURS)]
    if profile[load_columns].isna().any().any():
        raise ValueError("Load profile table contains missing hourly loads.")
    if (profile[load_columns] <= 0).any().any():
        raise ValueError("Load profile table contains non-positive hourly loads.")

    season_counts = profile["original_ssn"].astype(int).value_counts().sort_index().to_dict()
    expected_counts = {1: 48, 2: 48, 3: 48, 4: 48}
    if season_counts != expected_counts:
        raise ValueError(f"N300 held-out normal seasons are not balanced: {season_counts}")


def export_load_profiles(output_path: Path) -> pd.DataFrame:
    """Build and save the matched-load profile file used by MATLAB."""

    output_path = resolve_project_path(output_path)
    heldout_normals = extract_heldout_normal_rows()
    profile = build_load_profile_table(heldout_normals)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(output_path, index=False)
    print(f"Saved N300 held-out normal load profiles: {output_path}")
    print(f"profiles: {len(profile)}")
    print("season counts:")
    print(profile["original_ssn"].astype(int).value_counts().sort_index().to_string())
    return profile


def parse_args() -> argparse.Namespace:
    """Read command-line options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where to save the matched N300 held-out normal load profiles.",
    )
    return parser.parse_args()


def main() -> None:
    """Export the matched held-out normal profiles."""

    args = parse_args()
    export_load_profiles(args.output)


if __name__ == "__main__":
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    main()
